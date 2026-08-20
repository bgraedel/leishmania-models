#!/usr/bin/env python
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy",
#     "pillow",
#     "tifffile",
# ]
# ///
"""Get a model out of the index, and an image into the shape it was trained on.

    from fetch import fetch, load_image

    weights, entry = fetch("leishmania-pose")        # the newest version
    weights, entry = fetch("leishmania-pose", "v1")  # a particular one

`entry` is that model's index entry, and it carries the settings. The examples
read `imgsz` and the preprocessing out of it rather than repeating them, so a
republished model changes what they do without any of them being edited.

Weights land in ~/.cache/leishmania-models, are checked against the digest the
index records, and an archive is unpacked. Later calls are cache hits and never
reach the network. LEISHMANIA_MODELS_CACHE puts them somewhere else, and
LEISHMANIA_MODELS_INDEX reads a local index.json instead of the published one --
`export LEISHMANIA_MODELS_INDEX=index.json` from a clone.

    python fetch.py leishmania-m2f-1024      # download only, print the path
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

# The index is a release asset, so this URL always names the current set.
INDEX_URL = "https://github.com/bgraedel/leishmania-models/releases/latest/download/index.json"
CACHE = Path(os.environ.get("LEISHMANIA_MODELS_CACHE",
                            Path.home() / ".cache" / "leishmania-models"))
ARCHIVES = (".zip", ".tar.gz", ".tgz")


def load_index(source: str | None = None) -> dict:
    source = source or os.environ.get("LEISHMANIA_MODELS_INDEX") or INDEX_URL
    # A scheme decides it, not whether the path happens to exist: "https://..." is not
    # a filename on any platform, and asking the filesystem about it on Windows is a
    # question about an invalid path rather than about a URL.
    if "://" not in source:
        return json.loads(Path(source).read_text(encoding="utf-8"))
    with urllib.request.urlopen(source) as response:
        return json.loads(response.read())


def resolve(model_id: str, version: str | None = None, source: str | None = None) -> dict:
    """The index entry for a model, newest version unless one is named."""
    models = [m for m in load_index(source)["models"] if m["id"] == model_id]
    if not models:
        known = sorted({m["id"] for m in load_index(source)["models"]})
        raise SystemExit(f"{model_id} is not in the index. There is: {', '.join(known)}")
    if version:
        for entry in models:
            if entry["version"] == version:
                return entry
        have = ", ".join(m["version"] for m in models)
        raise SystemExit(f"{model_id} has no {version}. There is: {have}")
    # Shorter strings first, so v9 sorts below v10 rather than above it.
    return max(models, key=lambda m: (len(m["version"]), m["version"]))


def fetch(model_id: str, version: str | None = None,
          source: str | None = None) -> tuple[Path, dict]:
    """-> (weights, entry). The weights are a file, or a directory for an archive."""
    entry = resolve(model_id, version, source)
    name = entry["url"].rsplit("/", 1)[-1]
    stem = _archive_stem(name)
    CACHE.mkdir(parents=True, exist_ok=True)

    if stem and (CACHE / stem).is_dir():
        return CACHE / stem, entry
    archive = CACHE / name
    if not archive.exists():
        _download(entry, archive)
    if stem:
        _unpack(archive, CACHE / stem)
        return CACHE / stem, entry
    return archive, entry


def _archive_stem(name: str) -> str | None:
    for suffix in ARCHIVES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def _download(entry: dict, path: Path) -> None:
    # Written to .part and renamed only once the digest matches, so a file that is
    # there at all is a file that was verified. A Ctrl-C leaves no half-download to
    # be trusted on the next run.
    part = path.with_name(path.name + ".part")
    sha = hashlib.sha256()
    done = 0
    total = entry.get("size", 0)
    print(f"downloading {path.name} ({total / 1e6:.0f} MB)")
    # A "certificate verify failed" here is almost always macOS with a python.org
    # build whose root certificates were never installed: run
    # /Applications/Python 3.x/Install Certificates.command once.
    with urllib.request.urlopen(entry["url"]) as response, part.open("wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
            sha.update(block)
            done += len(block)
            if total:
                print(f"\r  {100 * done / total:3.0f}%", end="", file=sys.stderr)
    print("", file=sys.stderr)
    if sha.hexdigest() != entry["sha256"]:
        part.unlink()
        raise SystemExit(f"{path.name}: sha256 is not what the index records; refusing it")
    part.replace(path)


def _unpack(archive: Path, dest: Path) -> None:
    print(f"unpacking {archive.name}")
    staging = dest.with_name(dest.name + ".part")
    shutil.rmtree(staging, ignore_errors=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(staging)
    else:
        with tarfile.open(archive) as bundle:
            # "data" refuses absolute paths, ..\ escapes and device files. It is the
            # default from Python 3.14 and a warning before that; naming it keeps the
            # behaviour the same on every version rather than shifting under us.
            if hasattr(tarfile, "data_filter"):
                bundle.extractall(staging, filter="data")
            else:
                bundle.extractall(staging)
    staging.replace(dest)


def load_image(path, frame: int = 0) -> np.ndarray:
    """One frame as (H, W, 3) uint8, stretched the way the training data was.

    These models were trained on 8-bit frames whose 0.05 and 99.95 percentiles had
    been mapped to 0 and 255. Microscopy frames are uint16 and mostly background,
    so dividing by 256 instead leaves a flat grey image with almost nothing in it
    for the model to find. Grey is repeated across three channels, which is what
    the networks take.
    """
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        pixels = tifffile.imread(path, key=frame)
    else:
        from PIL import Image
        pixels = np.array(Image.open(path))
    pixels = np.squeeze(pixels)
    if pixels.ndim == 3:  # a colour frame; the channels are the same image
        pixels = pixels[..., :3].mean(-1)
    if pixels.ndim != 2:
        raise SystemExit(f"{path.name}: expected one 2-D frame, got {pixels.shape}")
    low, high = np.percentile(pixels, [0.05, 99.95])
    grey = (np.clip((pixels - low) / (high - low + 1e-9), 0, 1) * 255).astype(np.uint8)
    return np.stack([grey] * 3, -1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    weights, entry = fetch(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"{entry['id']} {entry['version']}  {entry['framework']} {entry['task']}")
    print(weights)
