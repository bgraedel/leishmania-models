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

`entry` is that model's index entry and carries its settings: the examples read `imgsz`
and the preprocessing out of it.

Weights land in ~/.cache/leishmania-models, are verified against the sha256 the index
records, and archives are unpacked; later calls are cache hits. LEISHMANIA_MODELS_CACHE
puts them elsewhere, LEISHMANIA_MODELS_INDEX reads a local index.json --
`export LEISHMANIA_MODELS_INDEX=index.json` from a clone.

    python fetch.py leishmania-m2f-1024      # download only, print the path
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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
    # Decided by the scheme: asking the filesystem about "https://..." is a question
    # about an invalid path on Windows.
    if "://" not in source:
        return json.loads(Path(source).read_text(encoding="utf-8"))
    with urllib.request.urlopen(source) as response:
        return json.loads(response.read())


def newest(models: list) -> dict:
    """Which of one model's entries is the current one.

    Shorter strings first, so v9 sorts below v10. gui.py calls it too, so the form and
    the run cannot disagree.
    """
    return max(models, key=lambda m: (len(m["version"]), m["version"]))


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
    return newest(models)


STAMP = ".sha256"  # what an unpacked directory was unpacked from


def fetch(model_id: str, version: str | None = None,
          source: str | None = None) -> tuple[Path, dict]:
    """-> (weights, entry). The weights are a file, or a directory for an archive.

    The cache is keyed on the asset's FILE NAME, so what is already there is verified
    against the sha256 the index records and a model republished under the same name is
    not served stale; a mismatch is thrown away and fetched again. An unpacked archive
    cannot be hashed, so it carries a stamp of what it came from.
    """
    entry = resolve(model_id, version, source)
    want = entry.get("sha256")
    if not want:
        raise SystemExit(f"{model_id} {entry.get('version', '')}: the index records no "
                         f"sha256, so there is nothing to verify it against")
    name = entry["url"].rsplit("/", 1)[-1]
    stem = _archive_stem(name)
    CACHE.mkdir(parents=True, exist_ok=True)

    if stem and (CACHE / stem / STAMP).is_file():
        if (CACHE / stem / STAMP).read_text(encoding="utf-8").strip() == want:
            return CACHE / stem, entry
        print(f"{stem}: unpacked from something else than the index now records; "
              f"fetching again")
        shutil.rmtree(CACHE / stem, ignore_errors=True)
    elif stem and (CACHE / stem).is_dir():
        # Unpacked by a version that left no stamp: worth no more than an unverified
        # download.
        shutil.rmtree(CACHE / stem, ignore_errors=True)

    archive = CACHE / name
    if archive.exists() and digest(archive) != want:
        print(f"{name}: what is cached is not what the index records; fetching again")
        archive.unlink()
    if not archive.exists():
        _download(entry, archive)
    if stem:
        _unpack(archive, CACHE / stem)
        (CACHE / stem / STAMP).write_text(want, encoding="utf-8")
        return CACHE / stem, entry
    return archive, entry


def on_disk(path) -> tuple[Path, dict]:
    """A checkpoint that is not in the index: (its path, an empty entry).

    What `--weights` hands the scripts. The entry is empty, so tile size, input size and
    keypoint chain all fall back to their defaults.
    """
    weights = Path(path)
    if not weights.exists():
        raise SystemExit(f"--weights {path}: no such file or folder")
    return weights, {}


def digest(path: Path) -> str:
    """The sha256 of a file that is already on disk."""
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _archive_stem(name: str) -> str | None:
    for suffix in ARCHIVES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def _download(entry: dict, path: Path) -> None:
    # Written to .part and renamed only once the digest matches, so a Ctrl-C leaves no
    # half-download to be trusted on the next run.
    part = path.with_name(path.name + ".part")
    sha = hashlib.sha256()
    done = 0
    total = entry.get("size", 0)
    print(f"downloading {path.name} ({total / 1e6:.0f} MB)")
    # "certificate verify failed" here is usually a python.org build on macOS: run
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
            # "data" refuses absolute paths, ..\ escapes and device files; named so
            # every Python version behaves the same.
            if hasattr(tarfile, "data_filter"):
                bundle.extractall(staging, filter="data")
            else:
                bundle.extractall(staging)
    staging.replace(dest)


IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp")

_SEQUENCES: dict = {}
_SIZES: dict = {}


def _pages(file: Path) -> tuple:
    """(how many frames one file holds, how big they are). A non-stack holds one.

    Counted as PAGES, which is what `tifffile.imread(key=...)` indexes and so what
    `load_image` can read; a series of shape (T, C, Y, X) has T x C pages. The size comes
    from the header, so a folder can be checked before anything runs over it.
    """
    if file.suffix.lower() not in (".tif", ".tiff"):
        from PIL import Image

        with Image.open(file) as handle:  # lazy: reads the header, not the pixels
            return 1, (handle.height, handle.width)
    import tifffile

    with tifffile.TiffFile(file) as handle:
        count = len(handle.pages)
        size = tuple(handle.pages[0].shape[:2]) if count else (0, 0)
        # On a multi-channel file, page 1 is the second CHANNEL of the first time point;
        # the axes say so and are free to read, so say it out loud.
        series = handle.series[0] if handle.series else None
        axes = getattr(series, "axes", "") or ""
        if "C" in axes and count > 1:
            channels = int(series.shape[axes.index("C")])
            if channels > 1:
                print(f"note: {file.name} is {axes}, {channels} channels x "
                      f"{count // channels} time point(s). --frames counts PAGES, so "
                      f"frame 1 is the second channel, not the second time point.")
        return count, size


def name_order(path: Path):
    """Sort key for a folder's files: the order a file browser shows them.

    Numbers by value, so frame_2 comes before frame_10, and case ignored, so Frame_3 sits
    between them. Plain `sorted` puts frame_10 first and every capital before every
    lower-case letter, and then the T axis is not the order the frames were taken in.
    """
    parts = re.split(r"(\d+)", path.name)  # str, digits, str, ..., str: types line up
    return [int(part) if part.isdigit() else part.lower() for part in parts], path.name


def images_in(folder) -> list:
    """The image files in a folder, in name order (see `name_order`). Non-images are
    passed over; a folder with none is refused."""
    folder = Path(folder)
    files = sorted((item for item in folder.iterdir()
                    if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
                   key=name_order)
    if not files:
        raise SystemExit(f"{folder}: no images here "
                         f"(looking for {', '.join(IMAGE_SUFFIXES)})")
    return files


def frames_in(path) -> list:
    """Every frame the input offers, as (file, index within that file) pairs.

    A file is its own pages. A folder given here is every image in it, in name order,
    joined into one sequence -- what `--sequence` asks for; `outputs.runs_of` is what
    makes a folder one run per image, and hands the files here one at a time. Worked
    out once per path, since it costs a header read per file.
    """
    path = Path(path)
    cached = _SEQUENCES.get(str(path))
    if cached is not None:
        return cached
    if path.is_dir():
        files = images_in(path)
    elif path.is_file():
        files = [path]
    else:
        raise SystemExit(f"{path}: no such file or folder")
    counted = [(file, *_pages(file)) for file in files]
    # Sizes are recorded off the header read just done, since one stack cannot hold two
    # of them and finding that out at the last frame throws away the whole run.
    frames = [(file, index) for file, count, _ in counted for index in range(count)]
    _SEQUENCES[str(path)] = frames
    _SIZES[str(path)] = [size for _, count, size in counted for _ in range(count)]
    return frames


def frame_sizes(path) -> list:
    """The (height, width) of every frame the input offers, in the same order.

    Read from the headers `frames_in` has already opened, so the outputs can be grouped
    by size before a single frame is run.
    """
    frames_in(path)
    return _SIZES[str(Path(path))]


def load_image(path, frame: int = 0) -> np.ndarray:
    """One frame as (H, W, 3) uint8, stretched the way the training data was.

    `frame` counts across the whole input: a stack's pages, or a folder's images in name
    order. See `frames_in`. The 0.05 and 99.95 percentiles are mapped to 0 and 255 --
    dividing uint16 by 256 instead leaves a flat grey image -- and the grey is repeated
    across three channels.
    """
    sequence = frames_in(path)
    if not 0 <= frame < len(sequence):
        raise SystemExit(f"frame {frame}: this input has {len(sequence)}")
    path, page = sequence[frame]
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        pixels = tifffile.imread(path, key=page)
    else:
        from PIL import Image
        pixels = np.array(Image.open(path))
    pixels = np.squeeze(pixels)
    if pixels.ndim == 3:
        # Check which axis the channels are on: a planar (C, H, W) page read as
        # (H, W, C) has its width averaged away into a 3-row image nothing notices.
        if pixels.shape[-1] <= 4:
            pixels = pixels[..., :3].mean(-1)
        elif pixels.shape[0] <= 4:
            pixels = pixels[:3].mean(0)
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
