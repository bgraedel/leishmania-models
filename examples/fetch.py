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


STAMP = ".sha256"  # what an unpacked directory was unpacked from


def fetch(model_id: str, version: str | None = None,
          source: str | None = None) -> tuple[Path, dict]:
    """-> (weights, entry). The weights are a file, or a directory for an archive.

    What is already cached is checked against the digest the index records, not merely
    found. The cache is keyed on the asset's FILE NAME, so a model republished under
    the same name -- which is exactly what a v2 usually is -- lands on the v1 already
    sitting there, and trusting the name alone runs the old weights and says nothing.
    A file that does not match is thrown away and fetched again.

    Re-hashing costs a fraction of a second per hundred megabytes. An unpacked archive
    cannot be hashed, so it carries a stamp of what it came from instead.
    """
    entry = resolve(model_id, version, source)
    want = entry.get("sha256")
    if not want:
        raise SystemExit(f"{model_id} {entry.get('version', '')}: the index records no "
                         f"sha256 for it, so there is nothing to verify it against")
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
        # Unpacked by a version of this file that left no stamp. Nothing says what it
        # holds, so it is worth no more than an unverified download.
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


IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp")

_SEQUENCES: dict = {}
_SIZES: dict = {}


def _pages(file: Path) -> tuple:
    """(how many frames one file holds, how big they are). A non-stack holds one.

    Counted as pages rather than off the series shape, because pages are exactly what
    `tifffile.imread(key=...)` indexes and so the only count that cannot disagree with
    what `load_image` can then read. A series shape says `(T, C, Y, X)` for a file
    whose pages number T x C, and asking for frame T-1 of it walked off the end.

    The size comes from the header, which is already open, so a folder can be checked
    for agreement before anything is run over it rather than after.
    """
    if file.suffix.lower() not in (".tif", ".tiff"):
        from PIL import Image

        with Image.open(file) as handle:  # lazy: this reads the header, not the pixels
            return 1, (handle.height, handle.width)
    import tifffile

    with tifffile.TiffFile(file) as handle:
        count = len(handle.pages)
        size = tuple(handle.pages[0].shape[:2]) if count else (0, 0)
        # Counting pages is deliberate, but on a multi-channel file it means frame 1 is
        # the second CHANNEL of the first time point rather than the second time point,
        # and that is not something to leave anyone to discover from the output. The
        # axes say so and are free to read, so they are read.
        series = handle.series[0] if handle.series else None
        axes = getattr(series, "axes", "") or ""
        if "C" in axes and count > 1:
            channels = int(series.shape[axes.index("C")])
            if channels > 1:
                print(f"note: {file.name} is {axes}, {channels} channels x "
                      f"{count // channels} time point(s). --frames counts PAGES, so "
                      f"frame 1 is the second channel of the first time point, not the "
                      f"second time point.")
        return count, size


def frames_in(path) -> list:
    """Every frame the input offers, as (file, index within that file) pairs.

    A file is its own pages. **A folder is every image in it, in name order, joined
    into one sequence** -- so `--frames`, the T axis the outputs grow and the progress
    all count across the folder exactly as they count across a stack, and a folder of
    stacks is simply the pages of each in turn.

    The answer is worked out once per path: it costs a header read per file, which is
    not something to repeat for every frame of a long run.
    """
    path = Path(path)
    cached = _SEQUENCES.get(str(path))
    if cached is not None:
        return cached
    if path.is_dir():
        files = sorted(item for item in path.iterdir()
                       if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise SystemExit(f"{path}: no images here "
                             f"(looking for {', '.join(IMAGE_SUFFIXES)})")
    elif path.is_file():
        files = [path]
    else:
        raise SystemExit(f"{path}: no such file or folder")
    counted = [(file, *_pages(file)) for file in files]
    # Checked here, where it costs the header read that has just been done anyway, and
    # not when the first frame is handed to a writer: one stack cannot hold two sizes,
    # and finding that out at the last frame of a folder throws away the whole run.
    frames = [(file, index) for file, count, _ in counted for index in range(count)]
    _SEQUENCES[str(path)] = frames
    _SIZES[str(path)] = [size for _, count, size in counted for _ in range(count)]
    return frames


def frame_sizes(path) -> list:
    """The (height, width) of every frame the input offers, in the same order.

    Read from the headers, which `frames_in` has opened already, so this costs nothing
    beyond the walk it has done. It is what lets the outputs be grouped by size before
    a single frame has been run, rather than a folder of two sizes failing partway
    through on a stack that cannot hold the second one.
    """
    frames_in(path)
    return _SIZES[str(Path(path))]


def load_image(path, frame: int = 0) -> np.ndarray:
    """One frame as (H, W, 3) uint8, stretched the way the training data was.

    `frame` counts across whatever the input is: the pages of a stack, or every image
    in a folder in name order. See `frames_in`.

    These models were trained on 8-bit frames whose 0.05 and 99.95 percentiles had
    been mapped to 0 and 255. Microscopy frames are uint16 and mostly background,
    so dividing by 256 instead leaves a flat grey image with almost nothing in it
    for the model to find. Grey is repeated across three channels, which is what
    the networks take.
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
        # A colour frame, whose channels are the same image. WHICH axis they are on has
        # to be established rather than assumed: a planar (C, H, W) page read as
        # (H, W, C) has its WIDTH averaged away instead, and what comes out is a 3-row
        # image that is still 2-D, so nothing below notices and the run continues on it.
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
