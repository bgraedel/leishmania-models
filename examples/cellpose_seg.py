#!/usr/bin/env python
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "cellpose>=4.0.1",
#     # --guide places tiles with the detector, an ultralytics checkpoint.
#     "ultralytics>=8.4.142",
#     "numpy",
#     "scipy",
#     "pillow",
#     "tifffile",
#     "roifile",
#     "opencv-python-headless",
#     "tqdm",
# ]
# ///
"""leishmania-cellpose-{body,animal,flagellum}: cellpose 4, one mask per cell.

    python cellpose_seg.py cells.tif                                # body, the default
    python cellpose_seg.py cells.tif --id leishmania-cellpose-animal --frames 0-99
    python cellpose_seg.py cells.tif --niter 600 --flow-threshold 0.6 --rois masks.zip

Cellpose 4 syntax: `CellposeModel` only -- no `Cellpose` wrapper, no size model, no
`channels`; the network takes three channels in any order and `diameter` is the only
thing that rescales a frame.

The default whole-frame pass is the right one here. Cellpose blocks the frame up itself
-- 256 px at a tenth of overlap -- and averages the FLOWS where blocks meet before
running the dynamics once over the whole frame, so cells are already at 1.0x and
nothing is cut. `--tile` here is not about scale: it is for a frame too large to hold
at once, and for `--guide`, which makes an instance number mean the same cell it means
in the other models' outputs.

The dials, in the order they matter for a long thin cell:

  --niter               how far the dynamics walk (default 200 at native scale). It is
                        a path length, and a promastigote with its flagellum is a few
                        hundred px of it; a flagellum whose pixels never reach the
                        cell's centre comes back separately or not at all. Raise first.
  --flow-threshold      a mask whose flows disagree with the model by more than this is
                        discarded (default 0.4). Elongated cells score badly here; 0
                        turns the check off.
  --cellprob-threshold  the pixels the dynamics start from (default 0.0). Lower finds
                        more cells and fatter ones.
  --diameter            0 (default) hands the frame over at its own scale, which a
                        cellpose-4 model expects. Anything else resizes the frame so a
                        cell that many px across arrives at 30, and moves the default
                        --niter the OTHER way: --diameter 60 doubles the walk, to 400.
  --min-size            cellpose drops masks under 15 px before this script sees them.

`--normalize` matters when tiling: cellpose normalises every image it is handed, and a
tile is an image. See `normalization`.

Not named cellpose.py: a script's own folder comes first on the import path, so
`from cellpose import models` would find itself.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np

from fetch import fetch
from guide import add_guide_arguments, guide_from, tiling_on
from outputs import (Found, add_mask_arguments, add_output_arguments, cleanup_from,
                     frame_count, instance_mask, parse_frames, progress, runs_of,
                     resolve_classes, say, writers_for)
from run import pick_device, run_frames
from tiling import (DEFAULT_OVERLAP, add_stitch_argument, resolve_stitch, settings)

MODEL = "leishmania-cellpose-body"

# Tile size for --guide when neither the flag nor the index says: two of cellpose's own
# 256 px blocks, so a guided tile never does the cutting itself.
GUIDE_TILE = 512

# Defaults where nothing says otherwise. An index entry may set any of these under
# `config.cellpose`, and a flag beats the index.
CELLPOSE = {
    "diameter": 0.0,            # 0: hand the frame over at its own scale
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
    "niter": 0,                 # 0: cellpose's own 200, scaled by --diameter
    "min_size": 15,
    "max_size_fraction": 0.4,
    "bsize": 0,                 # 0: 256 for cpsam, 384 for cpdino
    "block_overlap": 0.1,       # cellpose calls this tile_overlap
    "resample": True,
    "augment": False,
    "batch": 8,                 # cellpose calls this batch_size, and counts blocks
    "normalize": "frame",
    "percentile": [1.0, 99.0],
    "tile_norm": 0,             # cellpose calls this tile_norm_blocksize
    "sharpen": 0,               # ... and this one sharpen_radius
    "invert": False,
}


def weights_for(args) -> tuple[str, dict]:
    """(the checkpoint to hand cellpose, the index entry behind it).

    `--weights` skips the index and takes a path or one of cellpose's own model names.
    An unknown name is refused here: cellpose only warns about a pretrained_model it
    cannot find and then quietly loads its default, so the run would be the wrong model.
    """
    if not args.weights:
        try:
            weights, entry = fetch(args.id, args.version)
        except SystemExit as absent:
            raise SystemExit(f"{absent}\n--weights runs a checkpoint the index does "
                             f"not have: a path, or one of cellpose's own model names.")
        return _checkpoint(weights, entry), entry
    # is_file, not exists: a directory would satisfy exists and then fail inside torch
    # instead of reaching the message below.
    if Path(args.weights).is_file():
        return str(args.weights), {}
    from cellpose import models

    try:
        known = sorted(set(models.MODEL_NAMES) | set(models.get_user_models()))
    except Exception:
        known = sorted(models.MODEL_NAMES)
    if args.weights in known:
        return args.weights, {}
    raise SystemExit(f"--weights {args.weights}: not a file, and not a model cellpose "
                     f"knows ({', '.join(known)}). It would warn and load its default "
                     f"model instead, so this is refused.")


def _checkpoint(weights: Path, entry: dict) -> str:
    """The file inside what `fetch` returned. A cellpose checkpoint is one file, but a
    model published as an archive comes back as a folder."""
    if weights.is_file():
        return str(weights)
    named = ((entry.get("config") or {}).get("cellpose") or {}).get("file")
    if named:
        return str(weights / named)
    files = [item for item in sorted(weights.iterdir())
             if item.is_file() and item.name != ".sha256"]
    if len(files) != 1:
        raise SystemExit(f"{weights} holds {len(files)} files, so which one cellpose "
                         f"should load is unclear. The index entry can say, as "
                         f'"config": {{"cellpose": {{"file": "..."}}}}.')
    return str(files[0])


def resolved(entry: dict, args) -> tuple[dict, dict]:
    """Every cellpose setting, and where each came from: a flag, the index, cellpose.

    The flags default to None so "not given" stays distinct from "given the default",
    which is what lets the index set one and a flag still beat it.
    """
    recorded = (entry.get("config") or {}).get("cellpose") or {}
    unknown = sorted(set(recorded) - set(CELLPOSE) - {"file", "class"})
    if unknown:
        say(f"note: ignoring cellpose settings this script does not know: "
            f"{', '.join(unknown)}")
    values, source = {}, {}
    for name, fallback in CELLPOSE.items():
        given = getattr(args, name)
        if given is not None:
            values[name], source[name] = given, "flag"
        elif name in recorded:
            values[name], source[name] = recorded[name], "index"
        else:
            values[name], source[name] = fallback, "cellpose"
    return values, source


def normalization(values: dict, frame: np.ndarray):
    """The `normalize` argument for one frame: the frame's own bounds, or cellpose's.

    Cellpose normalises EVERY image it is handed to its 1st and 99th percentiles, and a
    tile is an image, so a tiled run would show each tile at its own contrast. `frame`,
    the default, measures once per frame and hands every crop the same two numbers.

    Cellpose ignores sharpening and tile-wise normalisation when given fixed bounds, so
    asking for either falls back to per-image percentiles.

    `off` measures no percentile -- `load_image`'s 0.05/99.95 stretch is the whole of
    it -- and is a fixed 0-255 -> 0-1 map, NOT cellpose's `normalize=False`, which
    hands the network 0-255 where it wants 0-1 (1 cell against 17 on one crop).
    """
    if values["normalize"] == "off":
        return {"normalize": True, "lowhigh": [0.0, 255.0],
                "invert": bool(values["invert"])}
    percentile = [float(p) for p in values["percentile"]]
    params = {"normalize": True, "percentile": percentile,
              "tile_norm_blocksize": int(values["tile_norm"]),
              "sharpen_radius": int(values["sharpen"]),
              "invert": bool(values["invert"])}
    if (values["normalize"] == "frame" and not params["tile_norm_blocksize"]
            and not params["sharpen_radius"]):
        low, high = np.percentile(frame, percentile)
        # A flat frame has nothing to stretch, and dividing by that spread turns a
        # blank tile into NaN. Cellpose skips a flat channel itself.
        if high - low >= 1:
            params["lowhigh"] = [float(low), float(high)]
    return params


def confidence(prob, window, patch: np.ndarray) -> float:
    """A score for one mask: the mean cell probability under it, squashed to 0-1.

    Cellpose scores no instances, but `--stitch` and the label stack both rank by
    score, and a flat 1.0 would make those arbitrary.
    """
    if prob is None:
        return 1.0
    values = prob[window][patch]
    if not values.size:
        return 1.0
    return float(1.0 / (1.0 + np.exp(-float(values.mean()))))


def instances(labels: np.ndarray, prob, origin: tuple, tile: int, name: str,
              cleanup: tuple) -> list:
    """One cellpose label image -> a `Found` per instance, in frame coordinates.

    `find_objects` gives every label's bounding box in one pass, so each mask is cut
    from a small window rather than compared over the whole frame.
    """
    from scipy.ndimage import find_objects

    found = []
    for number, window in enumerate(find_objects(labels), 1):
        if window is None:  # min_size and max_size_fraction remove masks by value
            continue
        patch = labels[window] == number
        placed = instance_mask(patch, (origin[0] + window[0].start,
                                       origin[1] + window[1].start), *cleanup)
        if placed is None:
            continue
        mask, where = placed
        found.append(Found(label=name, score=confidence(prob, window, patch),
                           mask=mask, origin=where, source=tile))
    return found


def add_cellpose_arguments(parser) -> None:
    """Cellpose's own parameters, grouped under one heading.

    Every default is None, so `resolved` can tell a flag that was given from one left
    to the index entry.
    """
    group = parser.add_argument_group("cellpose")
    group.add_argument("--diameter", type=float, default=None,
                       help="resize the frame so a cell this many px across arrives at "
                            "cellpose's 30. Default 0 does not resize, which is what a "
                            "cellpose-4 model expects. It also moves the default "
                            "--niter the other way: 60 doubles the walk, to 400")
    group.add_argument("--flow-threshold", type=float, default=None,
                       help="discard a mask whose flows disagree with the model by "
                            "more than this (default 0.4). Long thin cells score badly "
                            "here, so raise it; 0 turns the check off")
    group.add_argument("--cellprob-threshold", "--prob", type=float, default=None,
                       help="the cell probability the dynamics start from (default "
                            "0.0). Lower finds more cells, and fatter ones")
    group.add_argument("--niter", type=int, default=None,
                       help="iterations of the dynamics (default 200, divided by any "
                            "--diameter rescaling). It is a path length: a cell longer "
                            "than the walk loses the end it never reached, so raise it "
                            "first for a flagellum that comes back separately")
    group.add_argument("--min-size", type=int, default=None,
                       help="cellpose drops masks under this many px (default 15); "
                            "--min-area does the same after this script's cleanup")
    group.add_argument("--max-size-fraction", type=float, default=None,
                       help="drop a mask covering more than this fraction of the image "
                            "(default 0.4). When tiling it is a fraction of the TILE")
    group.add_argument("--bsize", type=int, default=None,
                       help="cellpose's block size; 0 for its default, 256 for cpsam "
                            "(which takes no other) and 384 for cpdino")
    group.add_argument("--block-overlap", type=float, default=None,
                       help="overlap of cellpose's own blocks (default 0.1). Not "
                            "--overlap, which is this script's tiles")
    group.add_argument("--batch", type=int, default=None,
                       help="blocks per forward pass (default 8); lower it if the card "
                            "runs out of memory")
    group.add_argument("--normalize", choices=("frame", "tile", "off"), default=None,
                       help="frame (default): percentiles measured once per frame, "
                            "every tile given the same bounds. tile: cellpose's own, "
                            "each tile at its own contrast. off: keep load_image's "
                            "0.05/99.95 stretch")
    group.add_argument("--percentile", type=float, nargs=2, default=None,
                       metavar=("LO", "HI"),
                       help="the percentiles normalisation maps to 0 and 1 "
                            "(default 1 99)")
    group.add_argument("--tile-norm", type=int, default=None,
                       help="normalise in blocks of this many px, to lift the dark "
                            "corners of an unevenly lit frame; 0 (default) is off")
    group.add_argument("--sharpen", type=int, default=None,
                       help="high-pass radius before normalising; cellpose suggests an "
                            "eighth to a quarter of a cell width. 0 is off")
    # BooleanOptionalAction (3.9+) gives each of these a third state -- not given --
    # which leaves an index entry's value in force.
    boolean = argparse.BooleanOptionalAction
    group.add_argument("--resample", action=boolean, default=None,
                       help="run the dynamics at the frame's own resolution rather "
                            "than at what --diameter rescaled to (default on; it is "
                            "what makes an outline follow the cell)")
    group.add_argument("--augment", action=boolean, default=None,
                       help="run each block four ways and average, at four times the "
                            "forward passes (default off)")
    group.add_argument("--invert", action=boolean, default=None,
                       help="for cells darker than their background (default off)")


def arguments() -> argparse.ArgumentParser:
    """This script's flags, as a parser gui.py can build a form out of."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path,
                        help="a .tif stack, a single image, or a FOLDER of images "
                             "taken in name order")
    parser.add_argument("--id", default=MODEL, help=f"which model; default {MODEL}")
    parser.add_argument("--version", default=None, help="index version; default is the newest")
    parser.add_argument("--weights", default=None,
                        help="a cellpose checkpoint on disk, or one of cellpose's own "
                             "model names such as cpsam; skips the index")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:1, mps or cpu")
    parser.add_argument("--precision", choices=("auto", "bf16", "f32"), default="auto",
                        help="the weights' dtype; auto is bfloat16 on cuda and float32 "
                             "everywhere else")
    parser.add_argument("--verbose", action="store_true",
                        help="cellpose's own log: what it loaded, and stage timings")
    tiling = parser.add_argument_group("tiling")
    tiling.add_argument("--tile", type=int, default=0,
                        help="tile size in px; 0 (default) is one pass over the whole "
                             "frame, which is what cellpose wants. Tiling is for a "
                             "frame too large to hold, and for --guide")
    tiling.add_argument("--overlap", type=float, default=None,
                        help=f"tile overlap as a fraction of the tile, 0 to 0.9 "
                             f"(default {DEFAULT_OVERLAP} where the index says nothing)")
    add_stitch_argument(tiling)
    add_cellpose_arguments(parser)
    add_guide_arguments(parser)
    add_mask_arguments(parser)
    add_output_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = arguments().parse_args(argv)

    from cellpose import models

    if args.verbose:
        from cellpose import io

        io.logger_setup()

    weights, entry = weights_for(args)
    values, source = resolved(entry, args)
    tile, overlap = settings(entry, args.tile, args.overlap)
    tile, overlap = tiling_on(args, entry, tile, overlap, GUIDE_TILE)
    name = args.id if not args.weights else Path(weights).stem
    jobs = runs_of(args, name)
    how = resolve_stitch(args.stitch, args.guide)
    device = pick_device(args.device)

    kwargs = {"pretrained_model": weights, "device": device}
    # bfloat16 arrived in cellpose 4.2; asked of the signature so an older one says so
    # instead of raising a TypeError from inside cellpose.
    if "use_bfloat16" in inspect.signature(models.CellposeModel.__init__).parameters:
        # auto is cuda only: cpu has no fast bfloat16 path, and mps only a recent one.
        kwargs["use_bfloat16"] = (device.type == "cuda" if args.precision == "auto"
                                  else args.precision == "bf16")
    elif args.precision != "auto":
        say("note: this cellpose is older than the --precision argument; ignoring it")
    model = models.CellposeModel(**kwargs)

    if model.backbone == "sam_vitl" and int(values["bsize"]) not in (0, 256):
        raise SystemExit(f"--bsize {values['bsize']}: cpsam was trained on 256 px "
                         f"blocks and cellpose refuses any other. Use --tile to cut "
                         f"the frame up.")
    block = int(values["bsize"]) or (256 if model.backbone == "sam_vitl" else 384)

    # One label image, so one class to name. `--classes` renames it; segment.py calls a
    # whole cell "animal", for a run meant to be read beside that one.
    name = ((entry.get("config") or {}).get("cellpose") or {}).get("class", "cell")
    classes = resolve_classes({0: name}, args.classes)
    name = classes[0]
    cleanup = cleanup_from(args)
    # The detector's batch stays at guide_from's default: --batch here counts cellpose
    # blocks, the detector's counts whole tiles.
    guide = guide_from(args, str(device))

    rescale = 30.0 / values["diameter"] if values["diameter"] else 1.0
    niter = int(values["niter"]) or int(200 / (rescale if values["resample"] else 1.0))
    params = dict(batch_size=int(values["batch"]), resample=bool(values["resample"]),
                  channel_axis=2, diameter=float(values["diameter"]) or None,
                  flow_threshold=float(values["flow_threshold"]),
                  cellprob_threshold=float(values["cellprob_threshold"]),
                  min_size=int(values["min_size"]),
                  max_size_fraction=float(values["max_size_fraction"]),
                  niter=niter, augment=bool(values["augment"]),
                  tile_overlap=float(values["block_overlap"]),
                  bsize=int(values["bsize"]) or None)

    numbers: list = []  # the current run's frames; `announce` reads it

    def announce(frame, boxes):
        # Recomputed here: one percentile over one frame, and only the first is
        # announced.
        normalize = normalization(values, frame)
        say(f"cellpose {model.backbone}: {block} px blocks, "
            f"{float(values['block_overlap']):.0%} overlap, "
            + ("cells at 1.00x" if rescale == 1.0 else
               f"frame rescaled {rescale:.3g}x so a "
               f"{float(values['diameter']):g} px cell arrives at 30")
            + f", niter {niter}, on {device}")
        say(f"flow {values['flow_threshold']}, cellprob "
            f"{values['cellprob_threshold']}, min_size {values['min_size']}, "
            f"normalize {values['normalize']}"
            + (f" ({normalize['lowhigh'][0]:.0f}-{normalize['lowhigh'][1]:.0f} "
               f"-> 0-1)" if "lowhigh" in normalize else "")
            + f"; {len(numbers)} frame(s), classes: {', '.join(classes)}")
        named = sorted(key for key, where in source.items() if where == "index")
        if named:
            say(f"the index sets {', '.join(named)} for this model")

    def predict(frame, boxes):
        normalize = normalization(values, frame)
        found = []
        for index, (y0, x0, y1, x1) in progress(list(enumerate(boxes)), "tiles",
                                                leave=False):
            crop = frame[y0:y1, x0:x1]
            masks, flows, _ = model.eval(crop, normalize=normalize, **params)
            masks = np.asarray(masks)
            # flows[2] is the cell probability, at the frame's size only where the
            # dynamics were resampled to it; otherwise a mask cannot index into it.
            prob = np.asarray(flows[2]) if len(flows) > 2 else None
            if prob is None or prob.shape != masks.shape:
                prob = None
            found += instances(masks, prob, (y0, x0), index, name, cleanup)
        return found

    for job in progress(jobs, "images"):
        numbers = parse_frames(args.frames, frame_count(job.image))
        writers = writers_for(job, job.image, name, classes, numbers)
        run_frames(job.image, numbers, writers, predict, tile=tile, overlap=overlap,
                   how=how, guide=guide, strict=args.guide_strict, announce=announce,
                   item=job.item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
