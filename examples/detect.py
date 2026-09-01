#!/usr/bin/env python
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "ultralytics",
#     "tifffile",
#     "roifile",
#     "tqdm",
# ]
# ///
"""leishmania-detect: a box round every cell.

    python detect.py cells.tif
    python detect.py cells.tif --frames 0-99 --color instance --rois boxes.zip

yolo26m, one class, trained on 640 px tiles of phase-contrast frames.

By default the whole frame goes through in one pass at the 640 px input the index
records, so a larger frame reaches the model downscaled and the run says by how much.
`--tile 640` cuts the frame into input-sized tiles instead, putting cells back at 1.0x
and recovering the small faint ones a downscale erases. Boxes come back in frame
coordinates, each kept by the one tile whose core it sits in, so a cell seen by two
tiles is not counted twice; `--stitch merge` deduplicates by overlap instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fetch import fetch, on_disk
from outputs import (add_output_arguments, frame_count, parse_frames,
                     resolve_classes, say, writers_for)
from run import detect_tiles, run_frames
from tiling import (DEFAULT_IOU, add_tiling_arguments, imgsz_for, resolve_stitch,
                    scale_note, settings, tiling_hint)

MODEL = "leishmania-detect"


def arguments() -> argparse.ArgumentParser:
    """This script's flags, as a parser gui.py can build a form out of."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path,
                        help="a .tif stack, a single image, or a FOLDER of images "
                             "taken in name order")
    parser.add_argument("--version", default=None, help="index version; default is the newest")
    parser.add_argument("--weights", default=None,
                        help="an ultralytics checkpoint on disk; skips the index, so "
                             "--tile and --imgsz say what it needs")
    parser.add_argument("--imgsz", type=int, default=0, help="override the inference size")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=None,
                        help=f"--stitch merge only: boxes from neighbouring tiles "
                             f"overlapping more than this are one cell (default "
                             f"{DEFAULT_IOU})")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, 0, ...")
    add_tiling_arguments(parser)
    add_output_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = arguments().parse_args(argv)
    how = resolve_stitch(args.stitch, guided=False)
    if args.iou is not None and how != "merge":
        say("note: --iou is the box merger's threshold; --stitch core does not merge, "
            "so it is ignored")

    from ultralytics import YOLO

    weights, entry = (on_disk(args.weights) if args.weights
                      else fetch(MODEL, args.version))
    tile, overlap = settings(entry, args.tile, args.overlap)
    numbers = parse_frames(args.frames, frame_count(args.image))
    device = None if args.device == "auto" else args.device
    model = YOLO(weights)
    classes = resolve_classes(model.names, args.classes)

    writers = writers_for(args, args.image,
                          MODEL if not args.weights else weights.stem,
                          classes, numbers)

    def sizing(frame):
        return imgsz_for(entry, frame.shape, tile, args.imgsz)

    def announce(frame, boxes):
        say(scale_note(boxes, sizing(frame))
            + f", {len(numbers)} frame(s), classes: {', '.join(classes)}")
        hint = tiling_hint(entry, tile)
        if hint:
            say(hint)

    def predict(frame, boxes):
        return detect_tiles(model, frame, boxes, sizing(frame), args.conf, device,
                            args.batch, classes)

    run_frames(args.image, numbers, writers, predict, tile=tile, overlap=overlap,
               how=how, iou=args.iou, announce=announce)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
