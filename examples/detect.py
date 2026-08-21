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

**By default the frame goes through whole, at the 640 px input the index records** --
one plain pass, nothing cut up. Ultralytics fits the frame's long side to that, so a
2048 px frame reaches the model at a third of the scale it was trained on and the run
says so.

`--tile 640` opts into tiling, which is what puts cells back at 1.0x: the frame is cut
into tiles the size of the input and each goes through whole. On one 2048 px frame that
is 256 cells against 220, the gain being the small and faint ones a 3x downscale
erases. What it costs is that a cell straddling a seam is seen twice or in halves. The boxes come back in frame
coordinates and each is kept by the one tile whose core it sits in, so a cell seen by
two tiles is not counted twice; `--stitch merge` deduplicates by overlap instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fetch import fetch, frame_sizes, load_image
from outputs import (Found, Writers, add_output_arguments, default_output,
                     frame_count, parse_frames, progress, report, resolve_classes,
                     say)
from tiling import (DEFAULT_IOU, add_tiling_arguments, batches, describe,
                    native_imgsz, recorded_imgsz, resolve_stitch, scale_note,
                    settings, slices, stitch, tiling_hint,
                    warn_if_longer_than_overlap)

MODEL = "leishmania-detect"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path,
                        help="a .tif stack, any single image, or a FOLDER of "
                             "images taken in name order as one sequence")
    parser.add_argument("--version", default=None, help="index version; default is the newest")
    parser.add_argument("--imgsz", type=int, default=0, help="override the inference size")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=None,
                        help=f"--stitch merge only: two boxes from neighbouring tiles "
                             f"overlapping more than this are the same cell (default "
                             f"{DEFAULT_IOU}). The default --stitch core owns rather "
                             f"than compares, and has nothing to apply it to")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, 0, ...")
    add_tiling_arguments(parser)
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    how = resolve_stitch(args.stitch, guided=False)
    if args.iou is not None and how != "merge":
        say("note: --iou is the box merger's threshold and --stitch core does not "
            "merge; it is ignored here")

    from ultralytics import YOLO

    weights, entry = fetch(MODEL, args.version)
    tile, overlap = settings(entry, args.tile, args.overlap)
    numbers = parse_frames(args.frames, frame_count(args.image))
    device = None if args.device == "auto" else args.device
    model = YOLO(weights)
    classes = resolve_classes(model.names, args.classes)

    out = args.out or default_output(args.image, MODEL)
    sizes = frame_sizes(args.image)
    writers = Writers(len(numbers), classes, args.color, out, args.tiff,
                      args.rois, [sizes[n] for n in numbers])

    for position, number in progress(list(enumerate(numbers)), "frames"):
        frame = load_image(args.image, number)
        boxes = slices(*frame.shape[:2], tile, overlap)
        # Tiled, the tile itself, so a crop reaches the model at 1.0x. Untiled, the
        # input size the index records: the frame goes in whole, at the size the
        # network wants, and `scale_note` says what that does to the cells.
        imgsz = args.imgsz or (native_imgsz(frame.shape, tile) if tile
                               else recorded_imgsz(entry, frame.shape))
        if position == 0:
            say(describe(frame.shape, boxes, tile, overlap))
            say(scale_note(boxes, imgsz)
                + f", {len(numbers)} frame(s), classes: {', '.join(classes)}")
            hint = tiling_hint(entry, tile)
            if hint:
                say(hint)

        found = []
        for first, group in progress(list(batches(boxes, args.batch)),
                                     "tiles", leave=False):
            crops = [frame[y0:y1, x0:x1] for y0, x0, y1, x1 in group]
            results = model.predict(crops, imgsz=imgsz, conf=args.conf, device=device,
                                    verbose=False)
            for tile_index, ((y0, x0, _, _), result) in enumerate(zip(group, results),
                                                                  first):
                for label, score, corners in zip(result.boxes.cls.tolist(),
                                                 result.boxes.conf.tolist(),
                                                 result.boxes.xyxy.tolist()):
                    bx0, by0, bx1, by1 = corners
                    found.append(Found(label=classes[int(label)], score=float(score),
                                       box=(bx0 + x0, by0 + y0, bx1 + x0, by1 + y0),
                                       source=tile_index))
        if position == 0:
            warn_if_longer_than_overlap(found, tile, overlap)
        found = stitch(found, boxes, frame.shape, how, iou=args.iou)
        report(number, found, len(numbers))
        writers.add(position, number, frame, found)

    writers.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
