#!/usr/bin/env python
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "ultralytics",
#     "tifffile",
#     "roifile",
# ]
# ///
"""leishmania-detect: a box round every cell.

    python detect.py cells.tif
    python detect.py cells.tif --frames 0-99 --color instance --rois boxes.zip

yolo26m, one class, trained on 640 px tiles of phase-contrast frames. The frame is
cut into tiles that size and each one is run at 1.0x, so cells stay the size the
model was trained on however large the frame is; the boxes are then put back in
frame coordinates and deduplicated where tiles overlap. `--tile 0` runs the whole
frame in one pass instead, at the imgsz the index records.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fetch import fetch, load_image
from outputs import (Found, Writers, add_output_arguments, frame_count, parse_frames,
                     report, resolve_classes)
from tiling import (add_tiling_arguments, batches, describe, merge_boxes, native_imgsz,
                    settings, slices)

MODEL = "leishmania-detect"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="a .tif stack, or any single image")
    parser.add_argument("--version", default=None, help="index version; default is the newest")
    parser.add_argument("--imgsz", type=int, default=0, help="override the inference size")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5,
                        help="two boxes from neighbouring tiles overlapping more than "
                             "this are the same cell")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, 0, ...")
    add_tiling_arguments(parser)
    add_output_arguments(parser)
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    weights, entry = fetch(MODEL, args.version)
    tile, overlap = settings(entry, args.tile, args.overlap)
    numbers = parse_frames(args.frames, frame_count(args.image))
    device = None if args.device == "auto" else args.device
    model = YOLO(weights)
    classes = resolve_classes(model.names, args.classes)

    out = args.out or args.image.with_suffix(f".{MODEL}.png")
    writers = Writers(len(numbers), classes, args.color, out, args.tiff, args.rois)

    for position, number in enumerate(numbers):
        frame = load_image(args.image, number)
        boxes = slices(*frame.shape[:2], tile, overlap)
        imgsz = args.imgsz or (native_imgsz(frame.shape, tile) if tile
                               else entry["config"]["detection"]["imgsz"])
        if position == 0:
            print(describe(frame.shape, boxes, tile, overlap))
            print(f"imgsz {imgsz}, {len(numbers)} frame(s), classes: {', '.join(classes)}")

        found = []
        for first, group in batches(boxes, args.batch):
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
        found = merge_boxes(found, boxes, frame.shape, args.iou)
        report(number, found, len(numbers))
        writers.add(position, number, frame, found)

    writers.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
