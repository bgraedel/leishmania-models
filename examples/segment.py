#!/usr/bin/env python
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "ultralytics",
#     "tifffile",
#     "roifile",
# ]
# ///
"""leishmania-seg: a mask per cell, split into body and flagellum.

    python segment.py cells.tif
    python segment.py cells.tif --frames 0-99 --tiff labels.tif --rois masks.zip

yolo26m-seg, three classes: "animal" is the whole cell, "body" and "flagellum" are
its parts, so the three overlap by design rather than partitioning the frame. That is
also why the label stack keeps a channel each -- one label image could not hold a
pixel belonging to two of them.

The frame is cut into 640 px tiles and each is run at 1.0x, the scale the model was
trained on. Masks come back in frame coordinates, and pieces of the same cell found
in neighbouring tiles are unioned rather than suppressed: a flagellum is longer than
the overlap zone is deep, so the tile edge cuts it in two and both halves are real.
`--tile 0` runs the whole frame in one pass instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fetch import fetch, load_image
from outputs import (Found, Writers, add_mask_arguments, add_output_arguments,
                     frame_count, instance_mask, parse_frames, report, resolve_classes)
from tiling import (add_tiling_arguments, batches, describe, merge_masks, native_imgsz,
                    settings, slices)

MODEL = "leishmania-seg"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="a .tif stack, or any single image")
    parser.add_argument("--version", default=None, help="index version; default is the newest")
    parser.add_argument("--imgsz", type=int, default=0, help="override the inference size")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, 0, ...")
    add_tiling_arguments(parser)
    add_mask_arguments(parser)
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
            # retina_masks keeps the masks at the tile's own resolution. Without it they
            # come back at the letterboxed size, and a flagellum a few pixels wide is
            # the first thing to disappear.
            results = model.predict(crops, imgsz=imgsz, conf=args.conf, device=device,
                                    retina_masks=True, verbose=False)
            for tile_index, ((y0, x0, _, _), result) in enumerate(zip(group, results),
                                                                  first):
                if result.masks is None:
                    continue
                masks = result.masks.data.cpu().numpy().astype(bool)
                for label, score, mask in zip(result.boxes.cls.tolist(),
                                              result.boxes.conf.tolist(), masks):
                    placed = instance_mask(mask, (y0, x0), args.components,
                                           args.min_component_area,
                                           args.min_component_frac, args.min_area)
                    if placed is None:
                        continue
                    patch, origin = placed
                    found.append(Found(label=classes[int(label)], score=float(score),
                                       mask=patch, origin=origin, source=tile_index))
        found = merge_masks(found, boxes, frame.shape)
        report(number, found, len(numbers))
        writers.add(position, number, frame, found)

    writers.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
