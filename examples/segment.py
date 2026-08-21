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
"""leishmania-seg: a mask per cell, split into body and flagellum.

    python segment.py cells.tif
    python segment.py cells.tif --frames 0-99 --tiff labels.tif --rois masks.zip

yolo26m-seg, three classes: "animal" is the whole cell, "body" and "flagellum" are
its parts, so the three overlap by design rather than partitioning the frame. That is
also why the label stack keeps a channel each -- one label image could not hold a
pixel belonging to two of them.

**By default the frame goes through whole, at the 640 px input the index records** --
one plain pass, nothing cut up. A frame larger than that reaches the model scaled down,
and the run says by how much.

`--tile 640` opts into tiling, which is what puts cells back at 1.0x. Masks come back in frame
coordinates, and each instance is kept by the one tile whose core holds its centroid
-- nothing is compared, so nothing can be counted twice or chained into a blob.
`--stitch merge` instead unions pieces of the same cell across neighbouring tiles,
which can rejoin a cell the tile edge cut in two but risks fusing cells that merely
touch. Both want the overlap wider than the longest cell; the run says so when it is
not.

`--guide` skips that question: it runs the detector first and puts the tiles where the
cells actually are, so each is inside one in one piece and there is nothing to repair.
See guide.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fetch import fetch, frame_sizes, load_image
from guide import add_guide_arguments, guide_from
from outputs import (Found, Writers, add_mask_arguments, add_output_arguments,
                     default_output, frame_count, instance_mask, parse_frames,
                     progress, report, resolve_classes, say)
from tiling import (add_tiling_arguments, batches, describe, native_imgsz,
                    recorded_imgsz, resolve_stitch, scale_note, settings, slices,
                    stitch, tiling_hint, warn_if_longer_than_overlap)

MODEL = "leishmania-seg"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path,
                        help="a .tif stack, any single image, or a FOLDER of "
                             "images taken in name order as one sequence")
    parser.add_argument("--version", default=None, help="index version; default is the newest")
    parser.add_argument("--imgsz", type=int, default=0, help="override the inference size")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, 0, ...")
    add_tiling_arguments(parser)
    add_guide_arguments(parser)
    add_mask_arguments(parser)
    add_output_arguments(parser)
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    weights, entry = fetch(MODEL, args.version)
    tile, overlap = settings(entry, args.tile, args.overlap)
    if args.guide and not tile:
        # --guide IS a tiling strategy, so asking for it asks for tiling. With no
        # --tile of its own it takes the size the index records.
        tile, overlap = settings(entry, None, args.overlap)
        say(f"--guide places tiles, so it turns tiling on: {tile} px, "
            f"the size the index records")
    numbers = parse_frames(args.frames, frame_count(args.image))
    device = None if args.device == "auto" else args.device
    how = resolve_stitch(args.stitch, args.guide)
    guide = guide_from(args, device)
    model = YOLO(weights)
    classes = resolve_classes(model.names, args.classes)

    out = args.out or default_output(args.image, MODEL)
    sizes = frame_sizes(args.image)
    writers = Writers(len(numbers), classes, args.color, out, args.tiff,
                      args.rois, [sizes[n] for n in numbers])

    for position, number in progress(list(enumerate(numbers)), "frames"):
        frame = load_image(args.image, number)
        boxes = slices(*frame.shape[:2], tile, overlap)
        planned, note, plan = (guide.plan(frame, tile, overlap) if guide
                               else (None, None, None))
        boxes = planned or boxes
        # Tiled, the tile itself, so a crop reaches the model at 1.0x. Untiled, the
        # input size the index records: the frame goes in whole, at the size the
        # network wants, and `scale_note` says what that does to the cells.
        imgsz = args.imgsz or (native_imgsz(frame.shape, tile) if tile
                               else recorded_imgsz(entry, frame.shape))
        if position == 0:
            say(note or describe(frame.shape, boxes, tile, overlap))
            say(scale_note(boxes, imgsz)
                + f", {len(numbers)} frame(s), classes: {', '.join(classes)}")
            hint = tiling_hint(entry, tile)
            if hint:
                say(hint)

        found = []
        for first, group in progress(list(batches(boxes, args.batch)),
                                     "tiles", leave=False):
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
        if position == 0 and not planned:
            # Before stitching, so the lengths are the ones the tiles saw. Guided tiles
            # have no one overlap to be longer than, and the plan says for itself which
            # cells no tile could hold whole.
            warn_if_longer_than_overlap(found, tile, overlap)
        found = stitch(found, boxes, frame.shape, how, plan=plan)
        # Only where there IS a plan: a frame the detector found nothing in has fallen
        # back to the grid, and there is no register there to be strict about.
        loose = 0
        if plan and args.guide_strict:
            loose = sum(1 for item in found if not item.cell)
            found = [item for item in found if item.cell]
        report(number, found, len(numbers), loose)
        writers.add(position, number, frame, found)

    writers.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
