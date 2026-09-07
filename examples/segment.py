#!/usr/bin/env python
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "ultralytics>=8.4.142",
#     "tifffile",
#     "roifile",
#     "tqdm",
# ]
# ///
"""leishmania-seg: a mask per cell, split into body and flagellum.

    python segment.py cells.tif
    python segment.py cells.tif --frames 0-99 --tiff labels.tif --rois masks.zip

yolo26m-seg, three classes: "animal" is the whole cell, "body" and "flagellum" are its
parts, so the three overlap by design; the label stack keeps a channel each.

By default the whole frame goes through in one pass at the 640 px input the index
records, so a larger frame reaches the model downscaled and the run says by how much.
`--tile 640` cuts input-sized tiles instead, putting cells back at 1.0x. Masks come
back in frame coordinates, each instance kept by the one tile whose core holds its
centroid. `--stitch merge` unions pieces across neighbouring tiles instead, which can
rejoin a cell the tile edge cut but risks fusing cells that merely touch. Both want the
overlap wider than the longest cell; the run says so when it is not.

`--guide` runs the detector first and places the tiles on the cells, so each is inside
one tile whole and nothing has to be repaired. See guide.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fetch import fetch, on_disk
from guide import add_guide_arguments, guide_from, tiling_on
from outputs import (Found, add_mask_arguments, add_output_arguments, cleanup_from,
                     frame_count, instance_mask, parse_frames, progress,
                     resolve_classes, runs_of, say, writers_for)
from run import head_options, run_frames
from tiling import (add_tiling_arguments, batches, imgsz_for, resolve_stitch,
                    scale_note, settings, tiling_hint)

MODEL = "leishmania-seg"


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
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, 0, ...")
    parser.add_argument("--nms", type=float, default=None,
                        help="run the model's one-to-many head through NMS at this "
                             "IoU instead of its end-to-end head, which runs otherwise; "
                             "the two keep different cells in a crowd")
    add_tiling_arguments(parser)
    add_guide_arguments(parser)
    add_mask_arguments(parser)
    add_output_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = arguments().parse_args(argv)

    from ultralytics import YOLO

    weights, entry = (on_disk(args.weights) if args.weights
                      else fetch(MODEL, args.version))
    name = MODEL if not args.weights else weights.stem
    jobs = runs_of(args, name)
    tile, overlap = settings(entry, args.tile, args.overlap)
    tile, overlap = tiling_on(args, entry, tile, overlap)
    device = None if args.device == "auto" else args.device
    how = resolve_stitch(args.stitch, args.guide)
    guide = guide_from(args, device, args.batch)
    model = YOLO(weights)
    classes = resolve_classes(model.names, args.classes)
    numbers: list = []  # the current run's frames; `announce` reads it
    cleanup = cleanup_from(args)

    def sizing(frame):
        return imgsz_for(entry, frame.shape, tile, args.imgsz)

    def announce(frame, boxes):
        say(scale_note(boxes, sizing(frame))
            + f", {len(numbers)} frame(s), classes: {', '.join(classes)}")
        hint = tiling_hint(entry, tile)
        if hint:
            say(hint)

    def predict(frame, boxes):
        imgsz = sizing(frame)
        found = []
        for first, group in progress(list(batches(boxes, args.batch)),
                                     "tiles", leave=False):
            crops = [frame[y0:y1, x0:x1] for y0, x0, y1, x1 in group]
            # retina_masks keeps the masks at the tile's own resolution; without it a
            # flagellum a few pixels wide is lost to the letterboxed size.
            results = model.predict(crops, imgsz=imgsz, conf=args.conf, device=device,
                                    retina_masks=True, verbose=False,
                                    **head_options(args.nms))
            for tile_index, ((y0, x0, _, _), result) in enumerate(zip(group, results),
                                                                  first):
                if result.masks is None:
                    continue
                masks = result.masks.data.cpu().numpy().astype(bool)
                for label, score, mask in zip(result.boxes.cls.tolist(),
                                              result.boxes.conf.tolist(), masks):
                    placed = instance_mask(mask, (y0, x0), *cleanup)
                    if placed is None:
                        continue
                    patch, origin = placed
                    found.append(Found(label=classes[int(label)], score=float(score),
                                       mask=patch, origin=origin, source=tile_index))
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
