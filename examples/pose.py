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
"""leishmania-pose: eight ordered points along each cell, head to flagellar tip.

    python pose.py cells.tif
    python pose.py cells.tif --frames 0-99 --color instance --rois chains.zip

yolo26x-pose, one class. The nodes are Head, Base, Flag1..Flag5, Tip, wired as one
open chain; the names and the edges come from the index rather than from here. Base
is the flagellum junction, so Head -> Base also fixes which way a cell points.

**The frame goes through whole, at the 1280 px input the index records** -- one plain
pass, nothing cut up, as in every other script here.

For this model that is not merely the plain default but the better one: it is
scale-robust, and the whole frame at that input finds more cells than tiling does to
present them at their nominal training scale.

Tiling is available, and `--stitch core` is what makes it safe: half a chain is not
half an answer, so unioning pieces across a tile edge was never an option -- the node
order is the whole point and two orderings cannot be spliced. Ownership never joins
anything, so a chain is only ever kept whole, by the tile whose core holds it.
`--tile 1280` turns it on. What it still cannot do is complete a chain whose far nodes
fell outside that tile, so the margin has to cover the cell.

A chain is not a region, so it leaves through the shared outputs as a polyline: an
ImageJ polyline roi, and a one-pixel line in the label stack.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fetch import fetch, frame_sizes, load_image
from outputs import (Found, Writers, add_output_arguments, default_output,
                     frame_count, parse_frames, progress, report, resolve_classes,
                     say)
from tiling import (add_tiling_arguments, batches, describe, native_imgsz,
                    recorded_imgsz, resolve_stitch, scale_note, settings, slices,
                    stitch, tiling_hint, warn_if_longer_than_overlap)

MODEL = "leishmania-pose"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path,
                        help="a .tif stack, any single image, or a FOLDER of "
                             "images taken in name order as one sequence")
    parser.add_argument("--version", default=None, help="index version; default is the newest")
    parser.add_argument("--imgsz", type=int, default=0,
                        help="override the imgsz the index records; see the note below")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, 0, ...")
    add_tiling_arguments(parser)
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    how = resolve_stitch(args.stitch, guided=False)

    from ultralytics import YOLO

    weights, entry = fetch(MODEL, args.version)
    tile, overlap = settings(entry, args.tile, args.overlap)
    nodes = entry["config"]["keypoints"]["nodes"]
    # The skeleton is the index's, by name. Ultralytics only knows the COCO-17 one and
    # would leave these eight points unjoined, so nothing else here could supply it.
    edges = [(nodes.index(start), nodes.index(end))
             for start, end in entry["config"]["keypoints"]["edges"]]
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
        height, width = frame.shape[:2]
        boxes = slices(height, width, tile, overlap)
        # Tiled, the tile itself, so a crop reaches the model at 1.0x. Untiled, the
        # input size the index records: the frame goes in whole, at the size the
        # network wants, and `scale_note` says what that does to the cells.
        imgsz = args.imgsz or (native_imgsz(frame.shape, tile) if tile
                               else recorded_imgsz(entry, frame.shape))
        if position == 0:
            say(describe(frame.shape, boxes, tile, overlap))
            say(scale_note(boxes, imgsz))
            say(f"{len(numbers)} frame(s), {len(nodes)} points each: "
                  f"{', '.join(nodes)}")
            hint = tiling_hint(entry, tile)
            if hint:
                say(hint)

        found, single = [], None
        for first, group in progress(list(batches(boxes, args.batch)), "tiles",
                                     leave=False):
            crops = [frame[y0:y1, x0:x1] for y0, x0, y1, x1 in group]
            results = model.predict(crops, imgsz=imgsz, conf=args.conf, device=device,
                                    verbose=False)
            for tile_index, ((y0, x0, _, _), result) in enumerate(zip(group, results),
                                                                  first):
                if len(boxes) == 1:
                    single = result
                if result.keypoints is None or not len(result.boxes):
                    continue
                chains = result.keypoints.xy.cpu().numpy().astype(float)
                # (0, 0) is how ultralytics says a keypoint was not visible, and it is
                # also a real coordinate, so it has to go before anything measures the
                # chain -- and before the tile's own corner is added back.
                chains[(chains[..., 0] == 0) & (chains[..., 1] == 0)] = np.nan
                chains[..., 0] += x0
                chains[..., 1] += y0
                found += [Found(label=classes[int(label)], score=float(score),
                                line=chain, edges=edges, source=tile_index)
                          for label, score, chain in zip(result.boxes.cls.tolist(),
                                                         result.boxes.conf.tolist(),
                                                         chains)]
        if position == 0:
            warn_if_longer_than_overlap(found, tile, overlap)
        found = stitch(found, boxes, (height, width), how)
        report(number, found, len(numbers))

        if len(numbers) == 1 and found and single is not None:
            scores = single.keypoints.conf
            scores = None if scores is None else scores.cpu().numpy()
            say("\nthe first cell, node by node")
            for index, (node, (x, y)) in enumerate(zip(nodes, found[0].line)):
                seen = "" if scores is None else f"  conf {scores[0][index]:.2f}"
                say(f"  {node:6s} ({x:7.1f}, {y:7.1f}){seen}")

        writers.add(position, number, frame, found)

    writers.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
