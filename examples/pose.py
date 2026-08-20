#!/usr/bin/env python
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "ultralytics",
#     "tifffile",
#     "roifile",
# ]
# ///
"""leishmania-pose: eight ordered points along each cell, head to flagellar tip.

    python pose.py cells.tif
    python pose.py cells.tif --frames 0-99 --color instance --rois chains.zip

yolo26x-pose, one class. The nodes are Head, Base, Flag1..Flag5, Tip, wired as one
open chain; the names and the edges come from the index rather than from here. Base
is the flagellum junction, so Head -> Base also fixes which way a cell points.

**This one is not tiled**, unlike detect.py and segment.py. The model is scale-robust,
and running whole frames at the 1280 the index records was measured to find more cells
than presenting them at their nominal training scale. Tiling would also cut chains at
tile edges, and half a chain is not half an answer -- the node order is the whole point.

A chain is not a region, so it leaves through the shared outputs as a polyline: an
ImageJ polyline roi, and a one-pixel line in the label stack.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fetch import fetch, load_image
from outputs import (Found, Writers, add_output_arguments, frame_count, parse_frames,
                     report, resolve_classes)

MODEL = "leishmania-pose"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="a .tif stack, or any single image")
    parser.add_argument("--version", default=None, help="index version; default is the newest")
    parser.add_argument("--imgsz", type=int, default=0,
                        help="override the imgsz the index records; see the note below")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, 0, ...")
    add_output_arguments(parser)
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    weights, entry = fetch(MODEL, args.version)
    imgsz = args.imgsz or entry["config"]["detection"]["imgsz"]
    nodes = entry["config"]["keypoints"]["nodes"]
    # The skeleton is the index's, by name. Ultralytics only knows the COCO-17 one and
    # would leave these eight points unjoined, so nothing else here could supply it.
    edges = [(nodes.index(start), nodes.index(end))
             for start, end in entry["config"]["keypoints"]["edges"]]
    numbers = parse_frames(args.frames, frame_count(args.image))
    device = None if args.device == "auto" else args.device
    model = YOLO(weights)
    classes = resolve_classes(model.names, args.classes)

    out = args.out or args.image.with_suffix(f".{MODEL}.png")
    writers = Writers(len(numbers), classes, args.color, out, args.tiff, args.rois)

    for position, number in enumerate(numbers):
        frame = load_image(args.image, number)
        if position == 0:
            height, width = frame.shape[:2]
            # Ultralytics fits the LONGEST side to imgsz, so this is the scale cells are
            # actually presented at. A report, not a complaint: see the note above.
            print(f"{width}x{height} -> imgsz {imgsz}: "
                  f"cells at {imgsz / max(height, width):.2f}x native")
            print(f"{len(numbers)} frame(s), {len(nodes)} points each: "
                  f"{', '.join(nodes)}")

        result = model.predict(frame, imgsz=imgsz, conf=args.conf, device=device,
                               verbose=False)[0]
        found = []
        if result.keypoints is not None and len(result.boxes):
            chains = result.keypoints.xy.cpu().numpy().astype(float)
            # (0, 0) is how ultralytics says a keypoint was not visible, and it is also
            # a real coordinate, so it has to go before anything measures the chain.
            chains[(chains[..., 0] == 0) & (chains[..., 1] == 0)] = np.nan
            found = [Found(label=classes[int(label)], score=float(score), line=chain,
                           edges=edges)
                     for label, score, chain in zip(result.boxes.cls.tolist(),
                                                    result.boxes.conf.tolist(), chains)]
        report(number, found, len(numbers))

        if len(numbers) == 1 and found:
            scores = result.keypoints.conf
            scores = None if scores is None else scores.cpu().numpy()
            print("\nthe first cell, node by node")
            for index, (node, (x, y)) in enumerate(zip(nodes, found[0].line)):
                seen = "" if scores is None else f"  conf {scores[0][index]:.2f}"
                print(f"  {node:6s} ({x:7.1f}, {y:7.1f}){seen}")

        writers.add(position, number, frame, found)

    writers.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
