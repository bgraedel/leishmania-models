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

yolo26x-pose, one class. The nodes are Head, Base, Flag1..Flag5, Tip, wired as one open
chain; the names and edges come from the index. Base is the flagellum junction, so
Head -> Base fixes which way a cell points.

By default the whole frame goes through in one pass at the 1280 px input the index
records. This model is scale-robust, so that finds more cells here than tiling does.

`--tile 1280` turns tiling on, and only `--stitch core` is safe with it: two node
orderings cannot be spliced, so a chain is only ever kept whole by the tile whose core
holds it. A chain whose far nodes fell outside that tile stays incomplete, so the tile
has to cover the cell.

Chains leave through the shared outputs as polylines: an ImageJ polyline roi, and a
one-pixel line in the label stack.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fetch import fetch, on_disk
from outputs import (Found, add_output_arguments, frame_count, parse_frames,
                     progress, resolve_classes, say, writers_for)
from run import run_frames
from tiling import (add_tiling_arguments, batches, imgsz_for, resolve_stitch,
                    scale_note, settings, tiling_hint)

MODEL = "leishmania-pose"


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
                             "the chain falls back to numbered nodes joined in order")
    parser.add_argument("--imgsz", type=int, default=0,
                        help="override the imgsz the index records")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, 0, ...")
    add_tiling_arguments(parser)
    add_output_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = arguments().parse_args(argv)
    how = resolve_stitch(args.stitch, guided=False)
    if how == "merge":
        # Refused before a model loads; `stitch` would only find out mid-run.
        raise SystemExit(
            "--stitch merge cannot rejoin chains: no area to compare, and no way to "
            "splice two node orderings. Use --stitch core.")

    from ultralytics import YOLO

    weights, entry = (on_disk(args.weights) if args.weights
                      else fetch(MODEL, args.version))
    tile, overlap = settings(entry, args.tile, args.overlap)
    numbers = parse_frames(args.frames, frame_count(args.image))
    device = None if args.device == "auto" else args.device
    model = YOLO(weights)
    recorded = (entry.get("config") or {}).get("keypoints") or {}
    nodes = recorded.get("nodes")
    if nodes:
        # The skeleton is the index's, by name; ultralytics only knows COCO-17.
        edges = [(nodes.index(start), nodes.index(end))
                 for start, end in recorded["edges"]]
    else:
        # An unpublished checkpoint carries no chain: number the nodes, join in order.
        count = int((getattr(model.model, "kpt_shape", None) or [0])[0])
        if not count:
            raise SystemExit(f"{weights}: this checkpoint reports no keypoint shape, "
                             f"so the chain length is unknown")
        nodes = [f"node{index + 1}" for index in range(count)]
        edges = [(index, index + 1) for index in range(count - 1)]
    classes = resolve_classes(model.names, args.classes)

    writers = writers_for(args, args.image,
                          MODEL if not args.weights else weights.stem,
                          classes, numbers)
    # The one untiled result, kept so the epilogue can read per-node confidences off it.
    whole = {}

    def sizing(frame):
        return imgsz_for(entry, frame.shape, tile, args.imgsz)

    def announce(frame, boxes):
        say(scale_note(boxes, sizing(frame)))
        say(f"{len(numbers)} frame(s), {len(nodes)} points each: {', '.join(nodes)}")
        hint = tiling_hint(entry, tile)
        if hint:
            say(hint)

    def predict(frame, boxes):
        imgsz = sizing(frame)
        found = []
        whole.clear()
        for first, group in progress(list(batches(boxes, args.batch)), "tiles",
                                     leave=False):
            crops = [frame[y0:y1, x0:x1] for y0, x0, y1, x1 in group]
            results = model.predict(crops, imgsz=imgsz, conf=args.conf, device=device,
                                    verbose=False)
            for tile_index, ((y0, x0, _, _), result) in enumerate(zip(group, results),
                                                                  first):
                if len(boxes) == 1:
                    whole["result"] = result
                if result.keypoints is None or not len(result.boxes):
                    continue
                chains = result.keypoints.xy.cpu().numpy().astype(float)
                # (0, 0) means "not visible" in ultralytics, and is also a real
                # coordinate: drop it before the tile's corner is added back.
                chains[(chains[..., 0] == 0) & (chains[..., 1] == 0)] = np.nan
                chains[..., 0] += x0
                chains[..., 1] += y0
                found += [Found(label=classes[int(label)], score=float(score),
                                line=chain, edges=edges, source=tile_index)
                          for label, score, chain in zip(result.boxes.cls.tolist(),
                                                         result.boxes.conf.tolist(),
                                                         chains)]
        return found

    def epilogue(number, found):
        single = whole.get("result")
        if len(numbers) != 1 or not found or single is None:
            return
        scores = single.keypoints.conf
        scores = None if scores is None else scores.cpu().numpy()
        say("\nthe first cell, node by node")
        for index, (node, (x, y)) in enumerate(zip(nodes, found[0].line)):
            seen = "" if scores is None else f"  conf {scores[0][index]:.2f}"
            say(f"  {node:6s} ({x:7.1f}, {y:7.1f}){seen}")

    run_frames(args.image, numbers, writers, predict, tile=tile, overlap=overlap,
               how=how, announce=announce, epilogue=epilogue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
