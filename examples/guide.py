#!/usr/bin/env python
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "ultralytics",
#     "numpy",
#     "pillow",
#     "tifffile",
#     "tqdm",
# ]
# ///
"""Let the detector choose where the segmenter's tiles go.

    python segment.py cells.tif --guide
    python mask2former.py cells.tif --guide --guide-pad 48

    python guide.py cells.tif              # the plan for one frame, without running
                                           # anything on it

A grid is laid down knowing nothing about the frame, so it cuts through cells wherever
they happen to lie, and every `--stitch` mode is an attempt to repair that afterwards.
None of them repairs it completely: ownership double-counts a cell longer than the
margin it leaves, merging fuses cells that touch, and neither can invent the half of a
cell no tile ever saw.

The detector already knows where the cells are and costs a fraction of what a
segmenter does. Run it first and the tiles can be *placed*: `tiling.cover` puts each
cell inside a tile in one piece, using as few tiles as it can, so nothing is cut. On
the data these models were trained for that comes out at the same tile count as the
grid with none of the cells cut; where cells are long it costs more tiles, and the run
says how many against what the grid would have used.

Placed tiles overlap heavily and are not a grid, so `--stitch whole` puts each instance
in the one tile it sits most centrally in -- ownership, deciding from the centroid
alone and comparing nothing, because two tiles draw the same cell differently enough
that comparing their masks counts a thin one twice. See `tiling.stitch_by_whole`.

Every mask then comes back knowing which cell it belongs to, so an animal, its body
and its flagellum carry one number into the label stack and the RoiSet instead of three
unrelated ones. Masks matching no proposed cell are kept and counted separately;
`--guide-strict` discards them instead, which makes the detector the register of what
is in the frame and every count its count.

What it cannot do is find cells the detector missed. A guided run looks only where the
detector pointed, so a cell it did not see is not merely cut, it is never segmented --
and under `--guide-strict` it is dropped even if the segmenter found it anyway.
`--guide-conf` is the dial for that, and lower is safer here than it is for detection:
a false box costs a little tile area, a missed one costs a whole cell.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fetch import fetch, load_image
from outputs import Found, frame_count, parse_frames, progress, say
from tiling import (DEFAULT_OVERLAP, assign, batches, cover, describe_cover,
                    native_imgsz, recorded_imgsz, settings, slices, stitch)

MODEL = "leishmania-detect"


class Guide:
    """The detector, loaded once, planning one frame's tiles at a time.

    Kept as an object rather than a function because the weights are fetched and the
    graph built once for a run of any length, and a plan is wanted per frame: cells
    move, so the tiles that hold them whole move too.
    """

    def __init__(self, version: str | None = None, conf: float = 0.25, pad: int = 32,
                 device=None, batch: int = 4, model_id: str = MODEL):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise SystemExit(
                "--guide places its tiles with the detector, which is an ultralytics "
                "checkpoint: pip install ultralytics. Without it, --tile N grids them "
                "instead.")

        weights, self.entry = fetch(model_id, version)
        self.model = YOLO(weights)
        # The detector's OWN tiling, from its own index entry. It has nothing to do
        # with the tiles being planned for the segmenter; it is only how this model
        # wants to be shown a frame larger than it was trained on.
        self.tile, self.overlap = settings(self.entry, None, None)
        self.conf, self.pad, self.device, self.batch = conf, pad, device, batch

    def boxes(self, frame) -> list:
        """Every cell the detector finds, as (x0, y0, x1, y1) in frame coordinates."""
        grid = slices(*frame.shape[:2], self.tile, self.overlap)
        imgsz = (native_imgsz(frame.shape, self.tile) if self.tile
                 else recorded_imgsz(self.entry, frame.shape))
        found = []
        for first, group in batches(grid, self.batch):
            crops = [frame[y0:y1, x0:x1] for y0, x0, y1, x1 in group]
            results = self.model.predict(crops, imgsz=imgsz, conf=self.conf,
                                         device=self.device, verbose=False)
            for index, ((y0, x0, _, _), result) in enumerate(zip(group, results), first):
                for score, corners in zip(result.boxes.conf.tolist(),
                                          result.boxes.xyxy.tolist()):
                    bx0, by0, bx1, by1 = corners
                    found.append(Found(label="cell", score=float(score),
                                       box=(bx0 + x0, by0 + y0, bx1 + x0, by1 + y0),
                                       source=index))
        # Merging, not ownership. A cell counted twice here is not merely a wasted
        # tile: `assign` hands its two boxes to two different tiles, both of them then
        # own a copy of the same cell, and the duplicate the plan exists to prevent
        # comes back at the end. Ownership double-counts every cell longer than the
        # detector's own overlap, which on this data is most of them, so the boxes are
        # deduplicated by overlap instead -- suppressing a box costs at worst a tile
        # placed a little off, and the segmenter still sees the cell.
        return [item.box for item in stitch(found, grid, frame.shape, "merge")]

    def plan(self, frame, tile: int, overlap: float = 0.25) -> tuple:
        """(tiles, a line about them, which tile owns which cell).

        Tiles are None where there is nothing to place and the caller should keep its
        grid; `overlap` is only for saying what the grid it replaces would have cost.
        """
        if not tile:
            return None, ("--tile 0 is one pass over the whole frame, so there is "
                          "nothing to place and nothing to cut; --guide does nothing"), None
        boxes = self.boxes(frame)
        if not boxes:
            return None, ("the detector found nothing in this frame; running the grid "
                          "instead. --guide-conf lower would look harder"), None
        tiles, oversize = cover(boxes, frame.shape[:2], tile, self.pad)
        return (tiles,
                describe_cover(frame.shape, tiles, boxes, oversize, tile, overlap),
                assign(tiles, boxes))


def add_guide_arguments(parser, switches: bool = True) -> None:
    """The flags for planning tiles with the detector. Shared by the segmenters.

    `switches` off leaves out the two a segmenter alone can act on, for `guide.py`,
    which always runs the plan and has no masks to keep or discard by it.
    """
    if switches:
        parser.add_argument("--guide", action="store_true",
                            help="run the detector first and put the tiles where the "
                                 "cells are, so that each is inside one in one piece, "
                                 "instead of cutting a grid through them")
    parser.add_argument("--guide-version", default=None,
                        help="index version of the detector; default is the newest")
    parser.add_argument("--guide-conf", type=float, default=0.25,
                        help="detector confidence for the plan. A box that is not a "
                             "cell costs a little tile area; a cell with no box is "
                             "never segmented at all, so err low")
    if switches:
        parser.add_argument("--guide-strict", action="store_true",
                            help="keep ONLY masks belonging to a cell the detector "
                                 "proposed, discarding whatever else the segmenter "
                                 "found. Makes the detector the register of what is "
                                 "in the frame, so counts are its counts and every "
                                 "mask has a cell; the price is that a cell it missed "
                                 "is gone rather than merely unattributed. A frame "
                                 "the detector found nothing in still falls back to "
                                 "the grid, unfiltered")
    parser.add_argument("--guide-pad", type=int, default=32,
                        help="clear space in px asked for around every cell, so it "
                             "sits inside its tile rather than against the edge. Given "
                             "up where the frame edge leaves no room, and it costs "
                             "tiles where cells are long")


def guide_from(args, device=None):
    """The Guide a parsed `--guide` asks for, or None."""
    if not getattr(args, "guide", False):
        return None
    return Guide(args.guide_version, args.guide_conf, args.guide_pad, device,
                 getattr(args, "batch", 4))


def main(argv: list[str] | None = None) -> int:
    """`python guide.py cells.tif` -- the plan alone, for looking at before committing
    a long segmentation run to it."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("--tile", type=int, default=640,
                        help="the tile size to plan for; the segmenter's, not the "
                             "detector's")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP,
                        help="the overlap the grid this plan replaces would have run "
                             "at, which is what its tile count is compared against")
    parser.add_argument("--frames", default="0", help="which frames: 0, 10-19, all")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, 0, ...")
    parser.add_argument("--batch", type=int, default=4)
    add_guide_arguments(parser, switches=False)
    args = parser.parse_args(argv)

    numbers = parse_frames(args.frames, frame_count(args.image))
    guide = Guide(args.guide_version, args.guide_conf, args.guide_pad,
                  None if args.device == "auto" else args.device, args.batch)
    for number in progress(numbers, "frames"):
        frame = load_image(args.image, number)
        tiles, note, _ = guide.plan(frame, args.tile, args.overlap)
        say(f"frame {number}: {note}")
        if tiles and len(numbers) == 1:
            for y0, x0, y1, x1 in tiles:
                say(f"  ({x0}, {y0}) - ({x1}, {y1})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
