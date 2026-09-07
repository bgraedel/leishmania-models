#!/usr/bin/env python
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "ultralytics>=8.4.142",
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

A grid cuts through cells wherever they lie and no `--stitch` mode repairs that fully.
The detector is cheap, so run it first and `tiling.cover` places each tile to hold its
cells whole, in as few tiles as it can. Placed tiles overlap heavily and are not a grid,
so use `--stitch whole`, which gives each instance to the tile it sits most centrally in.

Every mask comes back knowing which cell it belongs to, so an animal, its body and its
flagellum carry one number into the label stack and the RoiSet. Masks matching no
proposed cell are kept and counted separately; `--guide-strict` discards them, making
the detector the register of what is in the frame.

A cell the detector missed is never segmented at all, so keep `--guide-conf` low: a
false box costs a little tile area, a missed one costs a whole cell.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fetch import fetch, load_image
from outputs import frame_count, parse_frames, progress, resolve_classes, runs_of, say
from run import detect_tiles
from tiling import (DEFAULT_OVERLAP, assign, cover, describe_cover, imgsz_for,
                    recorded_tile, settings, slices, stitch)

MODEL = "leishmania-detect"


class Guide:
    """The detector, loaded once, planning one frame's tiles at a time.

    Cells move, so the plan is remade per frame.
    """

    def __init__(self, version: str | None = None, conf: float = 0.25, pad: int = 32,
                 device=None, batch: int = 4, model_id: str = MODEL,
                 nms: float | None = None):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise SystemExit(
                "--guide places its tiles with the detector, which is an ultralytics "
                "checkpoint: pip install ultralytics. Without it, --tile N grids them "
                "instead.")

        weights, self.entry = fetch(model_id, version)
        self.model = YOLO(weights)
        self.names = resolve_classes(self.model.names)
        # The detector's OWN tiling, from its index entry: how it wants to be shown a
        # large frame, unrelated to the tiles being planned for the segmenter.
        self.tile, self.overlap = settings(self.entry, None, None)
        self.conf, self.pad, self.device, self.batch = conf, pad, device, batch
        self.nms = nms  # --guide-nms; see run.head_options

    def boxes(self, frame) -> list:
        """Every cell the detector finds, as (x0, y0, x1, y1) in frame coordinates.

        Runs `detect_tiles`, the same loop detect.py uses, so the plan and the detector
        script agree about which cells are in a frame.
        """
        grid = slices(*frame.shape[:2], self.tile, self.overlap)
        found = detect_tiles(self.model, frame, grid,
                             imgsz_for(self.entry, frame.shape, self.tile),
                             self.conf, self.device, self.batch, self.names,
                             bar=False, nms=self.nms)
        # Deduplicated by overlap: ownership double-counts any cell longer than the
        # detector's own overlap, and `assign` would give each duplicate its own tile.
        return [item.box for item in stitch(found, grid, frame.shape, "merge")]

    def plan(self, frame, tile: int, overlap: float = 0.25) -> tuple:
        """(tiles, a line about them, which tile owns which cell).

        Tiles are None where there is nothing to place and the caller should keep its
        grid; `overlap` is only for saying what that grid would have cost.
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
    """The flags for planning tiles with the detector, shared by the segmenters.

    `switches` off leaves out the two only a segmenter can act on, for `guide.py`.
    """
    group = parser.add_argument_group("guided tiles")
    if switches:
        group.add_argument("--guide", action="store_true",
                           help="run the detector first and place the tiles so each "
                                "cell sits inside one whole")
    group.add_argument("--guide-version", default=None,
                       help="index version of the detector; default is the newest")
    group.add_argument("--guide-conf", type=float, default=0.25,
                       help="detector confidence for the plan; err low, a cell with "
                            "no box is never segmented")
    if switches:
        group.add_argument("--guide-strict", action="store_true",
                           help="keep ONLY masks belonging to a detected cell; a "
                                "frame with no detections still falls back to the grid")
    group.add_argument("--guide-pad", type=int, default=32,
                       help="clear space in px around every cell inside its tile; "
                            "costs tiles where cells are long")
    group.add_argument("--guide-nms", type=float, default=None,
                       help="run the detector's one-to-many head through NMS at this "
                            "IoU instead of its end-to-end head, which runs otherwise; "
                            "the two keep different cells in a crowd. See --nms")


def tiling_on(args, entry: dict, tile: int, overlap: float, fallback: int = 0,
              why: str = "") -> tuple:
    """Turn tiling on where `--guide` asks for it and no `--tile` named a size.

    The (tile, overlap) `settings` resolved pass through untouched unless --guide is set
    and the tile is 0: then the size the index records for the model, or `fallback` where
    the entry records none. `why` says where a fallback came from.
    """
    if not getattr(args, "guide", False) or tile:
        return tile, overlap
    recorded = recorded_tile(entry)
    tile = recorded or fallback
    say(f"--guide places tiles, so it turns tiling on: {tile} px"
        + (", the size the index records" if recorded else why))
    return tile, overlap


def guide_from(args, device=None, batch: int = 4):
    """The Guide a parsed `--guide` asks for, or None.

    `batch` is the DETECTOR's tiles per forward pass, so a script whose own --batch
    counts something else (cellpose blocks) should leave it alone.
    """
    if not getattr(args, "guide", False):
        return None
    return Guide(args.guide_version, args.guide_conf, args.guide_pad, device, batch,
                 nms=args.guide_nms)


def arguments() -> argparse.ArgumentParser:
    """This script's flags as their own parser, so gui.py can build a form from them."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("--tile", type=int, default=640,
                        help="the segmenter's tile size to plan for, in px")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP,
                        help="overlap of the grid this plan's tile count is compared "
                             "against")
    parser.add_argument("--frames", default="0", help="which frames: 0, 10-19, all")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, 0, ...")
    parser.add_argument("--batch", type=int, default=4)
    add_guide_arguments(parser, switches=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    """`python guide.py cells.tif` -- the plan alone, before committing a long run."""
    args = arguments().parse_args(argv)

    jobs = runs_of(args, MODEL, writes=False)
    guide = Guide(args.guide_version, args.guide_conf, args.guide_pad,
                  None if args.device == "auto" else args.device, args.batch,
                  nms=args.guide_nms)
    for job in progress(jobs, "images"):
        numbers = parse_frames(args.frames, frame_count(job.image))
        head = ("" if job.item is None
                else f"[{job.item[0] + 1}/{job.item[1]}] {job.image.name}, ")
        for number in progress(numbers, "frames", leave=job.item is None):
            frame = load_image(job.image, number)
            tiles, note, _ = guide.plan(frame, args.tile, args.overlap)
            say(f"{head}frame {number}: {note}")
            if tiles and len(numbers) == 1 and job.item is None:
                for y0, x0, y1, x1 in tiles:
                    say(f"  ({x0}, {y0}) - ({x1}, {y1})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
