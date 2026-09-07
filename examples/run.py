#!/usr/bin/env python
"""The frame loop every example script shares: load a frame, tile it, run the model,
stitch the pieces back, report and write.

Only the model call differs between scripts, so each passes its own `predict` to
`run_frames` and everything else is common.
"""

from __future__ import annotations

from pathlib import Path

from fetch import load_image
from outputs import Found, progress, report, say
from tiling import batches, describe, slices, stitch, warn_if_longer_than_overlap


def pick_device(name: str):
    """cuda, then Apple's mps, then the cpu -- unless the flag names one.

    Returns a torch.device, so a card can be named as `cuda:1`. torch is imported
    here, not at the top, so importing a script for its `arguments()` stays cheap.
    """
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        # Some Mask2Former ops have no Metal kernel; PYTORCH_ENABLE_MPS_FALLBACK=1
        # sends those to the cpu rather than raising.
        return torch.device("mps")
    return torch.device("cpu")


def head_options(nms) -> dict:
    """What picks an ultralytics model's head: the end-to-end one, unless `--nms` asks
    for the one-to-many head put through NMS at that IoU.

    `nms=False` is the end-to-end head and `nms=None` external NMS, in ultralytics
    8.4.142 and later; said outright because the default there is NMS at 0.7, not the
    end-to-end head a YOLO26 checkpoint is published for. Ultralytics reads these when
    it first builds a model's predictor and not again, so they go on every call and
    hold for the run. A checkpoint with no end-to-end head runs its one head either
    way, at the IoU given or ultralytics' own.
    """
    return {"nms": False} if nms is None else {"nms": None, "iou": float(nms)}


def detect_tiles(model, frame, grid: list, imgsz: int, conf: float, device,
                 batch: int, names: list, bar: bool = True,
                 nms: float | None = None) -> list[Found]:
    """An ultralytics detector over one frame's crops -> box `Found`s in frame
    coordinates, each carrying the index of the tile it came out of.

    Shared by detect.py and the guide so both see the same cells in a frame.
    """
    groups = list(batches(grid, batch))
    if bar:
        groups = progress(groups, "tiles", leave=False)
    found = []
    for first, group in groups:
        crops = [frame[y0:y1, x0:x1] for y0, x0, y1, x1 in group]
        results = model.predict(crops, imgsz=imgsz, conf=conf, device=device,
                                verbose=False, **head_options(nms))
        for index, ((y0, x0, _, _), result) in enumerate(zip(group, results), first):
            for label, score, corners in zip(result.boxes.cls.tolist(),
                                             result.boxes.conf.tolist(),
                                             result.boxes.xyxy.tolist()):
                bx0, by0, bx1, by1 = corners
                found.append(Found(label=names[int(label)], score=float(score),
                                   box=(bx0 + x0, by0 + y0, bx1 + x0, by1 + y0),
                                   source=index))
    return found


def run_frames(image, numbers: list, writers, predict, *, tile: int = 0,
               overlap: float = 0.0, how: str = "core", guide=None,
               strict: bool = False, iou: float | None = None,
               suffix=None, announce=None, epilogue=None, item=None) -> None:
    """Run `predict` over every frame, and do the same things to what comes back.

    `predict(frame, boxes) -> list[Found]` is the model call: `boxes` are tiles
    (y0, x0, y1, x1) and the instances must come back in FRAME coordinates, each
    carrying the index of the tile it came out of. `announce(frame, boxes)` prints
    under the layout line on frame 0, `suffix(frame, boxes)` appends to it, and
    `epilogue(number, found)` prints after each frame's report.

    `item` is (which, of how many) when this is one image of a folder: the layout line
    then names the image, `announce` speaks for the first image only, and every frame
    gets one line rather than the first getting its instances listed.
    """
    warned = False
    batch = item is not None
    head = f"[{item[0] + 1}/{item[1]}] {Path(image).name}: " if batch else ""
    for position, number in progress(list(enumerate(numbers)), "frames",
                                     leave=not batch):
        frame = load_image(image, number)
        boxes = slices(*frame.shape[:2], tile, overlap)
        planned, note, plan = (guide.plan(frame, tile, overlap) if guide
                               else (None, None, None))
        boxes = planned or boxes
        if position == 0:
            say(head + (note or describe(frame.shape, boxes, tile, overlap))
                + (suffix(frame, boxes) if suffix is not None else ""))
            if announce is not None and (not batch or item[0] == 0):
                announce(frame, boxes)
        elif guide is not None and planned is None and tile:
            # A guided frame that fell back to the grid; only frame 0 is described above.
            say(f"  frame {number}: {note}")

        found = predict(frame, boxes)

        if not planned and (position == 0 or (guide is not None and not warned)):
            # Before stitching, so the lengths are the ones the tiles saw. Only for
            # gridded frames: placed tiles have no single overlap to be longer than.
            warn_if_longer_than_overlap(found, tile, overlap)
            warned = True
        # `whole` on a gridded frame keeps both halves of every straddling cell, so a
        # guided frame that fell back to the grid stitches by core instead.
        frame_how = ("core" if guide is not None and planned is None
                     and how == "whole" else how)
        found = stitch(found, boxes, frame.shape[:2], frame_how, iou=iou, plan=plan)
        # Only where there is a plan; a gridded fallback frame has no cells to require.
        loose = 0
        if plan and strict:
            loose = sum(1 for item in found if not item.cell)
            found = [item for item in found if item.cell]
        report(number, found, len(numbers), loose, brief=batch)
        if epilogue is not None:
            epilogue(number, found)
        writers.add(position, number, frame, found)
    writers.close()
