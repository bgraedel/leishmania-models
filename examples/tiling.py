#!/usr/bin/env python
"""Running a model over a frame in tiles, and putting the pieces back together.

A detector trained on 640 px tiles sees a 2048 px frame at a third of the scale it
knows, and the smallest cells go with it. Cutting the frame into tiles the size it
was trained on and running each one keeps every cell at 1.0x. The cost is that an
object straddling a tile edge is found twice, or in halves, so the pieces have to be
rejoined afterwards -- which is the rest of this file.

The tile size comes from the index (`config.tiling.tile`) and so does the overlap
where an entry records one. **Overlap is a fraction of the tile**, and it has to be
wider than the longest object: a cell longer than the overlap is never whole in any
tile, and no amount of merging afterwards invents the part that was never seen.
"""

from __future__ import annotations

import numpy as np

from outputs import Found

# Where an entry records no overlap. A quarter of a 640 px tile is 160 px, which is
# comfortably longer than a promastigote with its flagellum at these magnifications.
DEFAULT_OVERLAP = 0.25


def settings(entry: dict, tile: int | None, overlap: float | None) -> tuple[int, float]:
    """The tile size and overlap to use: the flags if given, else what the index says.

    `--tile 0` means the whole frame in one pass, and comes back as 0.
    """
    recorded = (entry.get("config") or {}).get("tiling") or {}
    if tile is None:
        tile = int(recorded.get("tile") or 0)
    if overlap is None:
        overlap = float(recorded.get("overlap") or DEFAULT_OVERLAP)
    if not 0.0 <= overlap < 0.9:
        raise SystemExit(f"--overlap is a fraction of the tile, 0 to 0.9; got {overlap}")
    return int(tile), float(overlap)


def slices(height: int, width: int, tile: int, overlap: float = 0.0):
    """Tile boxes (y0, x0, y1, x1) covering the frame.

    The last row and column are snapped back to the edge rather than allowed to run
    off it, so every tile is a full `tile` square wherever the frame is big enough,
    and the last one overlaps its neighbour a little more than the rest.
    """
    if not tile or (height <= tile and width <= tile):
        return [(0, 0, height, width)]
    step = max(1, int(round(tile * (1.0 - overlap))))

    def starts(total):
        if total <= tile:
            return [0]
        offsets = list(range(0, total - tile + 1, step))
        if offsets[-1] != total - tile:
            offsets.append(total - tile)
        return offsets

    return [(y, x, min(y + tile, height), min(x + tile, width))
            for y in starts(height) for x in starts(width)]


def describe(shape, boxes, tile: int, overlap: float) -> str:
    height, width = shape[:2]
    if len(boxes) == 1:
        return f"{width}x{height}: one pass over the whole frame"
    return (f"{width}x{height}: {len(boxes)} tiles of {tile} px, "
            f"overlap {overlap:.0%} ({int(round(tile * overlap))} px)")


# Putting the pieces back


def _overlap_pixels(a: Found, b: Found) -> int:
    """How many pixels two masks share, counted only where their boxes cross."""
    ay0, ax0, ay1, ax1 = a.bounds
    by0, bx0, by1, bx1 = b.bounds
    y0, x0 = max(ay0, by0), max(ax0, bx0)
    y1, x1 = min(ay1, by1), min(ax1, bx1)
    if y1 <= y0 or x1 <= x0:
        return 0
    inside_a = a.mask[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0]
    inside_b = b.mask[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]
    return int(np.count_nonzero(inside_a & inside_b))


def _union(items: list[Found]) -> Found:
    """One instance from several, the mask their union and the score the best of them."""
    bounds = [item.bounds for item in items]
    y0 = min(b[0] for b in bounds)
    x0 = min(b[1] for b in bounds)
    y1 = max(b[2] for b in bounds)
    x1 = max(b[3] for b in bounds)
    mask = np.zeros((y1 - y0, x1 - x0), bool)
    for item in items:
        iy0, ix0, iy1, ix1 = item.bounds
        mask[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] |= item.mask
    best = max(items, key=lambda item: item.score)
    return best._replace(mask=mask, origin=(y0, x0))


def touches_seam(extent, tile, shape, margin: int = 2) -> bool:
    """Whether something runs into a tile edge that is not also the frame edge.

    This is what tells a cut object from a whole one. A detection that stops short of
    every tile edge is all the model had to say about it; one that runs into a seam
    was interrupted there, and its other half is in the neighbouring tile.
    """
    x0, y0, x1, y1 = extent
    ty0, tx0, ty1, tx1 = tile
    height, width = shape[:2]
    return ((tx0 > 0 and x0 <= tx0 + margin)
            or (ty0 > 0 and y0 <= ty0 + margin)
            or (tx1 < width and x1 >= tx1 - margin)
            or (ty1 < height and y1 >= ty1 - margin))


def _extent(item: Found):
    if item.box is not None:
        return item.box
    y0, x0, y1, x1 = item.bounds
    return x0, y0, x1, y1


def merge_masks(found: list[Found], tiles: list, shape, min_overlap_px: int = 2,
                containment: float = 0.5) -> list[Found]:
    """Union same-class masks from different tiles that are really one object.

    Two things have to be caught, and they need different rules:

      * the same cell seen twice, whole in one tile and truncated in another. Nearly
        all of the truncated one lies inside the whole one, so `containment` -- the
        shared pixels as a fraction of the smaller mask -- settles it.
      * a cell longer than the overlap is deep, which no tile ever holds whole. Its
        halves share only the sliver inside the overlap zone, far below any sensible
        fraction, so here a couple of pixels is enough. That rule is only allowed
        where one of the two runs into the seam, which is what says it was cut.

    Letting the two-pixel rule apply everywhere is how the amodal model loses cells:
    its masks are drawn through each other on purpose, so two cells crossing share
    real pixels and would be fused into one. Only instances from DIFFERENT tiles are
    ever compared, so two cells lying against each other inside one tile stay two.
    """
    masked = [item for item in found if item.mask is not None]
    rest = [item for item in found if item.mask is None]
    if len(masked) <= 1:
        return found

    cut = [touches_seam(_extent(item), tiles[item.source], shape) for item in masked]
    areas = [int(item.mask.sum()) for item in masked]
    parent = list(range(len(masked)))

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i, a in enumerate(masked):
        for j in range(i + 1, len(masked)):
            b = masked[j]
            if a.label != b.label or a.source == b.source or root(i) == root(j):
                continue
            shared = _overlap_pixels(a, b)
            if not shared:
                continue
            smaller = min(areas[i], areas[j]) or 1
            if (shared / smaller >= containment
                    or ((cut[i] or cut[j]) and shared > max(0, int(min_overlap_px)))):
                parent[root(j)] = root(i)

    groups: dict[int, list[Found]] = {}
    for index, item in enumerate(masked):
        groups.setdefault(root(index), []).append(item)
    merged = [group[0] if len(group) == 1 else _union(group) for group in groups.values()]
    return sorted(merged + rest, key=lambda item: -item.score)


def merge_boxes(found: list[Found], tiles: list, shape, iou: float = 0.5,
                containment: float = 0.7) -> list[Found]:
    """Per-class NMS across tiles, for a detector that has boxes and no masks.

    A box has no shape to union, so the best-scoring copy wins and the rest go -- which
    is right for a whole-cell box, and would be wrong for a mask.

    IoU alone does not do it. A box clipped at a tile edge covers about half the cell,
    so its IoU with the whole box is about a half whatever threshold is set, and the
    cell ends up boxed twice: once properly and once truncated. What identifies it is
    that nearly all of the clipped box lies inside the whole one, so a box that runs
    into a seam is judged on containment instead. A clipped box already kept is
    replaced when the fuller one turns up later, so the answer does not depend on
    which of the two happened to score higher.

    Only boxes from different tiles are compared: ultralytics has already run NMS
    inside each tile, and a second, stricter pass over that just deletes cells that
    are genuinely lying against each other.
    """
    boxed = [item for item in found if item.box is not None]
    rest = [item for item in found if item.box is None]
    kept: list[Found] = []
    for item in sorted(boxed, key=lambda item: -item.score):
        clipped = touches_seam(item.box, tiles[item.source], shape)
        drop = False
        for index, other in enumerate(kept):
            if other.label != item.label or other.source == item.source:
                continue
            if _iou(item.box, other.box) >= iou:
                drop = True
                break
            if clipped and _inside(item.box, other.box) >= containment:
                drop = True
                break
            if (touches_seam(other.box, tiles[other.source], shape)
                    and _inside(other.box, item.box) >= containment):
                kept[index] = item  # the fuller box takes the truncated one's place
                drop = True
                break
        if not drop:
            kept.append(item)
    return sorted(kept + rest, key=lambda item: -item.score)


def _inside(a, b) -> float:
    """How much of box `a` lies within box `b`, as a fraction of `a`."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    shared = (max(0.0, min(ax1, bx1) - max(ax0, bx0))
              * max(0.0, min(ay1, by1) - max(ay0, by0)))
    area = (ax1 - ax0) * (ay1 - ay0)
    return shared / area if area > 0 else 0.0


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    shared = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if shared <= 0:
        return 0.0
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - shared
    return shared / union if union > 0 else 0.0


def native_imgsz(shape, tile: int) -> int:
    """The imgsz that puts a crop in front of the model at 1.0x.

    The tile size normally. Where the frame is smaller than one tile there is a single
    crop and it is the frame, so its own long side instead -- rounded up to the
    multiple of 32 ultralytics insists on, rather than letting it letterbox a 512 px
    frame up to 640 and present every cell 25% too large.
    """
    edge = min(int(tile), max(shape[:2])) if tile else max(shape[:2])
    return int(-(-int(edge) // 32) * 32)


def add_tiling_arguments(parser) -> None:
    """The tiling flags, shared so no two scripts can spell them differently."""
    parser.add_argument("--tile", type=int, default=None,
                        help="tile size in px, or 0 for one pass over the whole frame. "
                             "Default is what the index records")
    parser.add_argument("--overlap", type=float, default=None,
                        help=f"tile overlap as a fraction of the tile, 0 to 0.9 "
                             f"(default {DEFAULT_OVERLAP} where the index says nothing)")
    parser.add_argument("--batch", type=int, default=4,
                        help="tiles per forward pass; lower it if the card runs out")


def batches(boxes: list, size: int):
    """The tile boxes in groups of `size`, one forward pass each.

    Yields (index of the first tile in the group, the group), because what came out of
    which tile is what the merging keys on.
    """
    size = max(1, int(size))
    for start in range(0, len(boxes), size):
        yield start, boxes[start:start + size]
