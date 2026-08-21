#!/usr/bin/env python
"""Running a model over a frame in tiles, and putting the pieces back together.

**Nothing here happens by default.** Every script runs one plain pass: the whole frame
into the model at the input size the index records for it. Tiling is opt-in, because
it is a real change to what the model is shown and to what comes out, and both should
be asked for rather than arrived at.

What asking for it buys is scale. A model handed a frame larger than its input fits
that frame's long side to it, so every cell reaches the model shrunk by the same
factor, and the smallest go with it. Cutting the frame into tiles the size of that
input and running each puts every cell back at 1.0x. The cost is that an object
straddling a tile edge is found twice, or in halves, so the pieces have to be rejoined
afterwards -- which is the rest of this file, and none of it is free.

`--tile N` turns it on; the run says what N the index records for the model. `--guide`
places those tiles on the cells a detector found instead of gridding them, which is
the version of this with nothing to rejoin. **Overlap is a fraction of the tile**, and
on a grid it has to be wider than the longest object: a cell longer than the overlap
is never whole in any tile, and no amount of merging afterwards invents the part that
was never seen.
"""

from __future__ import annotations

import numpy as np

from outputs import Found, say

# Where an entry records no overlap. A quarter of a tile is comfortably longer than a
# promastigote with its flagellum at the magnifications these models are for.
DEFAULT_OVERLAP = 0.25

# Where `--stitch merge` has to decide whether two boxes from neighbouring tiles are
# one cell. Only the box merger has any use for it; see `stitch`.
DEFAULT_IOU = 0.5

# How many times `cover` alternates dropping redundant tiles and recentring what is
# left. It settles in one or two; the cap is only so a pathological plan cannot spin.
_SETTLE = 4


def settings(entry: dict, tile: int | None, overlap: float | None) -> tuple[int, float]:
    """The tile size and overlap to use: the number given, else what the index says.

    0 is the whole frame in one pass and comes back as 0, which is what every script
    defaults to: a model is shown the frame as it is unless tiling is asked for. None
    is what asks for the index's own recorded tile, and is reached by `--guide`, tiling
    being the whole point of that one.
    """
    recorded = (entry.get("config") or {}).get("tiling") or {}
    asked = tile is not None, overlap is not None
    if tile is None:
        tile = recorded.get("tile") or 0
    if overlap is None:
        # `is None`, not falsy: an entry recording no overlap at all wants the default,
        # and one recording zero wants zero.
        overlap = recorded.get("overlap")
        overlap = DEFAULT_OVERLAP if overlap is None else overlap
    where = "--tile" if asked[0] else "the tile the index records for this model"
    if int(tile) < 0:
        raise SystemExit(f"{where} is a size in pixels, or 0 for one pass over the "
                         f"whole frame; got {tile}")
    where = "--overlap" if asked[1] else "the overlap the index records for this model"
    if not 0.0 <= overlap < 0.9:
        raise SystemExit(f"{where} is a fraction of the tile, 0 to below 0.9; "
                         f"got {overlap}")
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


# Tiles chosen for what is actually there


def cover(boxes: list, shape, tile: int, pad: int = 0) -> tuple[list, list]:
    """Tiles holding every box whole, and as few of them as they can be.

    A grid is laid down without knowing where anything is, so it cuts through cells
    and everything after it is repair. Given the boxes a detector found, the tiles can
    instead be put where the cells are: each one inside a tile in one piece, and the
    only thing left to notice afterwards is that two tiles saw the same cell.

    A tile at origin `o` holds a box whole exactly when `far - tile <= o <= near`, so
    each box is a RANGE of origins on each axis and the question is the fewest points
    that fall inside every range. That is interval stabbing, which is solved exactly in
    one dimension -- so it is done twice: once down the frame to fix the rows, then
    once across each row. The result is optimal along each axis rather than over the
    plane, the latter being NP-hard.

    `pad` asks for that much clear space around every cell as well, so a cell sits
    inside its tile rather than against the edge of it. It is given up where the frame
    edge leaves nowhere to put it, since a cell there is at a real border and not a
    seam, and where one tile serves cells too far apart to pad all of them.

    Boxes are expected to lie inside the frame, which is what a detector run over the
    frame produces. One reaching past an edge is only held where the frame is longer
    than a tile on that axis.

    -> (tiles, the boxes longer than one tile, which no tile can ever hold whole)
    """
    height, width = shape[:2]
    if not tile:
        return [(0, 0, height, width)], []

    fitted, oversize = [], []
    for box in boxes:
        x0, y0, x1, y1 = box
        down = _origins(y0, y1, height, tile, pad)
        across = _origins(x0, x1, width, tile, pad)
        if down is None or across is None:
            oversize.append(box)
        else:
            fitted.append((down, across, box))

    # The rows first, then each row's own columns. A box that several rows could hold
    # goes to the first of them, so later rows stay as empty as they can be.
    rows = _stab([down for down, _, _ in fitted])
    mine: dict = {y: [] for y in rows}
    for down, across, _ in fitted:
        for y in rows:
            if down[0] <= y <= down[1]:
                mine[y].append(across)
                break
    tiles = [_tile_at(y, x, shape, tile) for y in rows for x in _stab(mine[y])]

    held = [box for _, _, box in fitted]
    # Recentring moves tiles, and moving them changes which boxes each holds: two that
    # were distinct before the move can end up holding the same set, and a tile dropped
    # for that reason hands its boxes to one that was centred without them. Both passes
    # are cheap, so they run until neither changes anything, ending on a recentre so no
    # tile is left centred on boxes it no longer has.
    for _ in range(_SETTLE):
        settled = _recentre(_drop_redundant(tiles, held), held, shape, tile, pad)
        if settled == tiles:
            break
        tiles = settled
    # A cell longer than the tile fits in none of them. It still has to be looked at,
    # so it gets a tile of its own with as much of it inside as there is room for.
    tiles += [_tile_at(*_middle_of(box, shape, tile), shape, tile) for box in oversize]
    return list(dict.fromkeys(tiles)), oversize


def assign(tiles: list, boxes: list) -> list:
    """Each box paired with the tile that owns it: the first one holding it whole.

    The plan's own answer to "which tile is this cell's". Made once, from the boxes, so
    that `stitch_by_plan` can settle every copy of a cell by the CELL rather than by
    the copy -- a copy's own geometry moves when a tile cuts it and its centroid moves
    with it, and a box does not move at all.

    A box no tile holds whole is an oversize one; it goes to whichever tile has most of
    it, which is the tile `cover` centred on it.
    """
    owners = []
    for box in boxes:
        holder = next((i for i, tile in enumerate(tiles) if _holds(tile, box)), None)
        if holder is None and tiles:
            holder = max(range(len(tiles)), key=lambda i: _box_overlap(tiles[i], box))
        if holder is not None:
            owners.append((box, holder))
    return owners


def _box_overlap(tile, box) -> float:
    """Area a tile and a box share. Tiles are (y0, x0, y1, x1) and boxes (x0, y0, ...)."""
    ty0, tx0, ty1, tx1 = tile
    x0, y0, x1, y1 = box
    return (max(0.0, min(x1, tx1) - max(x0, tx0))
            * max(0.0, min(y1, ty1) - max(y0, ty0)))


def _stab(spans: list) -> list:
    """The fewest points putting one inside every (lo, hi). Optimal, which is the
    whole reason `cover` works an axis at a time rather than on the plane.

    Take the range that ends soonest and put a point at its end: no point further on
    could reach it, and none further back reaches as much of what follows.
    """
    points: list = []
    for lo, hi in sorted(spans, key=lambda span: span[1]):
        if not points or points[-1] < lo:
            points.append(hi)
    return points


def _origins(near: float, far: float, total: int, tile: int, pad: int):
    """(lo, hi): the origins of a `tile` that hold [near, far] whole, or None."""
    near, far = int(np.floor(near)), int(np.ceil(far))
    room = max(0, total - tile)
    lo, hi = max(0, far - tile), min(room, near)
    if lo > hi:
        return None  # longer than the tile; no origin holds it
    want = min(int(pad), (tile - (far - near)) // 2)
    inner = max(lo, far + want - tile), min(hi, near - want)
    return inner if want > 0 and inner[0] <= inner[1] else (lo, hi)


def _tile_at(y: int, x: int, shape, tile: int) -> tuple:
    return int(y), int(x), int(min(y + tile, shape[0])), int(min(x + tile, shape[1]))


def _holds(tile, box) -> bool:
    """Whether a tile has a box inside it in one piece."""
    ty0, tx0, ty1, tx1 = tile
    x0, y0, x1, y1 = box
    return tx0 <= x0 and ty0 <= y0 and x1 <= tx1 and y1 <= ty1


def _drop_redundant(tiles: list, boxes: list) -> list:
    """Tiles not one of whose boxes needs them.

    Rows and columns are each minimal on their own axis, which does not make the
    rectangle they produce minimal: a tile can end up holding only cells that some
    other tile already holds whole. Tried least-useful first, so dropping one cannot
    make a more useful one look redundant afterwards.
    """
    tiles = list(dict.fromkeys(tiles))
    held = [{i for i, box in enumerate(boxes) if _holds(tile, box)} for tile in tiles]
    kept = set(range(len(tiles)))
    for index in sorted(range(len(tiles)), key=lambda i: len(held[i])):
        elsewhere = set().union(*[held[j] for j in kept if j != index] or [set()])
        if held[index] <= elsewhere:
            kept.discard(index)
    return [tile for index, tile in enumerate(tiles) if index in kept]


def _recentre(tiles: list, boxes: list, shape, tile: int, pad: int = 0) -> list:
    """Move each tile to the middle of the room its own boxes leave it.

    Stabbing puts a tile's edge exactly on some box's edge, which is the worst place
    for it: a cell lying against a tile edge is what a model reads as a cut one. Every
    origin between a tile's tightest boxes holds the same set of them, so the middle of
    that range is margin on both sides for nothing. Tiles left holding nothing go.
    """
    height, width = shape[:2]
    mine: dict = {}
    for box in boxes:
        for index, candidate in enumerate(tiles):
            if _holds(candidate, box):
                mine.setdefault(index, []).append(box)
                break
    moved = []
    for index in sorted(mine):
        held = mine[index]
        down = _padded(held, 1, 3, height, tile, pad)
        across = _padded(held, 0, 2, width, tile, pad)
        moved.append(_tile_at(sum(down) // 2, sum(across) // 2, shape, tile))
    return moved


def _padded(boxes: list, near: int, far: int, total: int, tile: int, pad: int) -> tuple:
    """The origins holding every box with `pad` clear of it, else merely holding them.

    Asked of each box separately and intersected, rather than of the one box that
    spans them all: `pad` is clear space around a CELL, and the room around the hull of
    several is not the room around any of them.

    Where the boxes lie too far apart for one tile to give all of them the full `pad`,
    they get the widest margin it can give all of them rather than none -- the ranges
    only tighten as the margin grows, so that is a bisection, and a margin of nothing
    always works because the tile holds every one of these boxes already.
    """
    def reach(want):
        spans = [_origins(box[near], box[far], total, tile, want) for box in boxes]
        return max(s[0] for s in spans), min(s[1] for s in spans)

    lo, hi = reach(pad)
    if lo <= hi:
        return lo, hi
    low, high = 0, int(pad)
    while low < high:
        middle = (low + high + 1) // 2
        first, last = reach(middle)
        low, high = (middle, high) if first <= last else (low, middle - 1)
    lo, hi = reach(low)
    return (lo, hi) if lo <= hi else reach(0)


def _middle_of(box, shape, tile: int) -> tuple:
    """The origin centring a tile on a box too long to fit inside one."""
    x0, y0, x1, y1 = box
    return (min(max(0, shape[0] - tile), max(0, int(round((y0 + y1 - tile) / 2)))),
            min(max(0, shape[1] - tile), max(0, int(round((x0 + x1 - tile) / 2)))))


def describe_cover(shape, tiles: list, boxes: list, oversize: list, tile: int,
                   overlap: float) -> str:
    """What the plan came to, against the grid it replaces."""
    height, width = shape[:2]
    grid = len(slices(height, width, tile, overlap))
    whole = sum(1 for box in boxes if any(_holds(t, box) for t in tiles))
    # "cells" is what the DETECTOR found, not what is there. Said that way round
    # because a guided run can only place tiles on what it was shown.
    line = (f"{width}x{height}: {len(tiles)} guided tiles of {tile} px for "
            f"{len(boxes)} detected cells, {whole} of them whole "
            f"({grid} on the grid at overlap {overlap:.0%})")
    if oversize:
        longest = max(max(b[2] - b[0], b[3] - b[1]) for b in oversize)
        line += (f"; {len(oversize)} longer than the tile and still cut, so "
                 f"--tile {int(longest) + 32} or more would hold them")
    return line


# Putting the pieces back


def _band(tile_a, tile_b):
    """The strip two tiles have in common, or None where they do not meet."""
    y0, x0 = max(tile_a[0], tile_b[0]), max(tile_a[1], tile_b[1])
    y1, x1 = min(tile_a[2], tile_b[2]), min(tile_a[3], tile_b[3])
    return (y0, x0, y1, x1) if y1 > y0 and x1 > x0 else None


def _within(item: Found, band) -> tuple:
    """An instance's mask cut down to a rectangle, and where that piece sits."""
    by0, bx0, by1, bx1 = band
    iy0, ix0, iy1, ix1 = item.bounds
    y0, x0 = max(iy0, by0), max(ix0, bx0)
    y1, x1 = min(iy1, by1), min(ix1, bx1)
    if y1 <= y0 or x1 <= x0:
        return None, (0, 0)
    return item.mask[y0 - iy0:y1 - iy0, x0 - ix0:x1 - ix0], (y0, x0)


def _agree_in_band(a: Found, b: Found, band, agreement: float) -> bool:
    """Whether two masks from neighbouring tiles are two views of one object.

    Judged only inside the strip the two tiles share, because that strip is the one
    ground both models actually looked at. Two copies of one object agree there almost
    exactly, however differently the rest of it was cut off: whether one tile saw the
    object whole or neither did, both saw all of it that lies in the strip.

    The agreement is symmetric -- shared pixels over what either of them has in the
    strip. Asking instead what fraction of the SMALLER one is shared says yes to any
    object barely reaching into the strip across something else, since almost all of
    the little it has there is then shared. Comparing the whole masks is weaker still:
    most of each lies outside the strip, where the other tile saw nothing at all.
    """
    mine, at_mine = _within(a, band)
    theirs, at_theirs = _within(b, band)
    if mine is None or theirs is None:
        return False
    y0, x0 = max(at_mine[0], at_theirs[0]), max(at_mine[1], at_theirs[1])
    y1 = min(at_mine[0] + mine.shape[0], at_theirs[0] + theirs.shape[0])
    x1 = min(at_mine[1] + mine.shape[1], at_theirs[1] + theirs.shape[1])
    if y1 <= y0 or x1 <= x0:
        return False
    both = (mine[y0 - at_mine[0]:y1 - at_mine[0], x0 - at_mine[1]:x1 - at_mine[1]]
            & theirs[y0 - at_theirs[0]:y1 - at_theirs[0],
                     x0 - at_theirs[1]:x1 - at_theirs[1]])
    shared = int(np.count_nonzero(both))
    either = int(mine.sum()) + int(theirs.sum()) - shared
    return bool(shared) and shared / (either or 1) >= agreement


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
    # The box came from one member and describes that member alone, so it would
    # disagree with the mask now under it. `_extent` falls back to the mask's own
    # bounds, which is the one thing here that is true of the union.
    return best._replace(mask=mask, origin=(y0, x0), box=None)


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
    """(x0, y0, x1, y1) around whatever geometry an instance has, or None if it has
    none: a chain no node of which the model saw encloses nothing."""
    if item.box is not None:
        return item.box
    if item.mask is not None:
        y0, x0, y1, x1 = item.bounds
        return x0, y0, x1, y1
    seen = item.line[np.isfinite(item.line).all(1)] if item.line is not None else []
    if not len(seen):
        return None
    return (float(seen[:, 0].min()), float(seen[:, 1].min()),
            float(seen[:, 0].max()), float(seen[:, 1].max()))


def merge_masks(found: list[Found], tiles: list, shape,
                agreement: float = 0.5) -> list[Found]:
    """Union same-class masks from different tiles that are really one object.

    Both cases this has to catch -- the same cell seen whole in one tile and truncated
    in another, and a cell longer than the overlap that no tile holds whole -- come
    down to one question: do the two masks agree about the strip their tiles share?
    That strip is the only ground both models looked at, so it is the only place their
    answers can be compared. `agreement` is how much of what either of them has there
    the two of them share. See `_agree_in_band`.

    Judging on the whole masks instead is a bare count of shared pixels against the
    smaller area, and that is what chains a crowded frame into one blob: a stray
    fragment scores a perfect containment against whatever it lies on, and the joins
    are transitive.

    Only instances from DIFFERENT tiles are ever compared, so two cells lying against
    each other inside one tile stay two -- which is what keeps the amodal model's
    masks, drawn through each other on purpose, from being fused.
    """
    masked = [item for item in found if item.mask is not None]
    rest = [item for item in found if item.mask is None]
    if len(masked) <= 1:
        return sorted(found, key=lambda item: -item.score)

    parent = list(range(len(masked)))

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    bands = {}
    for i, a in enumerate(masked):
        for j in range(i + 1, len(masked)):
            b = masked[j]
            if a.label != b.label or a.source == b.source or root(i) == root(j):
                continue
            pair = (a.source, b.source)
            if pair not in bands:
                bands[pair] = _band(tiles[a.source], tiles[b.source])
            band = bands[pair]
            if band and _agree_in_band(a, b, band, agreement):
                parent[root(j)] = root(i)

    groups: dict[int, list[Found]] = {}
    for index, item in enumerate(masked):
        groups.setdefault(root(index), []).append(item)
    merged = [group[0] if len(group) == 1 else _union(group) for group in groups.values()]
    return sorted(merged + rest, key=lambda item: -item.score)


def merge_boxes(found: list[Found], tiles: list, shape, iou: float = DEFAULT_IOU,
                containment: float = 0.7) -> list[Found]:
    """Per-class NMS across tiles, for a detector that has boxes and no masks.

    A box has no shape to union, so one copy of each cell wins and the rest go -- which
    is right for a whole-cell box, and would be wrong for a mask.

    IoU alone does not do it. A box clipped at a tile edge covers about half the cell,
    so its IoU with the whole box is about a half whatever threshold is set, and the
    cell ends up boxed twice: once properly and once truncated. What identifies it is
    that nearly all of the clipped box lies inside the whole one, so a box that runs
    into a seam is judged on containment instead.

    The boxes are taken largest first, so the copy that survives a cell is its fullest
    one. Deciding by score instead makes the answer depend on which copy the model
    happened to be more sure of: a truncated copy can outscore the cell it was cut
    from, and then it is the whole box that gets deleted.

    Only boxes from different tiles are compared: ultralytics has already run NMS
    inside each tile, and a second, stricter pass over that just deletes cells that
    are genuinely lying against each other.
    """
    boxed = [item for item in found if item.box is not None]
    rest = [item for item in found if item.box is None]
    kept: list[Found] = []
    for item in sorted(boxed, key=lambda item: (-_area(item.box), -item.score)):
        if not any(_same_cell(other, item, tiles, shape, iou, containment)
                   for other in kept):
            kept.append(item)
    return sorted(kept + rest, key=lambda item: -item.score)


def _area(box) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _same_cell(kept: Found, item: Found, tiles: list, shape, iou: float,
               containment: float) -> bool:
    """Whether `item` is another tile's copy of the cell `kept` already stands for.

    `kept` is never the smaller of the two, so the only question is whether `item` adds
    anything: it does not if the two boxes agree, nor if it is `kept` clipped at a seam.
    """
    if kept.label != item.label or kept.source == item.source:
        return False
    if _iou(kept.box, item.box) >= iou:
        return True
    return (_inside(item.box, kept.box) >= containment
            and touches_seam(item.box, tiles[item.source], shape))


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


def scale_note(boxes: list, imgsz: int) -> str:
    """What scale cells actually reach the model at.

    Ultralytics fits the LONGEST side of whatever it is handed to `imgsz`, so it is the
    ratio of the two that is the scale and the imgsz alone says nothing without it.
    Rounding the frame up to the multiple of 32 ultralytics insists on is why a whole
    frame comes out a per cent over 1.00 rather than exactly on it.

    Measured off the crops that will actually go in, rather than inferred from the tile
    size and the number of them. Guided tiles are placed, so there can be exactly one
    of them and it is a tile rather than the frame -- which the count alone cannot tell
    apart from an untiled pass.
    """
    longest = max(max(y1 - y0, x1 - x0) for y0, x0, y1, x1 in boxes)
    return f"imgsz {imgsz}: cells at {imgsz / longest:.2f}x native"


def recorded_tile(entry: dict) -> int:
    """The tile size the index records for a model, or 0 where it records none."""
    return int(((entry.get("config") or {}).get("tiling") or {}).get("tile") or 0)


def recorded_imgsz(entry: dict, shape) -> int:
    """The input size the index records for a model.

    This is what an untiled pass runs at: the frame goes in whole and at the size the
    network wants, which is the plainest thing that can be asked of a model and the
    only size it is known to behave at. Ultralytics fits the LONGEST side to it, so a
    frame bigger than it arrives scaled down and `scale_note` says by how much.

    Falling back to the frame's own long side where an entry records nothing is a
    guess, but the alternative is having no number at all.
    """
    recorded = int(((entry.get("config") or {}).get("detection") or {}).get("imgsz") or 0)
    return recorded or native_imgsz(shape, 0)


def tiling_hint(entry: dict, tile: int) -> str | None:
    """Where a run is not tiling, what the index says it could tile at.

    The default is one plain pass, so the tile size a model was actually trained at
    would otherwise be recorded in the index and never mentioned by anything that runs
    it. Said once, on the first frame, rather than left to be looked up.
    """
    recorded = recorded_tile(entry)
    if tile or not recorded:
        return None
    return (f"not tiling; --tile {recorded} is what the index records, and is what "
            f"puts cells at 1.0x rather than at the scale above (--guide places those "
            f"tiles on the cells rather than gridding them)")


def native_imgsz(shape, tile: int) -> int:
    """The imgsz that puts a crop in front of the model at 1.0x.

    The tile size normally. Where the frame is smaller than one tile there is a single
    crop and it is the frame, so its own long side instead -- rounded up to the
    multiple of 32 ultralytics insists on, rather than letting it letterbox a 512 px
    frame up to 640 and present every cell 25% too large.
    """
    edge = min(int(tile), max(shape[:2])) if tile else max(shape[:2])
    return int(-(-int(edge) // 32) * 32)


def _core_axis(starts: list, ends: list, total: int) -> dict:
    """Each tile edge pair -> the half-open span it owns along one axis.

    The spans partition the axis: neighbours meet in the middle of the strip they
    share, so every coordinate belongs to exactly one tile and none to two.

    Keyed on both edges rather than on the start alone, so two tiles that begin
    together and end apart cannot collapse into one entry and hand the same span to
    both -- which would leave one of them owning ground it does not cover.
    """
    order = sorted(set(zip(starts, ends)))
    spans = {}
    for index, (start, end) in enumerate(order):
        low = 0 if index == 0 else (start + order[index - 1][1]) // 2
        high = total if index == len(order) - 1 else (end + order[index + 1][0]) // 2
        spans[start, end] = (low, high)
    return spans


def _centroid(item: Found):
    """Where an instance sits, whatever geometry it has: mask, box or chain.

    None where it has nowhere to be, which is a chain none of whose nodes the model
    saw. There is no ground for a tile to own it by.
    """
    if item.mask is not None:
        rows, cols = np.nonzero(item.mask)
        return item.origin[0] + rows.mean(), item.origin[1] + cols.mean()
    if item.box is not None:
        x0, y0, x1, y1 = item.box
        return (y0 + y1) / 2.0, (x0 + x1) / 2.0
    # A chain: the mean of the nodes that were visible. NaN is how an invisible one is
    # carried, and a chain with none at all has nowhere to be -- None, so that every
    # scheme drops it rather than putting it somewhere arbitrary and owning it there.
    if item.line is not None and np.isfinite(item.line[:, 0]).any():
        return (float(np.nanmean(item.line[:, 1])), float(np.nanmean(item.line[:, 0])))
    return None


def stitch_by_core(found: list[Found], tiles: list, shape) -> list[Found]:
    """Keep each instance in the one tile that owns the ground it stands on.

    The alternative to merging, and what StarDist's `predict_instances_big` does. Each
    tile owns a core, the cores partition the frame, and an instance is kept only by
    the tile whose core holds its centroid; the rest of the tile is context, there so
    the model sees a whole object near its core's edge rather than a cut one.

    Nothing is ever compared with anything, so nothing can be double-counted, and no
    chain of pairwise joins can fuse a crowd into one blob -- which is the failure
    merging has to be defended against. It leaves more instances cut at a seam than
    merging does, and no mask larger than one a tile actually saw.

    What it needs in exchange is that an object whose centroid sits in the core is
    wholly inside the tile -- so the margin from core to tile edge has to exceed the
    object's reach. Where that fails the object is kept cut rather than joined, and an
    object cut across two cores is kept twice; see `warn_if_longer_than_overlap`.
    """
    rows = _core_axis([t[0] for t in tiles], [t[2] for t in tiles], shape[0])
    cols = _core_axis([t[1] for t in tiles], [t[3] for t in tiles], shape[1])
    kept = []
    for item in found:
        if item.mask is None and item.box is None and item.line is None:
            kept.append(item)
            continue
        centre = _centroid(item)
        if centre is None:
            continue
        tile = tiles[item.source]
        low_y, high_y = rows[tile[0], tile[2]]
        low_x, high_x = cols[tile[1], tile[3]]
        centre_y, centre_x = centre
        if low_y <= centre_y < high_y and low_x <= centre_x < high_x:
            kept.append(item)
    return sorted(kept, key=lambda item: -item.score)


def stitch_by_plan(found: list[Found], tiles: list, shape, plan: list) -> list[Found]:
    """Keep each instance in the tile the PLAN gave its cell.

    Ownership again, but settled by the detector's box rather than by the instance's
    own centroid -- which is what makes it hold. `cover` placed one tile per cell and
    `assign` recorded which; every copy of that cell, the whole one and any a
    neighbouring tile cut, lies inside the same box and so comes back with the same
    owner. Exactly one copy survives, and it is the one from the tile the cell was
    placed in, which is the tile that holds it whole.

    `stitch_by_whole` decides the same question from the copy in hand, and that is the
    difference: a copy CUT by a tile edge has its centroid pulled away from that edge,
    the two halves of a straddling cell land on opposite sides of the boundary between
    their tiles, and both elect themselves. Deciding by the cell keeps one of them.

    An instance no box explains -- something the segmenter found and the detector did
    not -- has no cell to be owned by, so it falls back to the tile it sits most
    centrally in. Those are the cells `--guide-conf` is for, and the run says how many
    there were.

    The cell each instance was matched to is kept on it, which is the point of doing it
    this way round: an animal, its body and its flagellum come out of three separate
    predictions carrying one number, so the outputs can say that they are one cell
    rather than three things that happen to overlap. See `outputs.numbers_for`.
    """
    kept = []
    for item in found:
        if item.mask is None and item.box is None and item.line is None:
            kept.append(item)
            continue
        matched = _planned_cell(item, plan)
        if matched is None:
            if _most_central(item, tiles, shape) == item.source:
                kept.append(item)
            continue
        box, owner = plan[matched]
        # The owner is the tile the cell was placed in, so it holds the cell whole and
        # every copy of it agrees on that. A cell longer than a tile has no such tile:
        # `assign` gave it whichever holds most of it, and a piece from any other tile
        # would then answer to an owner that never saw it, so those are settled by
        # where they sit instead of being dropped for disagreeing.
        settled = (item.source == owner if _holds(tiles[owner], box)
                   else _most_central(item, tiles, shape) == item.source)
        if settled:
            kept.append(item._replace(cell=matched + 1))
    return sorted(kept, key=lambda item: -item.score)


def _planned_cell(item: Found, plan: list):
    """Which of the plan's cells this instance is part of, as an index, or None.

    Scored on how much of the instance lies inside the box, because that is what being
    part of a cell means. The centroid only breaks ties between boxes that hold equal
    shares of it: a thin instance lying diagonally across a crowd can have its own
    centroid over a neighbouring cell, and deciding on the centroid first therefore
    puts the body and the flagellum of one cell on two different numbers.
    """
    extent = _extent(item)
    if extent is None:
        return None
    x0, y0, x1, y1 = extent
    span = max(1e-9, (x1 - x0) * (y1 - y0))
    centre = _centroid(item)
    best, most, inside = None, 0.0, False
    for index, (box, _) in enumerate(plan):
        bx0, by0, bx1, by1 = box
        shared = (max(0.0, min(x1, bx1) - max(x0, bx0))
                  * max(0.0, min(y1, by1) - max(y0, by0)))
        if shared <= 0:
            continue
        held = (centre is not None
                and bx0 <= centre[1] <= bx1 and by0 <= centre[0] <= by1)
        if (shared / span, held) > (most, inside):
            best, most, inside = index, shared / span, held
    return best


def _most_central(item: Found, tiles: list, shape):
    """The tile an instance sits furthest from the seams of. See `stitch_by_whole`."""
    centre = _centroid(item)
    if centre is None:
        return None
    centre_y, centre_x = centre
    best, most = None, None
    for index, tile in enumerate(tiles):
        ty0, tx0, ty1, tx1 = tile
        if not (ty0 <= centre_y < ty1 and tx0 <= centre_x < tx1):
            continue
        room = _clearance(centre_y, centre_x, tile, shape)
        if most is None or room > most:
            best, most = index, room
    return best


def _clearance(centre_y: float, centre_x: float, tile, shape) -> float:
    """How far a point is from the nearest SEAM of a tile, ignoring the frame's edge.

    A tile edge that is also the frame's edge is not a seam: nothing was cut there and
    there is no neighbouring tile holding the rest. Counting it would make every tile
    along a border equally cramped, so the maximum degenerates into a tie and ownership
    falls to the lowest tile index -- which is the one holding the cell hardest against
    its own real seam, and so the one with the most truncated copy of it.
    """
    ty0, tx0, ty1, tx1 = tile
    gaps = []
    if ty0 > 0:
        gaps.append(centre_y - ty0)
    if tx0 > 0:
        gaps.append(centre_x - tx0)
    if ty1 < shape[0]:
        gaps.append(ty1 - centre_y)
    if tx1 < shape[1]:
        gaps.append(tx1 - centre_x)
    return min(gaps) if gaps else float("inf")


def stitch_by_whole(found: list[Found], tiles: list, shape) -> list[Found]:
    """Keep each instance in the one tile it sits most comfortably inside.

    Ownership, like `stitch_by_core`, and for the same reason: nothing is compared with
    anything, so nothing CAN be double-counted. What it replaces is a core, which only
    partitions the frame while the tiles are a regular grid; guided tiles are not one,
    so the partition has to come from the tiles themselves. Every tile holding an
    instance's centroid is a candidate, and the one where that centroid has the most
    room to the tile's own edges wins, ties going to the earlier tile.

    It is decided from the centroid alone, and the ownership itself is an exact
    partition: every centroid in the frame has one winner, never none and never two.

    What it does NOT guarantee is that two copies of one cell agree on their centroid.
    A copy CUT by a seam has its centroid pulled away from that seam, into its own
    tile, and the two halves of a straddling cell are pushed to opposite sides of the
    ownership boundary between them -- so both elect their own tile and the cell is
    kept twice. That is why this scheme is for PLACED tiles and not for a grid: `cover`
    puts every cell inside a tile in one piece, so the copy that matters is not cut and
    its centroid is the cell's own. On a regular grid, where straddling cells are the
    norm, `core` is the default instead -- it is cheaper, it draws much the same
    boundary, and `resolve_stitch` insists on it under `--guide`.

    Comparing masks instead of owning by centroid fails on exactly the objects these
    models are for. Two copies of one thin diagonal cell, drawn a few pixels apart by
    two tiles, share too little to pass any threshold that does not also fuse cells
    lying against each other, so both survive and the count comes out at roughly one
    instance per tile that saw the cell. A threshold on shared pixels is a threshold on
    how thin an object is, and a promastigote with its flagellum is thin.
    """
    kept = []
    for item in found:
        if item.mask is None and item.box is None and item.line is None:
            kept.append(item)
            continue
        if _most_central(item, tiles, shape) == item.source:
            kept.append(item)
    return sorted(kept, key=lambda item: -item.score)


def _with_cells(found: list[Found], plan: list | None) -> list[Found]:
    """Each instance told which of the plan's cells it belongs to, where one does.

    The ownership schemes work this out while deciding who keeps what. Merging has no
    such step, so under `--guide` it is done afterwards -- otherwise a merged run
    carries no cell numbers at all, and `--guide-strict`, which keeps only instances
    that have one, discards the entire frame.
    """
    if not plan:
        return found
    labelled = []
    for item in found:
        matched = _planned_cell(item, plan)
        labelled.append(item if matched is None else item._replace(cell=matched + 1))
    return labelled


def resolve_stitch(how: str | None, guided: bool) -> str:
    """The scheme to use where `--stitch` did not name one.

    The tiling decides it: a grid has cores to own by, guided tiles have whole copies
    to choose between. Naming the wrong one for the tiling in use is refused rather
    than quietly producing a partition of the frame that is not one.
    """
    if how is None:
        return "whole" if guided else "core"
    if guided and how == "core":
        raise SystemExit(
            "--stitch core needs a regular grid: it gives every tile a core and those "
            "cores have to partition the frame, which guided tiles do not. Use "
            "--stitch whole, which is the default with --guide.")
    if not guided and how == "whole":
        say("note: --stitch whole owns by the tile an instance is most central in, "
            "which works on a grid too but has no reason to beat --stitch core there; "
            "it is --guide that puts a whole cell in the tile that wins")
    return how


def stitch(found: list[Found], tiles: list, shape, how: str,
           iou: float | None = None, plan: list | None = None) -> list[Found]:
    """Put the tiles' instances back together, whichever way `--stitch` asked for.

    `iou` belongs to the box merger alone: ownership compares nothing and the mask
    merger compares shapes rather than overlap, so neither has anywhere to put it. It
    is a named argument rather than `**kwargs` so that it cannot be forwarded to a
    stitcher that has no such argument.
    """
    if how == "core":
        return stitch_by_core(found, tiles, shape)
    if how == "whole":
        # The plan where there is one: settling a copy by the cell it belongs to beats
        # settling it by where the copy itself happens to sit.
        if plan:
            return stitch_by_plan(found, tiles, shape, plan)
        return stitch_by_whole(found, tiles, shape)
    if any(item.mask is not None for item in found):
        return _with_cells(merge_masks(found, tiles, shape), plan)
    if any(item.box is not None for item in found):
        return _with_cells(merge_boxes(found, tiles, shape,
                                       iou=DEFAULT_IOU if iou is None else iou), plan)
    if len(tiles) > 1 and found:
        # Chains, from a tiled pose run. There is nothing here to overlap: merging
        # compares regions, and a chain is an ordering. Saying so beats handing back
        # every cell once per tile that saw it.
        raise SystemExit("--stitch merge cannot put chains back together: they have no "
                         "area to compare and no way to splice two node orderings. "
                         "Use --stitch core, which keeps each chain whole in one tile.")
    return found


def warn_if_longer_than_overlap(found: list, tile: int, overlap: float) -> None:
    """Say so where objects are longer than the overlap is deep.

    This is the one assumption none of the merging can make up for. A cell that fits
    in no tile is never seen whole by the model, so the pieces are all there is: the
    strip-span rule can rejoin two of them, but three, or two that also disagree about
    where the cell is, cannot be recovered -- and ownership schemes double-count them
    instead. It is easily true at the defaults on crowded data, so it is checked
    against what the model actually found rather than assumed.

    Asked of the pieces the tiles produced, before stitching. Afterwards a straddling
    cell survives only as whichever fragment was kept, so the longest thing left is a
    fact about the tile spacing rather than about the cells.
    """
    if not tile:
        return
    deep = int(round(tile * overlap))
    sides = []
    for item in found:
        if item.mask is not None:
            y0, x0, y1, x1 = item.bounds
            sides.append(max(y1 - y0, x1 - x0))
        elif item.box is not None:
            x0, y0, x1, y1 = item.box
            sides.append(max(x1 - x0, y1 - y0))
        elif item.line is not None and np.isfinite(item.line[:, 0]).any():
            spread = np.nanmax(item.line, 0) - np.nanmin(item.line, 0)
            sides.append(float(max(spread)))
    if not sides or len(sides) < 5:
        return
    over = [side for side in sides if side > deep]
    if len(over) * 20 <= len(sides):  # under 5% of them, which is the tail, not a problem
        return
    longest = max(sides)
    # An overlap has to stay under the whole tile, so past a certain length no overlap
    # is deep enough and saying to raise it is advice `settings` would refuse.
    wider = longest / tile
    fix = (f"raise --overlap above {wider:.2f}" if wider < 0.89
           else f"run --tile {int(longest) + 32} or more")
    say(f"warning: {100 * len(over) / len(sides):.0f}% of what was found is longer "
        f"than the {deep} px overlap (longest {longest:.0f} px). Those cells fit in "
        f"no tile whole, so no merging afterwards can make them so -- {fix}, or "
        f"--guide to place the tiles on the cells instead.")


def add_tiling_arguments(parser, tile: int = 0) -> None:
    """The tiling flags, shared so no two scripts can spell them differently.

    Every script takes the default, 0: the whole frame in one pass, into the model at
    the input size the index records. `tile` is here for a script that wants a
    different one, and None still means "whatever the index records" to `settings`.
    """
    parser.add_argument("--tile", type=int, default=tile,
                        help="tile size in px. The default is 0: one pass over the "
                             "whole frame, into the model at the input size the index "
                             "records. Tiling is opt-in; it is what puts cells at 1.0x "
                             "on a frame larger than that input, and the run says both "
                             "the tile size the index records and the scale cells are "
                             "reaching the model at")
    parser.add_argument("--overlap", type=float, default=None,
                        help=f"tile overlap as a fraction of the tile, 0 to 0.9 "
                             f"(default {DEFAULT_OVERLAP} where the index says nothing)")
    parser.add_argument("--batch", type=int, default=4,
                        help="tiles per forward pass; lower it if the card runs out")
    add_stitch_argument(parser)


def add_stitch_argument(parser) -> None:
    """How the tiles' instances are put back together. Shared with mask2former.py."""
    parser.add_argument("--stitch", choices=("core", "whole", "merge"), default=None,
                        help="core (the default on a grid): keep each instance only "
                             "in the tile whose core holds its centroid, comparing "
                             "nothing (StarDist's scheme). It cannot fuse cells and "
                             "leaves about half as many cut, but double-counts "
                             "anything longer than the margin from a core to its tile "
                             "edge. whole (the default with --guide): keep the least "
                             "cut copy of each instance and drop the rest. merge: "
                             "join pieces across tiles that are one object")


def batches(boxes: list, size: int):
    """The tile boxes in groups of `size`, one forward pass each.

    Yields (index of the first tile in the group, the group), because what came out of
    which tile is what the merging keys on.
    """
    size = max(1, int(size))
    for start in range(0, len(boxes), size):
        yield start, boxes[start:start + size]
