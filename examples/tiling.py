#!/usr/bin/env python
"""Running a model over a frame in tiles, and putting the pieces back together.

Tiling is opt-in: by default a script runs one pass over the whole frame. `--tile N`
turns it on and puts cells at 1.0x. Overlap is a fraction of the tile and on a grid
must exceed the longest object, or no tile ever sees that object whole. `--guide`
places the tiles on the cells a detector found instead of gridding them.
"""

from __future__ import annotations

import numpy as np

from outputs import Found, say

# Used where an entry records no overlap.
DEFAULT_OVERLAP = 0.25

# Only the box merger uses this; see `stitch`.
DEFAULT_IOU = 0.5

# Cap on `cover`'s drop/recentre passes, so a pathological plan cannot spin.
_SETTLE = 4


def settings(entry: dict, tile: int | None, overlap: float | None) -> tuple[int, float]:
    """The tile size and overlap to use: the number given, else what the index says.

    tile 0 means one pass over the whole frame; tile None asks for the index's
    recorded tile. Overlap is a fraction of the tile.
    """
    recorded = (entry.get("config") or {}).get("tiling") or {}
    asked = tile is not None, overlap is not None
    if tile is None:
        tile = recorded.get("tile") or 0
    if overlap is None:
        # `is None`, not falsy: a recorded 0 means zero, a missing one means the default.
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

    `overlap` is a fraction of the tile. The last row and column snap back to the
    frame edge, so they overlap their neighbour more than the rest.
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
    """Tiles (y0, x0, y1, x1) holding every box (x0, y0, x1, y1) whole, and as few as
    possible.

    Interval stabbing per axis -- rows first, then each row's columns -- so the result
    is optimal along each axis, not over the plane. `pad` asks for that much clear
    space around each cell, given up where there is no room. Boxes are expected to lie
    inside the frame.

    -> (tiles, boxes longer than one tile, which no tile can hold whole)
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

    # A box several rows could hold goes to the first, so later rows stay emptier.
    rows = _stab([down for down, _, _ in fitted])
    mine: dict = {y: [] for y in rows}
    for down, across, _ in fitted:
        for y in rows:
            if down[0] <= y <= down[1]:
                mine[y].append(across)
                break
    tiles = [_tile_at(y, x, shape, tile) for y in rows for x in _stab(mine[y])]

    held = [box for _, _, box in fitted]
    # Recentring changes which boxes each tile holds, so drop and recentre alternate
    # until neither changes anything, ending on a recentre.
    for _ in range(_SETTLE):
        settled = _recentre(_drop_redundant(tiles, held), held, shape, tile, pad)
        if settled == tiles:
            break
        tiles = settled
    # A cell longer than the tile gets a tile of its own, centred on it.
    tiles += [_tile_at(*_middle_of(box, shape, tile), shape, tile) for box in oversize]
    return list(dict.fromkeys(tiles)), oversize


def assign(tiles: list, boxes: list) -> list:
    """Each box paired with the tile that owns it: the first one holding it whole.

    Made once from the boxes, so `stitch_by_plan` settles every copy of a cell by the
    cell rather than by the copy. An oversize box goes to the tile holding most of it.
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
    """The fewest points putting one inside every (lo, hi). Optimal.

    Interval stabbing: take the range ending soonest, put a point at its end, repeat.
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
    """Drop tiles whose every box some other tile already holds whole.

    Per-axis minimality does not make the rectangle minimal. Tried least-useful first,
    so dropping one cannot make a more useful one look redundant.
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

    Stabbing lands a tile edge exactly on a box edge, which a model reads as a cut
    cell. Every origin in the range holds the same boxes. Tiles holding nothing go.
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

    `pad` is clear space around each box, so the ranges are intersected per box, not
    taken around their hull. Where the full `pad` is impossible, the widest margin
    that works for all of them is found by bisection.
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
    # "detected cells" is what the detector found, not what is there.
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

    Judged only inside the strip the two tiles share, the one ground both models saw.
    `agreement` is symmetric: shared pixels over what either has in the strip.
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
    # box=None: one member's box would disagree with the union mask, and `_extent`
    # then falls back to the mask's own bounds.
    return best._replace(mask=mask, origin=(y0, x0), box=None)


def touches_seam(extent, tile, shape, margin: int = 2) -> bool:
    """Whether something runs into a tile edge that is not also the frame edge.

    How a cut object is told from a whole one. `extent` is (x0, y0, x1, y1); `tile`
    is (y0, x0, y1, x1).
    """
    x0, y0, x1, y1 = extent
    ty0, tx0, ty1, tx1 = tile
    height, width = shape[:2]
    return ((tx0 > 0 and x0 <= tx0 + margin)
            or (ty0 > 0 and y0 <= ty0 + margin)
            or (tx1 < width and x1 >= tx1 - margin)
            or (ty1 < height and y1 >= ty1 - margin))


def _extent(item: Found):
    """(x0, y0, x1, y1) around an instance's geometry, or None where it has none."""
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


def merge_masks(found: list[Found], tiles: list,
                agreement: float = 0.5) -> list[Found]:
    """Union same-class masks from different tiles that are really one object.

    Two masks are one object if they agree about the strip their tiles share; see
    `_agree_in_band`. Joins are transitive, so a loose rule fuses a crowded frame into
    one blob. Only instances from different tiles are compared, so cells lying against
    each other inside one tile stay two.
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

    One copy of each cell wins and the rest go. A box clipped at a seam has only about
    half the IoU of the whole one, so seam-touching boxes are judged on `containment`
    instead, or the cell is kept twice. Taken largest first, so the fullest copy
    survives. Only boxes from different tiles are compared -- ultralytics has already
    run NMS within each tile.
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

    `kept` is never the smaller of the two.
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

    Ultralytics fits the longest side of a crop to `imgsz`, so the scale is the ratio
    of the two. Measured off the crops in `boxes`, since a guided run can have one
    placed tile and the count alone cannot tell that from an untiled pass.
    """
    longest = max(max(y1 - y0, x1 - x0) for y0, x0, y1, x1 in boxes)
    return f"imgsz {imgsz}: cells at {imgsz / longest:.2f}x native"


def recorded_tile(entry: dict) -> int:
    """The tile size the index records for a model, or 0 where it records none."""
    return int(((entry.get("config") or {}).get("tiling") or {}).get("tile") or 0)


def recorded_imgsz(entry: dict, shape) -> int:
    """The input size the index records for a model, used by an untiled pass.

    Falls back to the frame's own long side where the entry records nothing.
    """
    recorded = int(((entry.get("config") or {}).get("detection") or {}).get("imgsz") or 0)
    return recorded or native_imgsz(shape, 0)


def imgsz_for(entry: dict, shape, tile: int, override: int = 0) -> int:
    """The inference size for one frame: the same rule for every ultralytics script.

    Tiled, the tile itself, so a crop reaches the model at 1.0x; untiled, the input
    size the index records. `override` is --imgsz.
    """
    return int(override) or (native_imgsz(shape, tile) if tile
                             else recorded_imgsz(entry, shape))


def tiling_hint(entry: dict, tile: int) -> str | None:
    """Where a run is not tiling, what the index says it could tile at.

    Said once, on the first frame.
    """
    recorded = recorded_tile(entry)
    if tile or not recorded:
        return None
    return (f"not tiling; --tile {recorded} is what the index records, and puts cells "
            f"at 1.0x (--guide places those tiles on the cells instead of gridding)")


def native_imgsz(shape, tile: int) -> int:
    """The imgsz that puts a crop in front of the model at 1.0x.

    The tile size, or the frame's long side where the frame is smaller than a tile.
    Rounded up to the multiple of 32 ultralytics requires.
    """
    edge = min(int(tile), max(shape[:2])) if tile else max(shape[:2])
    return int(-(-int(edge) // 32) * 32)


def _core_axis(starts: list, ends: list, total: int) -> dict:
    """Each (start, end) tile edge pair -> the half-open span it owns along one axis.

    The spans partition the axis: neighbours meet in the middle of the strip they
    share. Keyed on both edges, so tiles starting together but ending apart cannot
    collapse into one entry and be handed the same span.
    """
    order = sorted(set(zip(starts, ends)))
    spans = {}
    for index, (start, end) in enumerate(order):
        low = 0 if index == 0 else (start + order[index - 1][1]) // 2
        high = total if index == len(order) - 1 else (end + order[index + 1][0]) // 2
        spans[start, end] = (low, high)
    return spans


def _centroid(item: Found):
    """(y, x) where an instance sits, from its mask, box or chain.

    None where it has nowhere to be: a chain none of whose nodes the model saw.
    """
    if item.mask is not None:
        rows, cols = np.nonzero(item.mask)
        return item.origin[0] + rows.mean(), item.origin[1] + cols.mean()
    if item.box is not None:
        x0, y0, x1, y1 = item.box
        return (y0 + y1) / 2.0, (x0 + x1) / 2.0
    # A chain: the mean of the visible nodes. An invisible node is carried as NaN.
    if item.line is not None and np.isfinite(item.line[:, 0]).any():
        return (float(np.nanmean(item.line[:, 1])), float(np.nanmean(item.line[:, 0])))
    return None


def stitch_by_core(found: list[Found], tiles: list, shape) -> list[Found]:
    """Keep each instance in the one tile that owns the ground it stands on.

    StarDist's `predict_instances_big` scheme: each tile owns a core, the cores
    partition the frame, and an instance is kept only by the tile whose core holds its
    centroid. Nothing is compared, so nothing fuses. It needs the margin from core to
    tile edge to exceed the object's reach; past that an object is double-counted.
    See `warn_if_longer_than_overlap`.
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

    Ownership settled by the detector's box rather than by the copy's own centroid, so
    the two halves of a straddling cell agree on one owner instead of both electing
    themselves. One copy survives per cell and class; where the owner tile's own pass
    produced nothing, the least cut copy from another tile stands in. An instance no
    box explains falls back to the tile it sits most centrally in. The matched cell is
    kept on each instance, so body and flagellum share one number; see
    `outputs.numbers_for`.
    """
    kept = []
    claimed: set = set()   # (cell, class) pairs the owner tile's own pass produced
    unclaimed: dict = {}   # ... and the other tiles' copies, kept until that is known
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
        # A cell longer than a tile has no owner holding it whole, so its pieces are
        # settled by where they sit rather than dropped for disagreeing.
        if not _holds(tiles[owner], box):
            if _most_central(item, tiles, shape) == item.source:
                kept.append(item._replace(cell=matched + 1))
        elif item.source == owner:
            claimed.add((matched, item.label))
            kept.append(item._replace(cell=matched + 1))
        else:
            unclaimed.setdefault((matched, item.label), []).append(item)
    # Where the owner produced nothing of this class for this cell, the least cut copy
    # stands in rather than the cell being lost; ties go to the better score.
    for key, copies in unclaimed.items():
        if key in claimed:
            continue
        best = max(copies, key=lambda copy: (_room(copy, tiles, shape), copy.score))
        kept.append(best._replace(cell=key[0] + 1))
    return sorted(kept, key=lambda item: -item.score)


def _room(item: Found, tiles: list, shape) -> float:
    """How far an instance sits from its own tile's seams. Higher is less cut."""
    centre = _centroid(item)
    if centre is None:
        return float("-inf")
    return _clearance(centre[0], centre[1], tiles[item.source], shape)


def _planned_cell(item: Found, plan: list):
    """Which of the plan's cells this instance is part of, as an index, or None.

    Scored on how much of the instance lies inside the box; the centroid only breaks
    ties, since a thin diagonal instance can have its centroid over a neighbour.
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

    A tile edge on the frame edge is not a seam; counting it would tie every border
    tile and hand ownership to the lowest index, which holds the most truncated copy.
    inf where the tile has no seams at all.
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

    Ownership by centroid, for placed tiles, where cores do not partition the frame.
    Of the tiles holding a centroid, the one leaving it most room to its own edges
    wins; ties go to the earlier tile. It does not guarantee that two copies of one
    cell agree on their centroid, so a cell straddling a seam is kept twice -- which
    is why `core` stays the default on a grid.
    """
    # `stitch_by_plan` with nothing planned is exactly this decision, once per copy.
    return stitch_by_plan(found, tiles, shape, [])


def _with_cells(found: list[Found], plan: list | None) -> list[Found]:
    """Each instance told which of the plan's cells it belongs to, where one does.

    The last step of every `stitch` branch, so no scheme comes back without the cell
    numbers `--guide-strict` filters on. An instance already carrying one keeps it.
    """
    if not plan:
        return found
    labelled = []
    for item in found:
        if item.cell:
            labelled.append(item)
            continue
        matched = _planned_cell(item, plan)
        labelled.append(item if matched is None else item._replace(cell=matched + 1))
    return labelled


def resolve_stitch(how: str | None, guided: bool) -> str:
    """The scheme to use where `--stitch` did not name one: core on a grid, whole
    under --guide. Naming core for guided tiles is refused, since their cores do not
    partition the frame.
    """
    if how is None:
        return "whole" if guided else "core"
    if guided and how == "core":
        raise SystemExit(
            "--stitch core needs a regular grid: guided tiles have no cores that "
            "partition the frame. Use --stitch whole, the default with --guide.")
    if not guided and how == "whole":
        say("note: --stitch whole works on a grid but has no reason to beat "
            "--stitch core there")
    return how


def stitch(found: list[Found], tiles: list, shape, how: str,
           iou: float | None = None, plan: list | None = None) -> list[Found]:
    """Put the tiles' instances back together, whichever way `--stitch` asked for.

    `iou` is used by the box merger alone.
    """
    if how == "core":
        kept = stitch_by_core(found, tiles, shape)
    elif how == "whole":
        # Settle by the cell where there is a plan, else by where the copy sits.
        kept = (stitch_by_plan(found, tiles, shape, plan) if plan
                else stitch_by_whole(found, tiles, shape))
    elif any(item.mask is not None for item in found):
        kept = merge_masks(found, tiles)
    elif any(item.box is not None for item in found):
        kept = merge_boxes(found, tiles, shape,
                           iou=DEFAULT_IOU if iou is None else iou)
    elif len(tiles) > 1 and found:
        # Chains, from a tiled pose run: no area to compare, no orderings to splice.
        raise SystemExit("--stitch merge cannot put chains back together: they have no "
                         "area to compare. Use --stitch core, which keeps each chain "
                         "whole in one tile.")
    else:
        kept = found
    # Cell numbers come from the plan, not the scheme, so every branch leaves here.
    return _with_cells(kept, plan)


def warn_if_longer_than_overlap(found: list, tile: int, overlap: float) -> None:
    """Warn where objects are longer than the overlap is deep.

    A cell longer than the overlap is whole in no tile, so merging cannot recover it
    and ownership schemes double-count it. Must be called on the tiles' own pieces,
    BEFORE stitching; afterwards the lengths describe the tile spacing, not the cells.
    Silent under 5 pieces or where under 5% are over.
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
    if len(over) * 20 <= len(sides):  # under 5%: the tail, not a problem
        return
    longest = max(sides)
    # Overlap must stay under 0.9, so past a certain length only a bigger tile helps.
    wider = longest / tile
    fix = (f"raise --overlap above {wider:.2f}" if wider < 0.89
           else f"run --tile {int(longest) + 32} or more")
    say(f"warning: {100 * len(over) / len(sides):.0f}% of what was found is longer "
        f"than the {deep} px overlap (longest {longest:.0f} px). Those cells fit in "
        f"no tile whole -- {fix}, or --guide to place tiles on the cells.")


def add_tiling_arguments(parser, tile: int = 0) -> None:
    """The tiling flags, shared so no two scripts can spell them differently.

    `tile` is the default for --tile; None means the index's recorded tile.
    """
    group = parser.add_argument_group("tiling")
    group.add_argument("--tile", type=int, default=tile,
                       help="tile size in px; 0 (default) is one pass over the whole "
                            "frame. Tiling puts cells at 1.0x")
    group.add_argument("--overlap", type=float, default=None,
                       help=f"tile overlap as a fraction of the tile, 0 to 0.9 "
                            f"(default {DEFAULT_OVERLAP} where the index says nothing)")
    group.add_argument("--batch", type=int, default=4,
                       help="tiles per forward pass; lower it if the card runs out")
    add_stitch_argument(group)


def add_stitch_argument(parser) -> None:
    """How the tiles' instances are put back together.

    `parser` is the tiling argument group where there is one, so --stitch sits with it.
    """
    parser.add_argument("--stitch", choices=("core", "whole", "merge"), default=None,
                        help="core (default on a grid): keep each instance only in "
                             "the tile whose core holds its centroid. whole (default "
                             "with --guide): keep the least cut copy of each. merge: "
                             "join pieces across tiles that are one object")


def batches(boxes: list, size: int):
    """The tile boxes in groups of `size`, one forward pass each.

    Yields (index of the first tile in the group, the group); stitching keys on that
    index.
    """
    size = max(1, int(size))
    for start in range(0, len(boxes), size):
        yield start, boxes[start:start + size]
