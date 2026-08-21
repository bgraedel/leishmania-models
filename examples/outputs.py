#!/usr/bin/env python
"""What a model found, and what to do with it: colour it, stack it, hand it to ImageJ.

Each script turns its own results into a list of `Found` per frame and leaves the
rest here, so all four write the same files in the same layout and nothing in this
module has to know which model produced them.

    --frames 0-99        run a range rather than one frame; every output grows a T axis
    --color instance     a hue per instance instead of a colour per class
    --tiff labels.tif    an ImageJ hyperstack, one channel per class, pixel values the
                         instance's number in that frame
    --rois rois.zip      a RoiSet, one roi per instance, that ImageJ opens into the
                         ROI manager

The stacks are memory-mapped and filled a frame at a time, so a long run costs disk
rather than RAM.
"""

from __future__ import annotations

import colorsys
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np


class Found(NamedTuple):
    """One thing a model found in one frame.

    Each model fills in what it has: a mask from the segmenters, a box from the
    detector, an ordered chain of points from the pose model.

    A mask is stored as its own bounding box rather than a frame-sized array, with
    `origin` saying where that box sits. A tiled 2048 px frame can hold a thousand
    instances, and a thousand frame-sized bool arrays is four gigabytes for objects
    that are a few hundred pixels each.

    `cell` is which of the detector's cells this is part of, and is filled in only by a
    `--guide` run, which is the only one that knows. It is what makes an animal, its
    body and its flagellum three views of ONE thing rather than three unrelated
    instances that happen to overlap: they carry the same number into every output, so
    the three channels of a label stack can be read back together.
    """

    label: str
    score: float
    mask: np.ndarray | None = None  # bool, the size of its own bounding box
    origin: tuple[int, int] = (0, 0)  # (y, x) of mask[0, 0] in the frame
    box: tuple[float, float, float, float] | None = None
    line: np.ndarray | None = None  # (n, 2), NaN where a point is not visible
    edges: list | None = None  # pairs of indices into `line`; consecutive if absent
    source: int = 0  # which tile it came out of; see tiling.merge_masks
    cell: int | None = None  # which detected cell it is part of, under --guide

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """(y0, x0, y1, x1) of the mask, exclusive at the far edge."""
        y, x = self.origin
        height, width = self.mask.shape
        return y, x, y + height, x + width


def at_bounds(mask: np.ndarray, origin: tuple[int, int] = (0, 0)):
    """A `Found`-ready (mask, origin) pair: the mask cut down to its own bounding box.

    `origin` is where the given mask sits, so a tile's mask can be handed in with the
    tile's own corner and comes back in frame coordinates.
    """
    rows = np.flatnonzero(mask.any(1))
    if not len(rows):
        return None
    cols = np.flatnonzero(mask.any(0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    return mask[y0:y1, x0:x1], (origin[0] + y0, origin[1] + x0)


def clean_components(mask: np.ndarray, mode: str = "union", min_area: int = 60,
                     min_frac: float = 0.05) -> np.ndarray:
    """Drop spurious connected components from ONE instance's mask.

    Mask2Former will happily predict a good blob and a few specks elsewhere in the
    tile under a single instance id. The specks are not another cell and nothing
    downstream will ever split them off, so they go here, before the instance is
    placed in the frame.

      "all"      leave the mask as the model drew it
      "largest"  keep only the biggest 8-connected component
      "union"    drop components below max(min_area, min_frac x the instance) and keep
                 the union of the rest; where every component is tiny, keep the largest

    Only components *within one instance* are touched. Two instances that happen to
    touch are two instances; that question belongs to tiling.merge_masks.
    """
    if mode == "all":
        return mask
    labelled, count = _components(mask)
    if count <= 1:
        return mask
    areas = np.bincount(labelled.ravel(), minlength=count + 1)[1:]
    if mode == "largest":
        keep = np.array([int(np.argmax(areas)) + 1])
    else:
        floor = max(int(min_area), int(min_frac * int(areas.sum())))
        keep = np.flatnonzero(areas >= floor) + 1
        if not len(keep):  # all of them are specks; the biggest speck is the instance
            keep = np.array([int(np.argmax(areas)) + 1])
        elif len(keep) == count:
            return mask  # nothing dropped, so the union IS the input
    lookup = np.zeros(count + 1, bool)
    lookup[keep] = True
    return lookup[labelled]


def _components(mask: np.ndarray):
    """(labels, count) at 8-connectivity, from scipy or OpenCV, whichever is installed."""
    try:
        from scipy.ndimage import label
    except ImportError:
        pass
    else:
        return label(mask, structure=np.ones((3, 3), int))
    try:
        import cv2
    except ImportError:
        raise SystemExit(
            "cleaning up components needs scipy or OpenCV: pip install scipy, or pass"
            " --components all to keep the masks as the model drew them")
    count, labelled = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return labelled, count - 1


def instance_mask(mask: np.ndarray, origin: tuple[int, int], mode: str = "union",
                  min_area: int = 60, min_frac: float = 0.05, drop_below: int = 25):
    """A raw mask out of a tile -> the (mask, origin) a `Found` wants, or None.

    Cut to its bounding box first, which is also much the cheapest thing to label;
    cleaned of stray components; cut again, because dropping a component at the edge
    frees a row; and dropped altogether if what survives is too small to be a cell.
    """
    placed = at_bounds(mask, origin)
    if placed is None:
        return None
    patch, where = placed
    cleaned = clean_components(patch, mode, min_area, min_frac)
    if cleaned is not patch:
        placed = at_bounds(cleaned, where)
        if placed is None:
            return None
        patch, where = placed
    if int(patch.sum()) < drop_below:
        return None
    return patch, where


def resolve_classes(names, override: str | None = None) -> list[str]:
    """The model's class names in label-id order, with `--classes` able to replace them.

    A checkpoint whose id2label was never filled in comes back as LABEL_0, LABEL_1,
    and those names then go on to title the channels of the label stack and every roi
    in the RoiSet. `--classes body,flagellum` replaces them by position.
    """
    ordered = [str(names[key]) for key in sorted(names)]
    if override:
        given = [part.strip() for part in override.split(",") if part.strip()]
        if len(given) != len(ordered):
            raise SystemExit(f"--classes: the model has {len(ordered)} "
                             f"({', '.join(ordered)}), got {len(given)}")
        return given
    if all(re.fullmatch(r"label[_-]?\d+", name, re.IGNORECASE) for name in ordered):
        say(f"warning: this checkpoint calls its classes {', '.join(ordered)}. "
              f"--classes body,flagellum would say what they are.")
    return ordered


CLASS_COLOURS = {
    "animal": (245, 245, 245),
    "cell": (255, 210, 0),
    "body": (255, 90, 90),
    "flagellum": (90, 200, 255),
}
UNNAMED = (255, 210, 0)


def colour(index: int, label: str, by: str) -> tuple[int, int, int]:
    if by == "class":
        return CLASS_COLOURS.get(label, UNNAMED)
    # A golden-angle walk round the hue circle: consecutive instances land far apart
    # on it, so two cells that touch never come out the same colour, which is the
    # whole point of colouring by instance. Under --guide the number is the detector's
    # cell, so a body and its own flagellum come out the SAME hue and the pairing is
    # visible. Nothing tracks between frames, so a hue still changes frame to frame.
    hue = (index * 0.6180339887498949) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return round(255 * red), round(255 * green), round(255 * blue)


# Frames


def frame_count(path) -> int:
    """How many frames the input offers: the pages of a stack, or a folder's images."""
    from fetch import frames_in

    return len(frames_in(path))


def default_output(image, model: str) -> Path:
    """Where the overlay goes when --out is not given.

    A folder has no suffix to replace, so its results are named after it and land
    beside it rather than inside it, where a second run would pick them up as input.
    """
    image = Path(image)
    if image.is_dir():
        return image.parent / f"{image.name}.{model}.png"
    return image.with_suffix(f".{model}.png")


def progress(items, desc: str, leave: bool = True):
    """A bar over `items` where one is worth having, and `items` itself where not.

    tqdm arrives with ultralytics and with transformers, so every script that could
    want this already has it; it is declared as a dependency anyway rather than
    relied on second-hand, and its absence falls back to no bar rather than failing.
    """
    items = list(items)
    if len(items) < 2:
        return items
    try:
        from tqdm import tqdm
    except ImportError:
        return items
    # disable=None is tqdm's "only when attached to a terminal". Piped into a file or
    # a log, a bar is redrawn on every update and the record fills with carriage
    # returns, so there it prints nothing and the per-frame lines carry the progress.
    return tqdm(items, desc=desc, leave=leave, unit="", dynamic_ncols=True,
                disable=None)


def say(text: str) -> None:
    """print, but without tearing a progress bar that is running underneath it."""
    try:
        from tqdm import tqdm
        tqdm.write(text)
    except ImportError:
        print(text)


def parse_frames(spec: str, available: int) -> list[int]:
    """`0`, `10-19`, `10-`, `0-99:5` or `all` -> the frame numbers to run.

    An open end (`10-`) runs to the last frame. A range running off the end is cut
    to what the input holds and says so, rather than quietly returning fewer frames
    than were asked for.
    """
    spec = str(spec).strip()
    asked = None
    if spec == "all":
        first, last, step = 0, available - 1, 1
    else:
        span, _, tail = spec.partition(":")
        first_text, dash, last_text = span.partition("-")
        try:
            first = int(first_text)
            step = int(tail or 1)
            last = int(last_text) if last_text else (available - 1 if dash else first)
        except ValueError:
            raise SystemExit(f"--frames {spec}: expected 0, 10-19, 10-, 0-99:5 or all")
        asked = last
    chosen = list(range(first, min(last, available - 1) + 1, max(1, step)))
    if not chosen:
        raise SystemExit(f"--frames {spec}: nothing in that range; the file has {available}")
    if asked is not None and asked > available - 1:
        say(f"--frames {spec}: this input has {available} frame(s), so the run stops at "
            f"{chosen[-1]}")
    return chosen


# The overlay


def segments(item: Found):
    """A chain's edges, minus any whose endpoint the model did not see.

    The pairs come from the model's own entry in the index. Consecutive nodes are only
    the default, which happens to be right for a flagellum.
    """
    pairs = item.edges or [(i, i + 1) for i in range(len(item.line) - 1)]
    for start, end in pairs:
        first, last = item.line[start], item.line[end]
        if not (np.isnan(first).any() or np.isnan(last).any()):
            yield first, last


def draw(frame: np.ndarray, found: list[Found], by: str = "class",
         alpha: float = 0.45) -> np.ndarray:
    """The frame with everything found painted onto it."""
    from PIL import Image, ImageDraw

    painted = frame.copy()
    given = numbers_for(found)
    for index, item in zip(given, found):
        if item.mask is None:
            continue
        rgb = np.array(colour(index, item.label, by), np.float32)
        y0, x0, y1, x1 = item.bounds
        window = painted[y0:y1, x0:x1]
        window[item.mask] = (alpha * rgb + (1 - alpha) * window[item.mask]).astype(np.uint8)

    canvas = Image.fromarray(painted)
    pen = ImageDraw.Draw(canvas)
    for index, item in zip(given, found):
        rgb = colour(index, item.label, by)
        if item.box is not None:
            pen.rectangle([float(value) for value in item.box], outline=rgb, width=2)
        if item.line is not None:
            for first, last in segments(item):
                pen.line([tuple(first), tuple(last)], fill=rgb, width=2)
            for position, (x, y) in enumerate(item.line):
                if np.isnan(x):
                    continue
                # The first node is picked out whatever the rest is coloured, so which
                # end of a chain is the head stays readable.
                pen.ellipse([x - 2, y - 2, x + 2, y + 2],
                            fill=(255, 60, 60) if position == 0 else rgb)
    return np.asarray(canvas)


# The stacks


def numbers_for(found: list[Found]) -> list:
    """The number each instance goes into the outputs under.

    The detector's cell where a `--guide` run established one, so an animal, its body
    and its flagellum all carry the same number in their own channels and whoever reads
    the stack can put the three back together. Without a guide there is no such
    relation to record and the number is the position in the list.

    A number is taken by at most one instance of a class. The roi's name and the label
    plane's pixel value are both built from it, so two instances of one class sharing a
    number is one outline silently replacing the other. A cell the segmenter found two
    of the same class in -- and anything the detector did not propose at all -- is
    numbered after the last cell instead.
    """
    cells = [item.cell for item in found if item.cell]
    if not cells:
        return list(range(1, len(found) + 1))
    spare = max(cells) + 1
    taken: set = set()
    given = []
    for item in found:
        number = item.cell if item.cell and (item.label, item.cell) not in taken else 0
        if not number:
            number, spare = spare, spare + 1
        taken.add((item.label, number))
        given.append(number)
    return given


def _memmap(path, shape, dtype, axes, labels=None):
    import tifffile

    metadata = {"axes": axes}
    if labels:
        metadata["Labels"] = labels
    return tifffile.memmap(str(path), shape=shape, dtype=dtype, imagej=True,
                           metadata=metadata)


def _chain_patch(item: Found, shape: tuple[int, int]):
    """A chain drawn as a one-pixel line, as (bool patch, its (y, x) corner).

    Drawn into its own bounding box rather than straight onto the plane, so what it
    would claim can be compared with what is already there before anything is written.
    """
    cv2 = opencv()
    lines = list(segments(item))
    if not lines:
        return None
    points = np.array([point for pair in lines for point in pair], float)
    x0 = max(0, int(np.floor(points[:, 0].min())) - 1)
    y0 = max(0, int(np.floor(points[:, 1].min())) - 1)
    x1 = min(shape[1], int(np.ceil(points[:, 0].max())) + 2)
    y1 = min(shape[0], int(np.ceil(points[:, 1].max())) + 2)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for first, last in lines:
        # plain ints: older OpenCV refuses a numpy scalar in a point. A node outside
        # the patch is clipped by cv2, which is what it did on the plane before.
        cv2.line(patch, (int(round(first[0])) - x0, int(round(first[1])) - y0),
                 (int(round(last[0])) - x0, int(round(last[1])) - y0), 1, 1)
    return patch.astype(bool), (y0, x0)


class LabelStack:
    """Label images, one channel per class, one time point per frame.

    Pixel values are the instance's number within its frame, so a channel is a label
    image rather than a binary mask and Analyze Particles or any label reader takes it
    unchanged. A box is filled in and a chain is drawn as a one-pixel line, so the
    detector and the pose model land in the same file format as the segmenters.

    The number runs across the whole frame rather than restarting per channel, so a
    channel skips values -- and the value in the stack is the one in the roi's name,
    which is what makes the two files line up. Under `--guide` it is the DETECTOR's
    cell number, so a body in channel 2 and a flagellum in channel 3 sharing the value
    7 are the body and the flagellum of one cell. See `numbers_for`.

    One plane holds one number per pixel, so two instances of the same class that
    overlap cannot both have it. They are painted best-scoring first and a pixel is
    claimed only once, so the better instance keeps its shape whole and the other is
    the one that loses pixels; what that cost was is said at the end rather than left
    to be found later in the measurements. A detector's boxes are filled and overlap
    constantly, and an amodal model draws its cells through each other on purpose, so
    both will always pay it: `--rois` is where every instance keeps its own outline
    and nothing is contested.
    """

    def __init__(self, path, classes: list[str], shape: tuple[int, int], frames: int):
        self.classes = list(classes)
        self.channel = {name: index for index, name in enumerate(self.classes)}
        self.shape = tuple(shape)
        self.contested = 0  # pixels a lower-scoring instance could not have
        self.buried = 0  # instances left with none of their own at all
        self.stack = _memmap(path, (frames, len(self.classes), *shape), np.uint16,
                             "TCYX", labels=self.classes * frames)

    def add(self, position: int, found: list[Found]) -> None:
        # Best-scoring first, whatever order the list arrived in. The instance numbers
        # have to stay the ones the rois and the overlay use, so it is the painting
        # that is reordered here and not the list.
        given = numbers_for(found)
        for order in sorted(range(len(found)), key=lambda i: -found[i].score):
            item = found[order]
            if item.label not in self.channel:
                continue
            claim = self._claim(item, self.stack[position, self.channel[item.label]])
            if claim is None:
                continue
            window, wanted = claim
            free = wanted & (window == 0)
            window[free] = given[order]
            self.contested += int(wanted.sum()) - int(free.sum())
            if not free.any():
                self.buried += 1

    def _claim(self, item: Found, plane):
        """(the window of the plane this instance falls in, the pixels it wants)."""
        if item.mask is not None:
            y0, x0, y1, x1 = item.bounds
            return plane[y0:y1, x0:x1], item.mask
        if item.box is not None:
            x0, y0, x1, y1 = (int(round(value)) for value in item.box)
            # Both ends clamped: a negative far edge becomes a negative slice bound,
            # which is not an empty window but very nearly the whole plane.
            window = plane[max(y0, 0):max(y1 + 1, 0), max(x0, 0):max(x1 + 1, 0)]
            return (window, np.ones(window.shape, bool)) if window.size else None
        if item.line is not None:
            drawn = _chain_patch(item, self.shape)
            if drawn is None:
                return None
            patch, (y0, x0) = drawn
            return plane[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]], patch
        return None

    def close(self) -> None:
        self.stack.flush()

    def report(self) -> None:
        if not self.contested:
            return
        say(f"  {self.contested} px went to whichever instance scored higher"
            + (f", and {self.buried} instance(s) kept none of their own" if self.buried
               else "")
            + ": one plane holds one number per pixel, so same-class instances that "
              "overlap cannot both have it. --rois keeps every outline whole.")


# ImageJ rois


def opencv():
    try:
        import cv2
    except ImportError:
        raise SystemExit("this step needs OpenCV: pip install opencv-python-headless")
    return cv2


def _outline(item: Found) -> np.ndarray | None:
    """The largest outline of an instance's mask, as (n, 2) x-y points in the frame.

    The largest only. A mask that comes back in two pieces would need two rois and a
    naming scheme to say they belong together, which is not this file's decision.
    """
    cv2 = opencv()
    contours, _ = cv2.findContours(item.mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if len(biggest) < 3:
        return None
    y0, x0 = item.origin
    return biggest.reshape(-1, 2).astype(np.float32) + (x0, y0)


def _polyline(item: Found):
    """The longest unbroken run of a chain, as (n, 2) x-y points, or None.

    A polyline is ONE ordered path, so a chain the model saw in pieces cannot become a
    single roi without inventing the joins that are missing. The longest run only, for
    the same reason `_outline` keeps the largest contour: two rois would need a naming
    scheme saying they belong together, and that is not this file's decision.

    Built from `segments`, so it honours the model's own skeleton and skips an edge
    whose endpoint was not seen -- which the label stack and the overlay already do.
    Reading the nodes in array order instead drew a line straight across the gap, and
    for a skeleton that is not consecutive it drew the wrong shape entirely.
    """
    runs, run = [], []
    for first, last in segments(item):
        if run and np.array_equal(run[-1], first):
            run.append(last)
        else:
            if len(run) > 1:
                runs.append(run)
            run = [first, last]
    if len(run) > 1:
        runs.append(run)
    return np.array(max(runs, key=len), float) if runs else None


def rois(found: list[Found], position: int, number: int | None = None,
         by: str = "class") -> list:
    """One roi per instance, named <frame>-<class>-<instance>.

    `position` is the slice within THIS run's output, and it is what the roi is
    pinned to, because that stack is what the RoiSet gets opened over. `number` is
    the frame it came from and goes in the name only: `--frames 10-19` writes ten
    slices and its rois belong to slices 0-9 of them, however the input numbered them.

    The class is carried in the name and the outline colour rather than in the roi's
    channel: a RoiSet pinned to channel 2 disappears when it is opened over the
    original single-channel movie, which is where most of these end up.
    """
    import roifile

    if number is None:
        number = position
    made = []
    for index, item in zip(numbers_for(found), found):
        if item.mask is not None:
            points, kind = _outline(item), roifile.ROI_TYPE.POLYGON
        elif item.line is not None:
            points, kind = _polyline(item), roifile.ROI_TYPE.POLYLINE
        elif item.box is not None:
            x1, y1, x2, y2 = item.box
            # Four corners rather than ROI_TYPE.RECT: it draws the same, and one code
            # path means a box can never disagree with its own bounding box.
            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            kind = roifile.ROI_TYPE.POLYGON
        else:
            continue
        if points is None:
            continue
        roi = roifile.ImagejRoi.frompoints(
            points, name=f"{number:04d}-{item.label}-{index:03d}",
            t=position, position=position)
        roi.roitype = kind
        roi.stroke_color = bytes((255, *colour(index, item.label, by)))
        made.append(roi)
    return made


def write_rois(path, made: list) -> None:
    import roifile

    if not made:
        say("nothing to put in a RoiSet")
        return
    roifile.roiwrite(str(path), made, mode="w")
    say(f"wrote {path}  ({len(made)} rois)")


def _stack_path(out: Path, frames: int) -> Path:
    """One frame goes to a .png; several have to go to a stack."""
    if frames > 1 and out.suffix.lower() not in (".tif", ".tiff"):
        return out.with_suffix(".tif")
    return out


# What the scripts call


def add_output_arguments(parser) -> None:
    """The flags every script shares, so none of them can drift from the others."""
    parser.add_argument("--frames", default="0",
                        help="which frames: 0, 10-19, 0-99:5, or all")
    parser.add_argument("--color", "--colour", choices=("class", "instance"),
                        default="class",
                        help="a colour per class, or a hue per instance")
    parser.add_argument("--out", type=Path, default=None,
                        help="the overlay: a .png for one frame, an ImageJ RGB stack "
                             "for a range")
    parser.add_argument("--tiff", type=Path, default=None,
                        help="label images as an ImageJ hyperstack, one channel per class")
    parser.add_argument("--rois", type=Path, default=None,
                        help="a RoiSet .zip, one roi per instance, named "
                             "<frame>-<class>-<n>")
    parser.add_argument("--classes", default=None,
                        help="rename the model's classes, in label-id order, e.g. "
                             "body,flagellum. For a checkpoint whose id2label was "
                             "never filled in")


def add_mask_arguments(parser) -> None:
    """The per-instance mask cleanup, shared by the two scripts that produce masks."""
    parser.add_argument("--components", choices=("union", "largest", "all"),
                        default="union",
                        help="stray connected components inside one instance mask: "
                             "drop the small ones, keep only the largest, or keep all")
    parser.add_argument("--min-component-area", type=int, default=60,
                        help="union: a component below this many px is a speck")
    parser.add_argument("--min-component-frac", type=float, default=0.05,
                        help="union: ... and so is one below this fraction of the instance")
    parser.add_argument("--min-area", type=int, default=25,
                        help="drop an instance smaller than this many px altogether")


def report(number: int, found: list[Found], total: int, dropped: int = 0) -> None:
    """One line per frame over a range, the whole list for a single frame.

    `dropped` is what `--guide-strict` discarded. Counted out loud rather than left
    out: the whole point of the flag is that the detector decides what is in the
    frame, and how much the segmenter disagreed is the thing worth watching.
    """
    from collections import Counter

    counts = Counter(item.label for item in found)
    summary = ", ".join(f"{name} x{count}" for name, count in counts.items()) or "nothing"
    if any(item.cell for item in found):
        cells = len({item.cell for item in found if item.cell})
        loose = sum(1 for item in found if not item.cell)
        summary += (f" over {cells} detected cell(s)"
                    + (f", {loose} matching none of them" if loose else ""))
    if dropped:
        summary += f"; {dropped} discarded as matching no detected cell"
    if total > 1:
        say(f"  frame {number}: {len(found)} instances -- {summary}")
        return
    say(f"{len(found)} instances: {summary}")
    for index, item in enumerate(found[:10], 1):
        if item.mask is not None:
            y0, x0 = item.origin
            where = f"{int(item.mask.sum())} px at ({x0}, {y0})"
        elif item.box is not None:
            x1, y1, x2, y2 = item.box
            where = f"({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})"
        elif item.line is not None:
            seen = int(np.isfinite(item.line[:, 0]).sum())
            where = f"{seen} of {len(item.line)} points"
        else:
            where = "no geometry"
        say(f"  {index:3d}  {item.label:9s} {item.score:.2f}  {where}")
    if len(found) > 10:
        say(f"  ... and {len(found) - 10} more")


class Writers:
    """The three optional outputs, opened as they are first needed and closed at the end.

    One stack cannot hold two frame sizes, but nothing about running the model needs
    them to agree -- so frames of different sizes are still run in one pass, in order,
    and it is only the FILES that are split: one set of outputs per size, named after
    it. A folder of one size, which is the ordinary case, has a single group and keeps
    exactly the names that were asked for.

    `sizes` is the size of every frame this run will reach, in order, so each stack can
    be made the right length before the first frame arrives.
    """

    def __init__(self, frames: int, classes, by: str = "class",
                 overlay=None, labels=None, roiset=None, sizes=None):
        self.classes = list(classes)
        self.by = by
        self.overlay_path = None if overlay is None else Path(overlay)
        self.labels_path = None if labels is None else Path(labels)
        self.roiset_path = None if roiset is None else Path(roiset)
        # position in the run -> (its size, its index within that size's own stack)
        self.slot: dict = {}
        length: dict = {}
        for position, size in enumerate(tuple(s) for s in (sizes or [])):
            self.slot[position] = (size, length.get(size, 0))
            length[size] = length.get(size, 0) + 1
        self.length = length or {None: frames}
        self.split = len(length) > 1
        self.open: dict = {}

    def _named(self, base: Path, size) -> Path:
        """The file a group writes to: the name asked for, plus its size where there
        is more than one group to tell apart."""
        if not self.split:
            return base
        return base.with_name(f"{base.stem}-{size[1]}x{size[0]}{base.suffix}")

    def _overlay_named(self, size, frames: int) -> Path:
        """...and the overlay alone has to become a stack once there are several
        frames of that size, whatever suffix was asked for."""
        return _stack_path(self._named(self.overlay_path, size), frames)

    def add(self, position: int, number: int, frame: np.ndarray,
            found: list[Found]) -> None:
        shape = tuple(frame.shape[:2])
        size, slot = self.slot.get(position, (None, position))
        if size is not None and size != shape:
            raise SystemExit(
                f"frame {number} is {shape[1]}x{shape[0]}, but its header said "
                f"{size[1]}x{size[0]}, so the stack made for it is the wrong shape.")
        frames = self.length[size]
        group = self.open.setdefault(size, {"overlay": None, "labels": None,
                                            "rois": [], "frames": frames})
        if self.overlay_path is not None:
            painted = draw(frame, found, self.by)
            if frames == 1:
                from PIL import Image

                Image.fromarray(painted).save(self._overlay_named(size, 1))
            else:
                if group["overlay"] is None:
                    group["overlay"] = _memmap(
                        self._overlay_named(size, frames),
                        (frames, *shape, 3), np.uint8, "TYXS")
                group["overlay"][slot] = painted
        if self.labels_path is not None:
            if group["labels"] is None:
                group["labels"] = LabelStack(
                    self._named(self.labels_path, size), self.classes, shape, frames)
            group["labels"].add(slot, found)
        if self.roiset_path is not None:
            group["rois"] += rois(found, slot, number, self.by)

    def close(self) -> None:
        for size, group in self.open.items():
            frames = group["frames"]
            if group["overlay"] is not None:
                group["overlay"].flush()
            if self.overlay_path is not None:
                say(f"wrote {self._overlay_named(size, frames)}")
            if group["labels"] is not None:
                group["labels"].close()
                say(f"wrote {self._named(self.labels_path, size)}  "
                    f"({len(self.classes)} channels: {', '.join(self.classes)})")
                group["labels"].report()
            if self.roiset_path is not None:
                write_rois(self._named(self.roiset_path, size), group["rois"])
