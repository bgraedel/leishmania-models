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
    """

    label: str
    score: float
    mask: np.ndarray | None = None  # bool, the size of its own bounding box
    origin: tuple[int, int] = (0, 0)  # (y, x) of mask[0, 0] in the frame
    box: tuple[float, float, float, float] | None = None
    line: np.ndarray | None = None  # (n, 2), NaN where a point is not visible
    edges: list | None = None  # pairs of indices into `line`; consecutive if absent
    source: int = 0  # which tile it came out of; see tiling.merge_masks

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
        print(f"warning: this checkpoint calls its classes {', '.join(ordered)}. "
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
    # whole point of colouring by instance. Nothing tracks between frames, so an
    # instance is only the n-th thing found in its own frame and its hue will change.
    hue = (index * 0.6180339887498949) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return round(255 * red), round(255 * green), round(255 * blue)


# Frames


def frame_count(path) -> int:
    """How many frames a file holds. Anything that is not a stack holds one."""
    path = Path(path)
    if path.suffix.lower() not in (".tif", ".tiff"):
        return 1
    import tifffile

    with tifffile.TiffFile(path) as handle:
        shape = handle.series[0].shape
    if len(shape) >= 3 and shape[-1] not in (3, 4):
        return int(shape[0])
    return 1


def parse_frames(spec: str, available: int) -> list[int]:
    """`0`, `10-19`, `0-99:5` or `all` -> the frame numbers to run."""
    spec = str(spec).strip()
    if spec == "all":
        first, last, step = 0, available - 1, 1
    else:
        span, _, tail = spec.partition(":")
        first_text, dash, last_text = span.partition("-")
        try:
            first = int(first_text)
            step = int(tail or 1)
            last = int(last_text) if dash and last_text else first
        except ValueError:
            raise SystemExit(f"--frames {spec}: expected 0, 10-19, 0-99:5 or all")
    chosen = list(range(first, min(last, available - 1) + 1, max(1, step)))
    if not chosen:
        raise SystemExit(f"--frames {spec}: nothing in that range; the file has {available}")
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
    for index, item in enumerate(found):
        if item.mask is None:
            continue
        rgb = np.array(colour(index, item.label, by), np.float32)
        y0, x0, y1, x1 = item.bounds
        window = painted[y0:y1, x0:x1]
        window[item.mask] = (alpha * rgb + (1 - alpha) * window[item.mask]).astype(np.uint8)

    canvas = Image.fromarray(painted)
    pen = ImageDraw.Draw(canvas)
    for index, item in enumerate(found):
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


def _memmap(path, shape, dtype, axes, labels=None):
    import tifffile

    metadata = {"axes": axes}
    if labels:
        metadata["Labels"] = labels
    return tifffile.memmap(str(path), shape=shape, dtype=dtype, imagej=True,
                           metadata=metadata)


class LabelStack:
    """Label images, one channel per class, one time point per frame.

    Pixel values are the instance's number within its frame, so a channel is a label
    image rather than a binary mask and Analyze Particles or any label reader takes it
    unchanged. A box is filled in and a chain is drawn as a one-pixel line, so the
    detector and the pose model land in the same file format as the segmenters.

    The number runs across the whole frame rather than restarting per channel, so a
    channel skips values -- and the value in the stack is the one in the roi's name,
    which is what makes the two files line up.
    """

    def __init__(self, path, classes: list[str], shape: tuple[int, int], frames: int):
        self.classes = list(classes)
        self.channel = {name: index for index, name in enumerate(self.classes)}
        self.stack = _memmap(path, (frames, len(self.classes), *shape), np.uint16,
                             "TCYX", labels=self.classes * frames)

    def add(self, position: int, found: list[Found]) -> None:
        for index, item in enumerate(found, 1):
            if item.label not in self.channel:
                continue
            plane = self.stack[position, self.channel[item.label]]
            if item.mask is not None:
                y0, x0, y1, x1 = item.bounds
                plane[y0:y1, x0:x1][item.mask] = index
            elif item.box is not None:
                x1, y1, x2, y2 = (int(round(value)) for value in item.box)
                plane[max(y1, 0):y2 + 1, max(x1, 0):x2 + 1] = index
            elif item.line is not None:
                cv2 = opencv()
                for first, last in segments(item):
                    # plain ints: older OpenCV refuses a numpy scalar in a point
                    start = (int(round(first[0])), int(round(first[1])))
                    end = (int(round(last[0])), int(round(last[1])))
                    cv2.line(plane, start, end, int(index), 1)

    def close(self) -> None:
        self.stack.flush()


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


def rois(found: list[Found], position: int, by: str = "class") -> list:
    """One roi per instance, named <frame>-<class>-<instance>.

    The class is carried in the name and the outline colour rather than in the roi's
    channel: a RoiSet pinned to channel 2 disappears when it is opened over the
    original single-channel movie, which is where most of these end up.
    """
    import roifile

    made = []
    for index, item in enumerate(found, 1):
        if item.mask is not None:
            points, kind = _outline(item), roifile.ROI_TYPE.POLYGON
        elif item.line is not None:
            points = np.array([(x, y) for x, y in item.line if not np.isnan(x)])
            points = points if len(points) > 1 else None
            kind = roifile.ROI_TYPE.POLYLINE
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
            points, name=f"{position:04d}-{item.label}-{index:03d}",
            t=position, position=position)
        roi.roitype = kind
        roi.stroke_color = bytes((255, *colour(index - 1, item.label, by)))
        made.append(roi)
    return made


def write_rois(path, made: list) -> None:
    import roifile

    if not made:
        print("nothing to put in a RoiSet")
        return
    roifile.roiwrite(str(path), made, mode="w")
    print(f"wrote {path}  ({len(made)} rois)")


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


def report(number: int, found: list[Found], total: int) -> None:
    """One line per frame over a range, the whole list for a single frame."""
    from collections import Counter

    counts = Counter(item.label for item in found)
    summary = ", ".join(f"{name} x{count}" for name, count in counts.items()) or "nothing"
    if total > 1:
        print(f"  frame {number}: {len(found)} instances -- {summary}")
        return
    print(f"{len(found)} instances: {summary}")
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
        print(f"  {index:3d}  {item.label:9s} {item.score:.2f}  {where}")
    if len(found) > 10:
        print(f"  ... and {len(found) - 10} more")


class Writers:
    """The three optional outputs, opened on the first frame and closed at the end.

    Each is created only if it was asked for, and the stacks are sized from the first
    frame, so nothing here needs the image opened twice.
    """

    def __init__(self, frames: int, classes, by: str = "class",
                 overlay=None, labels=None, roiset=None):
        self.frames = frames
        self.classes = list(classes)
        self.by = by
        self.overlay_path = None if overlay is None else _stack_path(Path(overlay), frames)
        self.labels_path = None if labels is None else Path(labels)
        self.roiset_path = None if roiset is None else Path(roiset)
        self.overlay = None
        self.labels = None
        self.collected: list = []

    def add(self, position: int, number: int, frame: np.ndarray,
            found: list[Found]) -> None:
        if self.overlay_path is not None:
            painted = draw(frame, found, self.by)
            if self.frames == 1:
                from PIL import Image

                Image.fromarray(painted).save(self.overlay_path)
            else:
                if self.overlay is None:
                    self.overlay = _memmap(self.overlay_path,
                                           (self.frames, *frame.shape[:2], 3),
                                           np.uint8, "TYXS")
                self.overlay[position] = painted
        if self.labels_path is not None:
            if self.labels is None:
                self.labels = LabelStack(self.labels_path, self.classes,
                                         frame.shape[:2], self.frames)
            self.labels.add(position, found)
        if self.roiset_path is not None:
            self.collected += rois(found, number, self.by)

    def close(self) -> None:
        if self.overlay is not None:
            self.overlay.flush()
        if self.overlay_path is not None:
            print(f"wrote {self.overlay_path}")
        if self.labels is not None:
            self.labels.close()
            print(f"wrote {self.labels_path}  ({len(self.classes)} channels: "
                  f"{', '.join(self.classes)})")
        if self.roiset_path is not None:
            write_rois(self.roiset_path, self.collected)
