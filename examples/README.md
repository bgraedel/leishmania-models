# examples

A runnable script per model. Each fetches its own weights from the index, checks
them against the digest, runs a frame or a range of them in tiles, and writes an
overlay, a label stack, ImageJ rois, or all three.

Nothing is configured here. The tile size, the imgsz, the keypoint chain and the
preprocessing all come out of the index entry, so a republished model changes what
these do without any of them being edited.

| | |
|---|---|
| `detect.py` | `leishmania-detect` — a box round every cell |
| `segment.py` | `leishmania-seg` — masks for `animal`, `body` and `flagellum` |
| `pose.py` | `leishmania-pose` — eight ordered points, head to flagellar tip |
| `mask2former.py` | `leishmania-m2f-1024` and `leishmania-m2f-512-amodal` — body and flagellum masks |
| `fetch.py` | getting a model, and getting an image into the shape it was trained on |
| `tiling.py` | cutting a frame into tiles, and putting the pieces back |
| `outputs.py` | the overlay, the label stack, the RoiSet |

## Running them

With [uv](https://docs.astral.sh/uv/) there is nothing to install. Each script
carries its own dependencies inline (PEP 723), and `uv run` builds a throwaway
environment for it:

```bash
uv run examples/detect.py cells.tif
uv run examples/segment.py cells.tif --frames 0-99
uv run examples/pose.py cells.tif --version v1
uv run examples/mask2former.py cells.tif --id leishmania-m2f-512-amodal
```

Or install the dependencies yourself and use any Python:

```bash
pip install ultralytics tifffile                        # detect.py, segment.py, pose.py
pip install torch transformers scipy pillow tifffile    # mask2former.py
pip install roifile                                     # only for --rois

python examples/detect.py cells.tif
```

`scipy` is not optional for `mask2former.py`: transformers refuses to build a
Mask2Former without it. The mask cleanup below wants scipy or OpenCV, and both lines
above already bring one — ultralytics ships OpenCV, and Mask2Former brings scipy.

`uv run` takes torch from PyPI, whose Linux wheel already carries CUDA and whose
Windows one is CPU-only. On a Windows box with a card, name PyTorch's own index:

```bash
uv run --index https://download.pytorch.org/whl/cu124 examples/detect.py cells.tif
```

The input is a `.tif` stack or any single image. `--version` takes an older model
where the index offers several; without it the newest wins.

Weights are cached in `~/.cache/leishmania-models` and downloaded once. From a
clone, `LEISHMANIA_MODELS_INDEX=index.json` reads the committed index instead of
fetching the published one; `LEISHMANIA_MODELS_CACHE` moves the cache.

## Tiling

`detect` and `seg` were trained on 640 px tiles at full resolution. Run whole, a
2048 px frame reaches them at a third of that scale and the smallest cells go with
it, so by default the frame is cut into tiles the size the index records and each
one goes through at 1.0x:

```
2048x2048: 16 tiles of 640 px, overlap 25% (160 px)
```

It is worth the trouble. On one 2048 px frame the detector finds **256 cells tiled
against 220 whole** — the 36 it gains are the small and faint ones that a 3x
downscale erases.

Tiles overlap so that a cell landing on a seam is whole in at least one of them, and
the pieces are put back together afterwards. **Masks are unioned, boxes are
suppressed** — a flagellum can be longer than the overlap is deep and genuinely
arrives in two halves, both real, whereas a box has no shape to union and the fuller
copy simply wins.

Both rules key on whether a detection *runs into a seam*, which is what separates a
cut object from a whole one:

- Something that stops short of every tile edge is all the model had to say about it.
  A second copy of it is only a duplicate if it more or less coincides — high IoU for
  a box, or half the smaller mask for a mask.
- Something that touches a seam was interrupted there. A box is then judged on
  *containment*: nearly all of a clipped box lies inside the whole one, where its IoU
  with it is only about a half, so IoU alone leaves every seam cell boxed twice. A
  mask is allowed to join on a couple of shared pixels, which is what rejoins a cell
  too long to fit in any tile.

That distinction matters most for the amodal model, whose masks are drawn through
each other on purpose: two cells crossing share real pixels, and a blanket
few-pixels rule fuses them into one. And only instances from *different* tiles are
ever compared — two cells lying against each other inside one tile are two cells, and
the model already said so.

`--tile 0` runs the whole frame in one pass at the imgsz the index records.
`--overlap` is a fraction of the tile: it has to be wider than the longest object, or
a cell longer than the overlap is never whole in any tile and no merging afterwards
invents the part that was never seen. `--batch` is how many tiles go through at once.

`pose.py` is deliberately **not** tiled. The model is scale-robust, running whole
frames at the 1280 the index records was measured to find more cells than presenting
them at nominal scale, and a tile edge through a chain loses the node order that is
the whole point.

## The four modes

All four scripts take the same flags, so whatever you learn on one works on the rest.

```bash
python examples/segment.py cells.tif \
    --frames 0-99 --color instance \
    --out overlay.tif --tiff labels.tif --rois masks.zip
```

**`--frames`** — `0` (the default), `10-19`, `0-499:5`, or `all`. Every output below
grows a T axis to match, and the printing collapses to one line per frame.

**`--color class|instance`** — `class` gives body one colour and flagellum another;
`instance` walks the hue circle by the golden angle, so two cells that touch never
come out the same colour. Nothing tracks between frames, so an instance is only the
n-th thing found in its own frame and its hue will change from one frame to the next.

**`--out`** — the overlay. A `.png` for a single frame, an ImageJ RGB stack for a
range (a `.png` name is switched to `.tif`, since one file has to hold them all).

**`--tiff`** — label images as an ImageJ hyperstack, `TCYX`, **one channel per
class**, `uint16`. Pixel values are the instance's number in that frame, so each
channel is a label image rather than a binary mask and Analyze Particles takes it
unchanged. The classes overlap by design — `animal` contains `body` and `flagellum`,
and the amodal model's masks cross each other — which is exactly why they get a
channel each rather than one flattened label image. The stack is memory-mapped and
filled a frame at a time, so `--frames all` costs disk, not RAM.

A box has no mask, so `detect.py` fills its rectangle in; a chain has no area, so
`pose.py` draws it as a one-pixel line. Both land in the same file format as the
segmenters.

**`--rois`** — a RoiSet `.zip`, one ImageJ roi per instance, that *File → Open* drops
straight into the ROI manager. Masks become polygons, boxes become their four
corners, a pose chain becomes a polyline. Each roi is named
`<frame>-<class>-<instance>` and outlined in the colour `--color` chose, and its
number is the same one the label stack wrote — the two files line up. The class is in
the name rather than in the roi's channel on purpose: a RoiSet pinned to channel 2
vanishes when it is opened over the original single-channel movie, which is where
most of these end up.

A mask that comes back in two pieces contributes only its largest outline. Splitting
it into two rois would need a naming scheme to say they belong together, and that is
not this code's decision to make.

## Cleaning up what the model said

Mask2Former will predict a good blob and a few specks elsewhere in the tile, all
under one instance id. On one frame through the 512 model, **32 of 421 raw instances
came back in more than one piece**. The specks are not another cell and nothing
downstream will split them off, so they go before the instance is placed in the
frame.

`--components union`, the default, drops components below
`max(--min-component-area, --min-component-frac × the instance)` — 60 px or 5% —
and keeps the union of the rest; where every component is tiny it keeps the largest.
`--components largest` keeps only the biggest, and `--components all` leaves the mask
as the model drew it. `--min-area` then drops any instance too small to be a cell at
all. Only components *within one instance* are touched; whether two instances are
really one is the tiling's question, not this one.

It matters beyond tidiness. A speck lying across a tile seam is enough to make the
cross-tile merge fuse two cells into one, so cleaning up first *raises* the count
rather than lowering it. The same cleanup runs on the yolo-seg masks, where it is
very nearly a no-op — those come back in one piece.

`--classes body,flagellum` renames the model's classes by position. A checkpoint
whose `id2label` was never filled in reports `LABEL_0`, `LABEL_1`, and those names go
on to title the channels of the label stack and every roi in the RoiSet; the scripts
warn when they see them.

## Windows, macOS and Linux

Pure Python throughout — no shell-outs, no platform paths, nothing compiled here.
Python 3.9 or newer; tested on 3.9 and 3.12.

`--device` takes `auto` (the default), `cpu`, `cuda`, `mps`, or a CUDA index like
`0`. On `auto`, `mask2former.py` picks CUDA, then Apple's `mps`, then the CPU, and
ultralytics does the same for the other three. If an `mps` run stops on a missing
Metal kernel, `PYTORCH_ENABLE_MPS_FALLBACK=1` sends those operations to the CPU
rather than raising.

One macOS wrinkle worth knowing: a python.org build whose root certificates were
never installed fails the download with `certificate verify failed`. Running
`/Applications/Python 3.x/Install Certificates.command` once fixes it for good.

## Mask2Former wants VRAM

`post_process_instance_segmentation` resamples every one of the model's ~100 query
masks to the tile size before discarding the low-scoring ones, so one 1024 px tile
peaks at about **2.0 GB of VRAM** and two at once want 3.8. Past what the card has,
the driver spills to system memory over PCIe. Measured on a 2 GB card: one tile 5 s,
two tiles 38 s — which is why `--batch` defaults to 1 in that script alone.

`leishmania-m2f-512-amodal` costs a quarter as much per tile and is the one to reach
for on a small card: one frame of nine tiles with every output above took 27 s there,
against 2 m 29 s for the 1024 model on the CPU. `--device cpu` is a fallback, not a
speed-up.

The real fix is to drop the sub-threshold queries *before* the resample, which means
replacing the library's post-processing — worth doing in a pipeline, more than these
scripts should carry.

## The contrast stretch is not optional

`load_image` maps each frame's 0.05 and 99.95 percentiles to 0 and 255, which is what
the training data was. A uint16 microscopy frame is mostly background, so dividing by
256 instead gives a flat grey image in which these models find very little. If
detections are inexplicably poor, this is the first thing to check.

## What these are not

One pass per frame, and no tracking: instance numbers and `--color instance` hues are
per frame and do not follow a cell through a movie.
[trackanno](https://github.com/bgraedel/trackanno) reads the same index and tracks
with the `detect` and `pose` models.
