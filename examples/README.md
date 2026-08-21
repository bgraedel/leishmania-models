# examples

A runnable script per model. Each fetches its weights from the index, checks them
against the digest, runs a frame, a range, or a folder of images, and writes an
overlay, a label stack, ImageJ rois, or all three.

Settings come from the index entry — input size, tile size, keypoint chain,
preprocessing — so a republished model changes what these do without an edit here.

| | |
|---|---|
| `detect.py` | `leishmania-detect` — a box round every cell |
| `segment.py` | `leishmania-seg` — masks for `animal`, `body` and `flagellum` |
| `pose.py` | `leishmania-pose` — eight ordered points, head to flagellar tip |
| `mask2former.py` | `leishmania-m2f-1024`, `leishmania-m2f-512-amodal` — body and flagellum |
| `guide.py` | the tile plan for a frame, without running a segmenter on it |
| `fetch.py` | getting a model, and getting an image into the shape it was trained on |
| `tiling.py` | cutting a frame into tiles, and putting the pieces back |
| `outputs.py` | the overlay, the label stack, the RoiSet |

## Running them

With [uv](https://docs.astral.sh/uv/) there is nothing to install. Each script carries
its dependencies inline (PEP 723):

```bash
uv run examples/detect.py cells.tif
uv run examples/segment.py cells.tif --frames 0-99
uv run examples/mask2former.py cells.tif --id leishmania-m2f-512-amodal
```

Or install them yourself and use any Python 3.9+:

```bash
pip install ultralytics tifffile tqdm                              # detect, segment, pose
pip install torch torchvision transformers scipy pillow tifffile   # mask2former.py
pip install roifile                                                # for --rois
```

`torchvision` and `scipy` are required for `mask2former.py`: transformers builds its
image processors on the first and refuses to build a Mask2Former without the second.

`uv run` takes torch from PyPI, whose Linux wheel carries CUDA and whose Windows one is
CPU-only. On Windows with a card, name PyTorch's index:

```bash
uv run --index https://download.pytorch.org/whl/cu124 examples/detect.py cells.tif
```

`--device` takes `auto` (default), `cpu`, `cuda`, `mps`, or a CUDA index. If an `mps`
run stops on a missing Metal kernel, set `PYTORCH_ENABLE_MPS_FALLBACK=1`.

Weights cache in `~/.cache/leishmania-models`. `LEISHMANIA_MODELS_INDEX=index.json`
reads a local index; `LEISHMANIA_MODELS_CACHE` moves the cache.

## Input

A `.tif` stack, a single image, or a folder. A folder is taken in name order as one
sequence, a stack in it contributing its pages in turn, so `--frames`, the T axis of
the outputs and the progress all count across the folder as they do across one stack.
Mixed formats are fine and non-images are ignored.

`--frames` takes `0` (default), `10-19`, `10-`, `0-499:5` or `all`.

Frames of different sizes run in one pass. One stack cannot hold two shapes, so the
files are split — one set of outputs per size, suffixed with it:

```
ov-696x520.tif, labels-696x520.tif, rois-696x520.zip
ov-500x400.tif, labels-500x400.tif, rois-500x400.zip
```

Sizes are read from the file headers before the model loads. A folder of one size gets
the filenames you asked for, unsuffixed.

`load_image` maps each frame's 0.05 and 99.95 percentiles to 0 and 255, matching the
training data. A uint16 frame divided by 256 instead comes out flat grey and these
models find little in it. Check this first if detections are poor.

## Tiling

`detect` and `seg` were trained on 640 px tiles at full resolution. A 2048 px frame run
whole reaches them at a third of that scale, and the small cells go with it.

Tiling is opt-in. By default a frame goes in whole at the index's input size, and the
run says what that costs:

```
2048x2048: one pass over the whole frame
imgsz 640: cells at 0.31x native
not tiling; --tile 640 is what the index records, and is what puts cells at 1.0x
```

`--tile 640` cuts the frame into tiles that size, each going through at 1.0x. On one
2048 px frame the detector finds 256 cells tiled against 220 whole; the gain is the
small and faint ones. The cost is that a cell on a seam is seen twice or in halves.

`--overlap` is a fraction of the tile. `--batch` is tiles per forward pass.

### Putting the pieces back

`--stitch` picks the scheme.

**`core`** (default on a grid) is ownership, as in StarDist's
`predict_instances_big`: each tile owns a non-overlapping core, the rest of the tile is
context, and an instance is kept only where its centroid falls in that core. Nothing is
compared, so nothing can be double-counted or chained.

**`merge`** unions masks across tiles. Two masks are one object when they agree about
the strip their tiles share — shared pixels over what either has inside that strip.
That strip is the only ground both tiles looked at. Boxes have no shape to union, so
`merge_boxes` uses IoU plus containment and takes the largest copy first.

**`whole`** (default with `--guide`) keeps each instance in the tile it sits most
centrally in.

Only instances from different tiles are compared, so two cells lying against each other
inside one tile stay two, which keeps the amodal model's overlapping masks apart.

Measured on a 1392x1040 frame against a whole-frame pass finding 118 instances,
`animal` at 42. *cut* counts instances whose mask stops at a seam of its own tile;
*biggest* is the largest mask, where a real cell is about 11.5k px.

| tile/overlap | `core` n / animal | cut | biggest | `merge` n / animal | biggest |
|---|---|---|---|---|---|
| 640/0.25 *(default)* | 115 / 48 | 6 | 11.6k | 121 / 49 | 11.8k |
| **1024/0.5** | **114 / 43** | **0** | **11.5k** | 115 / 45 | 13.5k |

Neither scheme fuses cells: both keep their largest mask at real-cell size. `core`
double-counts a cell longer than the margin from a core to its tile edge. The geometry
matters more than the scheme — `--tile 1024 --overlap 0.5` is what this frame wants.

### The overlap has to exceed the longest object

A cell that fits in no tile is never seen whole, so the pieces are all there is. This is
checked against what the tiles found:

```
warning: 37% of what was found is longer than the 160 px overlap (longest 453 px).
Those cells fit in no tile whole, so no merging afterwards can make them so --
raise --overlap above 0.71, or --guide to place the tiles on the cells instead.
```

## --guide

A grid is laid down knowing nothing about the frame, so it cuts through whatever lies
on a seam. `--guide` runs the detector first and places the tiles on the cells it found,
each cell inside one tile in one piece:

```bash
python segment.py cells.tif --guide
python mask2former.py cells.tif --guide --guide-pad 48
python guide.py cells.tif                 # the plan alone
```

A tile at origin `o` holds a box whole when `far - tile <= o <= near`, so each cell is a
range of origins per axis and the fewest tiles is interval stabbing. `tiling.cover`
solves it once down the frame and once across each row, drops tiles nothing needs, and
centres what is left.

```
1392x1040: 7 guided tiles of 640 px for 45 detected cells, 45 of them whole (6 on the grid)
```

At promastigote lengths this costs no more tiles than the grid. Where cells are long it
costs tiles and removes the duplication; the run prints both numbers.

Every mask then carries the number of the cell it belongs to, so an animal, its body and
its flagellum share one number in the label stack and the RoiSet. `--guide-strict` keeps
only masks that matched a cell, making the detector the register of what is in the frame.

A guided run looks only where the detector pointed, so a cell it missed is never
segmented. `--guide-conf` is the dial, and lower is safer here than for detection: a
false box costs tile area, a missed one costs a cell. Where the detector finds nothing,
the frame falls back to the grid and the run says so.

`--guide-pad` asks for clear space around each cell. It is given up at a frame edge, and
where one tile serves cells too far apart to pad all of them.

### pose

`pose.py` defaults to whole frames. Tiling works with `--stitch core`, which never joins
anything, so a chain is only ever kept whole by one tile. `--stitch merge` is refused: a
chain is an ordering, and two orderings cannot be spliced.

| | cells | all 8 nodes |
|---|---|---|
| whole frame at imgsz 1280 *(default)* | 30 | 100% |
| tile 640, overlap 0.5 | **41** | **100%** |

`detect` finds 44 cells on this frame, so 41 is closer to the truth than 30.

## Outputs

All four scripts take the same flags.

```bash
python examples/segment.py cells.tif \
    --frames 0-99 --color instance \
    --out overlay.tif --tiff labels.tif --rois masks.zip
```

**`--out`** — the overlay. A `.png` for one frame, an ImageJ RGB stack for a range.

**`--tiff`** — label images as an ImageJ hyperstack, `TCYX`, one channel per class,
`uint16`. Pixel values are the instance's number in that frame, so each channel is a
label image and Analyze Particles takes it unchanged. Classes overlap by design —
`animal` contains `body` and `flagellum` — which is why they get a channel each. The
stack is memory-mapped, so `--frames all` costs disk rather than RAM. A box is filled
in; a chain is drawn as a one-pixel line.

**`--rois`** — a RoiSet `.zip`, one roi per instance, that *File → Open* drops into the
ROI manager. Masks become polygons, boxes their four corners, chains polylines. Each roi
is named `<frame>-<class>-<instance>`, carries the number the label stack wrote, and is
outlined in the colour `--color` chose. The class lives in the name rather than the
roi's channel, since a RoiSet pinned to channel 2 vanishes when opened over a
single-channel movie.

A number is taken by at most one instance of a class. Where a cell holds two of the same
class, the second is numbered after the last cell so neither outline is overwritten.

**`--color class|instance`** — a colour per class, or a hue per instance walking the
circle by the golden angle. Nothing tracks between frames, so instance hues change from
frame to frame.

A mask returning in two pieces contributes only its largest outline.

## Cleaning up masks

Mask2Former will predict a good blob and a few specks under one instance id. On one
frame through the 512 model, 32 of 421 raw instances came back in more than one piece.

`--components union` (default) drops components below
`max(--min-component-area, --min-component-frac × the instance)` — 60 px or 5% — and
keeps the union of the rest. `--components largest` keeps the biggest; `--components
all` leaves the mask alone. `--min-area` drops instances too small to be a cell. Only
components within one instance are touched.

A speck lying across a seam can make the merge fuse two cells, so cleaning first raises
the count. On yolo-seg masks it is close to a no-op.

`--classes body,flagellum` renames classes by position, for a checkpoint whose
`id2label` reports `LABEL_0`, `LABEL_1`. The scripts warn when they see those.

## mask2former.py

The crop is cut at the frame's own resolution and resized up to the network's input, so
`--tile` is in frame pixels. It defaults to 0, the whole frame in one pass.
`input_size / native_scale_factor` — 512 px for both models — is the crop that
reproduces the training scale; any other value presents cells larger or smaller and says
so:

```
1024x1024: 16 tiles of 384 px, overlap 25% (96 px) -> a 512 px input (1.33x)
warning: this model was trained on 512 px crops; 384 px presents cells at 1.33x ...
```

That 384 px run found 38 instances against 114 at the right scale.

The resize is isotropic. A non-square crop, which happens only at the edge of a frame
smaller than one tile, is padded to a square first; resizing it straight would put every
cell at the wrong aspect.

One 1024 px input peaks at about 2.0 GB of VRAM and two at once want 3.8. Past what the
card has, the driver spills to system memory: on a 2 GB card, one tile 5 s and two tiles
38 s, which is why `--batch` defaults to 1 in this script.
`leishmania-m2f-512-amodal` costs a quarter as much per tile — one frame of nine tiles
with every output took 27 s, against 2 m 29 s for the 1024 model on the CPU.

## Scope

One pass per frame, no tracking. Instance numbers and `--color instance` hues are per
frame and do not follow a cell through a movie.
[trackanno](https://github.com/bgraedel/trackanno) reads the same index and tracks with
the `detect` and `pose` models.

Pure Python throughout; Python 3.9 or newer, tested on 3.9 and 3.12. On macOS, a
python.org build whose root certificates were never installed fails downloads with
`certificate verify failed`; run `/Applications/Python 3.x/Install Certificates.command`
once.
