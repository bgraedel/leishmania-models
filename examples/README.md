# examples

A runnable script per model: it fetches the weights from the index, checks them against
the digest, runs a frame, a range, or a folder, and writes an overlay, a label stack,
ImageJ rois, or all three. Input size, tile size, keypoint chain and preprocessing come
from the index entry, so a republished model changes what these do.

The shared per-frame loop — tiling, the overlap warning, stitching, the cell numbers
`--guide` attaches — is `run.py`, so `--stitch` and `--guide` mean one thing everywhere.

| | |
|---|---|
| `detect.py` | `leishmania-detect` — a box round every cell |
| `segment.py` | `leishmania-seg` — masks for `animal`, `body` and `flagellum` |
| `pose.py` | `leishmania-pose` — eight ordered points, head to flagellar tip |
| `mask2former.py` | `leishmania-m2f-1024`, `leishmania-m2f-512-amodal` — body and flagellum |
| `cellpose_seg.py` | `leishmania-cellpose-{body,animal,flagellum}` — cellpose 4, one mask per cell |
| `gui.py` | a window over all of them: pick a model, pick folders, run |
| `guide.py` | the tile plan for a frame, without running a segmenter on it |
| `run.py` | the per-frame loop every script shares: tiles, stitching, reporting |
| `fetch.py` | getting a model, and getting an image into the shape it was trained on |
| `tiling.py` | cutting a frame into tiles, and putting the pieces back |
| `outputs.py` | the overlay, the label stack, the RoiSet |

## The window

```bash
python examples/gui.py
uv run examples/gui.py
```

tkinter, so nothing to install. Pick a model, point it at a file or a folder, pick an
output folder, press Run; the output streams into the bottom pane. A folder of images
is one run per image, and the scripts name each image's outputs themselves, so for a
folder the window hands them the output folder alone.

**The form is built from each script's own `arguments()` parser**, so a flag added to a
script appears here with no edit to `gui.py`. Only settings moved off their default are
passed, and the command is shown first:

```
C:\...\python.exe C:\...\examples\segment.py cells.tif --tile 640 --guide
    --out results\cells.leishmania-seg.png --rois results\cells.leishmania-seg-rois.zip
```

**Copy command** copies it; output files are named from the output folder and the
input's own name. **run with** is this interpreter where it can already import what the
model needs, else `uv run`. Settings, folders and the chosen model are remembered per
script, beside the cache.

### What the index fills in

Fetched in the background; the window works without it.

- **`--id`** lists the models this script runs, by `framework` and `task`.
- **`--version`** and **`--guide-version`** list published versions. Editable: the index
  says what is published, not what may be typed.
- A line under the picker names the entry a run would resolve to (`fetch.newest`, the
  rule the run uses), with its notes and pixel sizes.
- **What the entry already records** is said beside the flag that would change it, since
  it is otherwise resolved unseen inside the run.

```
leishmania-seg v1 — yolo26m-seg, 3 classes (animal, body, flagellum). mask mAP50 0.968 · 0.065-0.325 um/px

--imgsz  [0]     override the inference size  ·  the index records 640 for this model
```

Nothing here changes a default. An unpublished model says so and points at `--weights`.
**Reload index** re-reads it.

## Running them

With [uv](https://docs.astral.sh/uv/) there is nothing to install — each script carries
its dependencies inline (PEP 723):

```bash
uv run examples/detect.py cells.tif
uv run examples/segment.py cells.tif --frames 0-99
uv run examples/mask2former.py cells.tif --id leishmania-m2f-512-amodal
uv run examples/cellpose_seg.py cells.tif --niter 600
```

Or install them yourself and use any Python 3.9+:

```bash
pip install "ultralytics>=8.4.142" tifffile tqdm                   # detect, segment, pose
pip install torch torchvision transformers scipy pillow tifffile   # mask2former.py
pip install "cellpose>=4.0.1" scipy                                # cellpose_seg.py
pip install roifile                                                # for --rois
```

`mask2former.py` needs both `torchvision` and `scipy`: transformers builds its image
processors on the first and refuses to build a Mask2Former without the second.

`uv run` takes torch from PyPI, whose Linux wheel carries CUDA and whose Windows one is
CPU-only. On Windows with a card, name PyTorch's index:

```bash
uv run --index https://download.pytorch.org/whl/cu124 examples/detect.py cells.tif
```

`--device` takes `auto` (default), `cpu`, `cuda`, `mps`, or a CUDA index. If an `mps`
run stops on a missing Metal kernel, set `PYTORCH_ENABLE_MPS_FALLBACK=1`.

`--nms 0.5` runs an ultralytics model's one-to-many head through NMS at that IoU
instead of its end-to-end head, which the scripts run otherwise. They say so outright,
since ultralytics 8.4.142 itself defaults a YOLO26 model to NMS at 0.7, and they need
that version or newer for its `nms` argument. The two heads keep different cells in a
crowd: on one 1392x1040 frame the detector found 39 cells end-to-end and 49, 42 and 35
with NMS at 0.7, 0.5 and 0.3. `--guide-nms` does the same for the detector `--guide`
runs. `detect.py`'s `--iou` is something else: the box merger's threshold under
`--stitch merge`.

Weights cache in `~/.cache/leishmania-models`; `LEISHMANIA_MODELS_INDEX=index.json`
reads a local index, `LEISHMANIA_MODELS_CACHE` moves the cache.

`--weights` runs an unpublished checkpoint: a `.pt` for the ultralytics scripts, a
checkpoint folder (or any file inside it) for `mask2former.py`, a file or a cellpose
model name for `cellpose_seg.py`. It skips the index, so flags like `--tile` have to say
what the model needs, and the outputs are named after the file.

## Input

A `.tif` stack, a single image, or a folder. A folder is one run per image in it, in
name order -- a file browser's, `frame_2` before `frame_10`, case ignored -- each
writing its own outputs, named after the image, into the folders `--out`, `--tiff` and
`--rois` name; see [Outputs](#outputs). Mixed formats are fine; non-images are
ignored. `--frames` applies within each image, so a folder of stacks runs frame 0 of
every stack unless told otherwise.

`--sequence` runs a folder as one movie instead: its images in name order, a stack in
it contributing its pages in turn, so `--frames`, the T axis and the progress all count
across the folder, and one stack and one RoiSet come out.

`--frames` takes `0` (default), `10-19`, `10-`, `0-499:5` or `all`.

Frames of different sizes run in one pass, but one stack cannot hold two shapes, so the
outputs are split per size and suffixed with it:

```
ov-696x520.tif, labels-696x520.tif, rois-696x520.zip
ov-500x400.tif, labels-500x400.tif, rois-500x400.zip
```

Sizes come from the file headers before the model loads. A folder of one size keeps the
filenames you asked for, unsuffixed.

`load_image` maps each frame's 0.05 and 99.95 percentiles to 0 and 255, matching the
training data. A uint16 frame divided by 256 instead comes out flat grey and these
models find little in it — check this first if detections are poor.

## Tiling

`detect` and `seg` were trained on 640 px tiles at full resolution; a 2048 px frame run
whole reaches them at a third of that scale, and the small cells go with it.

Tiling is opt-in. By default a frame goes in whole at the index's input size, and the
run says what that costs:

```
2048x2048: one pass over the whole frame
imgsz 640: cells at 0.31x native
not tiling; --tile 640 is what the index records, and is what puts cells at 1.0x
```

`--tile 640` cuts tiles that size, each going through at 1.0x. On one 2048 px frame the
detector finds 256 cells tiled against 220 whole: the gain is the small and faint ones,
the cost a cell on a seam seen twice or in halves. `--overlap` is a fraction of the
tile, `--batch` tiles per forward pass.

### Putting the pieces back

`--stitch` picks the scheme.

**`core`** (default on a grid) — ownership, as in StarDist's `predict_instances_big`:
each tile owns a non-overlapping core, the rest is context, and an instance is kept only
where its centroid falls in that core. Nothing is compared, so nothing is double-counted.

**`merge`** — unions masks across tiles. Two masks are one object when they agree about
the strip their tiles share. Boxes have no shape to union, so `merge_boxes` uses IoU
plus containment, largest copy first.

**`whole`** (default with `--guide`) — keeps each instance in the tile it sits most
centrally in.

Only instances from different tiles are compared, so two cells touching inside one tile
stay two, keeping the amodal model's overlapping masks apart.

Measured on a 1392x1040 frame against a whole-frame pass finding 118 instances,
`animal` at 42. *cut* counts instances whose mask stops at a seam of its own tile;
*biggest* is the largest mask, where a real cell is about 11.5k px.

| tile/overlap | `core` n / animal | cut | biggest | `merge` n / animal | biggest |
|---|---|---|---|---|---|
| 640/0.25 *(default)* | 115 / 48 | 6 | 11.6k | 121 / 49 | 11.8k |
| **1024/0.5** | **114 / 43** | **0** | **11.5k** | 115 / 45 | 13.5k |

Neither scheme fuses cells. `core` double-counts a cell longer than the margin from a
core to its tile edge. The geometry matters more than the scheme — `--tile 1024
--overlap 0.5` is what this frame wants.

### The overlap has to exceed the longest object

A cell that fits in no tile is never seen whole. This is checked against what the tiles
found:

```
warning: 37% of what was found is longer than the 160 px overlap (longest 453 px).
Those cells fit in no tile whole, so no merging afterwards can make them so --
raise --overlap above 0.71, or --guide to place the tiles on the cells instead.
```

## --guide

A grid knows nothing about the frame and cuts through whatever lies on a seam.
`--guide` runs the detector first and places the tiles on the cells it found, each cell
whole inside one tile:

```bash
python segment.py cells.tif --guide
python mask2former.py cells.tif --guide --guide-pad 48
python guide.py cells.tif                 # the plan alone
```

A tile at origin `o` holds a box whole when `far - tile <= o <= near`, so the fewest
tiles is interval stabbing. `tiling.cover` solves it once down the frame and once across
each row, drops tiles nothing needs, and centres what is left. Stabbing one axis and then
the other is optimal along each axis but not over the plane, and which axis goes first
can cost a tile, so both orders are planned and the smaller kept; on random frames that
leaves about one plan in twenty a tile over the true minimum. A tile is settled flush
with a frame edge where its margin allows, since an edge there is no seam.

```
1392x1040: 7 guided tiles of 640 px for 45 detected cells, 45 of them whole (6 on the grid)
```

At promastigote lengths this costs no more tiles than the grid; where cells are long it
costs tiles and removes the duplication. The run prints both numbers.

Every mask then carries its cell's number, so an animal, its body and its flagellum
share one number in the label stack and the RoiSet. `--guide-strict` keeps only masks
that matched a cell.

A guided run looks only where the detector pointed, so a cell it missed is never
segmented. `--guide-conf` is the dial, and lower is safer here than for detection: a
false box costs tile area, a missed one costs a cell. With no detections the frame falls
back to the grid and says so. `--guide-pad` asks for clear space between each cell and
its tile's edges, given up on a side where the frame edge is closer than that, and
where one tile serves cells too far apart to pad all of them. A tile whose room reaches
a frame edge is then settled flush with it, since an edge there is no seam.
`--guide-nms` runs the detector with NMS at that IoU instead of end-to-end, as `--nms`
does for a segmenter; a duplicated box costs a plan a cell number, a missed one a cell.

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

**`--out`** — the overlay: a `.png` for one frame, an ImageJ RGB stack for a range.

**`--tiff`** — label images as an ImageJ hyperstack, `TCYX`, one channel per class,
`uint16`. Pixel values are the instance's number in that frame, so each channel is a
label image Analyze Particles takes unchanged; classes overlap (`animal` contains `body`
and `flagellum`), hence a channel each. Memory-mapped, so `--frames all` costs disk
rather than RAM. A box is filled in, a chain drawn as a one-pixel line.

**`--rois`** — a RoiSet `.zip`, one roi per instance, that *File → Open* drops into the
ROI manager. Masks become polygons, boxes their four corners, chains polylines. Each roi
is named `<frame>-<class>-<instance>`, carries the number the label stack wrote, and is
outlined in `--color`'s colour. The class is in the name because a RoiSet pinned to
channel 2 vanishes over a single-channel movie.

**`--color class|instance`** — a colour per class, or a hue per instance by the golden
angle. Nothing tracks between frames, so instance hues change frame to frame.

**A folder of images** makes each of these flags name a *folder*, and every image
writes its own set into it, named after itself and the model: `cells.tif` gives
`cells.leishmania-seg.png`, `cells.leishmania-seg-labels.tif` and
`cells.leishmania-seg-rois.zip`. Without `--out` the overlays go to a folder beside
the input named after it and the model, `frames.leishmania-seg/`. A file-shaped path
is refused rather than overwritten by every image in turn; `--sequence` is the way to
ask for one stack across the folder.

A number is taken by at most one instance of a class; a second of the same class in one
cell is numbered after the last cell. A mask returning in two pieces contributes only
its largest outline.

## Cleaning up masks

Mask2Former will predict a good blob and a few specks under one instance id: on one
frame through the 512 model, 32 of 421 raw instances came back in more than one piece.

`--components union` (default) drops components below
`max(--min-component-area, --min-component-frac × the instance)` — 60 px or 5% — and
keeps the union of the rest. `--components largest` keeps the biggest; `--components
all` leaves the mask alone. `--min-area` drops instances too small to be a cell. Only
components within one instance are touched.

A speck across a seam can make the merge fuse two cells, so cleaning first raises the
count. On yolo-seg masks it is close to a no-op.

`--classes body,flagellum` renames classes by position, for a checkpoint whose
`id2label` reports `LABEL_0`, `LABEL_1`. The scripts warn when they see those.

## mask2former.py

The crop is cut at the frame's own resolution and resized up to the network's input, so
`--tile` is in frame pixels; it defaults to 0, the whole frame in one pass.
`input_size / native_scale_factor` — 512 px for both models — reproduces the training
scale. Any other value presents cells larger or smaller and says so:

```
1024x1024: 16 tiles of 384 px, overlap 25% (96 px) -> a 512 px input (1.33x)
warning: this model was trained on 512 px crops; 384 px presents cells at 1.33x ...
```

That 384 px run found 38 instances against 114 at the right scale.

The resize is isotropic. A non-square crop — only at the edge of a frame smaller than
one tile — is padded square first, or every cell would arrive at the wrong aspect.

One 1024 px input peaks at about 2.0 GB of VRAM and two at once want 3.8. Past what the
card has, the driver spills to system memory: on a 2 GB card, one tile 5 s and two tiles
38 s, which is why `--batch` defaults to 1 here. `leishmania-m2f-512-amodal` costs a
quarter as much per tile — one frame of nine tiles with every output took 27 s, against
2 m 29 s for the 1024 model on the CPU.

## cellpose_seg.py

Cellpose 4's syntax, which is `CellposeModel` and nothing else: the `Cellpose` wrapper,
the size model and the `channels` argument are gone, the network takes three channels in
any order, and `diameter` is the only thing that rescales a frame. Not named
`cellpose.py`, because a script's own folder comes first on the import path and
`from cellpose import models` would find itself.

**Cellpose blocks the frame up itself** — 256 px at a tenth of overlap for a `cpsam`
backbone, 384 for `cpdino` — and averages the *flows* where blocks meet before running
the dynamics once over the whole frame. Cells are already at 1.0x and no instance is
ever cut, so there is nothing to stitch. `--tile` here is for a frame too large to hold
at once and for `--guide`, not the scale knob it is in `detect.py` and `segment.py`.
`--bsize` and `--block-overlap` are cellpose's own blocks; `--tile` and `--overlap` are
this script's.

```
1392x1040: one pass over the whole frame
cellpose sam_vitl: 256 px blocks, 10% overlap, cells at 1.00x, niter 200, on cuda
flow 0.4, cellprob 0.0, min_size 15, normalize frame (7-236 -> 0-1); 1 frame(s), classes: cell
```

### The dials

| flag | default | |
|---|---|---|
| `--niter` | 200 | how far the dynamics walk. **It is a path length**: a promastigote with its flagellum is a few hundred pixels of one, and pixels that never reach the cell's centre come back as their own mask or as nothing. First thing to raise here. |
| `--flow-threshold` | 0.4 | a mask whose flows disagree with the model by more than this is thrown away. An elongated cell is exactly what scores badly; `0` turns the check off. |
| `--cellprob-threshold`, `--prob` | 0.0 | where the dynamics start. Lower finds more cells and fatter ones, and a flagellum is a faint few pixels wide. |
| `--diameter` | 0 | resize so a cell this many px across arrives at cellpose's 30. `0` does not resize, which is what a cellpose-4 model expects. It also sets the default `--niter`, and in the *other* direction: the dynamics still run at the frame's own resolution, so `--diameter 60` halves what the network sees and **doubles** the walk, to 400. |
| `--min-size` | 15 | cellpose drops masks under this before the script sees them. `--min-area` does the same afterwards, to masks already cleaned. |
| `--max-size-fraction` | 0.4 | drops a mask covering more than this of the image — of the *tile*, when tiling. |
| `--resample` | on | run the dynamics at the frame's own resolution; it is what makes an outline follow the cell. |
| `--augment` | off | each block four ways, averaged, at four times the forward passes. |
| `--batch` | 8 | blocks per forward pass. |

Any of them can be set by the index entry under `config.cellpose`, and the run says
which the entry set. A flag beats the entry.

`cpsam` is a ViT-L, so it wants a card. On a 2 GB MX450 the default `--batch 8` spills
to system memory and a 512 px crop had not finished in five minutes; `--batch 1` did the
same crop in **32 s**, model load included, against **5 m 10 s** on the CPU. Same 17
cells either way — `--precision` is `bf16` on a card and `f32` on the CPU, differing by
a few pixels per mask.

### Tiling costs you here

Cellpose has already solved at the flow level what `--tile` re-creates at the instance
level. The same crop, whole and cut into nine tiles:

| | instances |
|---|---|
| one pass over the whole frame | **17** |
| `--tile 256 --overlap 0.25` | 24 |

Those extra seven are one cell each seen in halves, and the run says so — 78% of what
the tiles found was longer than the 64 px overlap. Reach for `--tile` when a frame does
not fit, and pair it with `--guide`, which leaves nothing cut:

```
--guide places tiles, so it turns tiling on: 512 px
512x512: 1 guided tiles of 512 px for 10 detected cells, 10 of them whole
17 instances: cell x17 over 9 detected cell(s)
```

Every mask then carries the detector's cell number, so a cellpose run and a `segment.py`
run number the same cell the same way.

### Normalisation

Cellpose normalises **every image it is handed** to its 1st and 99th percentiles, and a
tile is an image — a 512 px crop holding one bright cell has different percentiles from
its frame. `--normalize frame` (the default here) measures once per frame and hands
every crop the same two numbers, so tiling changes the field of view and nothing else.
`tile` is cellpose's own behaviour. `off` measures no percentile and takes the
0.05/99.95 stretch `load_image` already applied as the whole of it. All three found the
same 17 cells on one crop.

`off` is a fixed map from 0–255 onto 0–1. It is **not** cellpose's `normalize=False`,
which hands the network 0–255 where it wants 0–1: that found 1 cell against 17.

`--percentile`, `--tile-norm` (uneven illumination), `--sharpen` and `--invert` are the
rest of cellpose's normalisation. The middle two are measured from the image, which
cellpose will not do alongside fixed bounds, so asking for either falls back to
per-image percentiles.

### Which model

`--weights` skips the index: a checkpoint on disk, or one of cellpose's own names
(`cpsam`, `cpdino`, or anything in `~/.cellpose/models`).

```bash
python cellpose_seg.py cells.tif --weights leish_cp_animal --niter 600
python cellpose_seg.py cells.tif --weights cpsam --diameter 30
```

A name cellpose cannot place is refused here rather than passed on: cellpose warns about
a `pretrained_model` it cannot find and then quietly loads its default, so the run would
finish, look ordinary, and be the wrong model.

Cellpose scores no instances, but `--stitch` picks which copy of a cell survives by
score and the label stack gives a contested pixel to the higher-scoring instance. That
score is the mean cell probability under the mask, squashed to 0–1.

## Scope

One pass per frame, no tracking: instance numbers and `--color instance` hues are per
frame and do not follow a cell through a movie.
[trackanno](https://github.com/bgraedel/trackanno) reads the same index and tracks with
the `detect` and `pose` models.

Pure Python; Python 3.9 or newer, tested on 3.9 and 3.12. On macOS, a python.org build
whose root certificates were never installed fails downloads with `certificate verify
failed`; run `/Applications/Python 3.x/Install Certificates.command` once.
