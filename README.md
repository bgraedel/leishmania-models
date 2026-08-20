# leishmania-models

Trained weights for *Leishmania* imaging, published as release assets with an
index that records how each model should be run.

Any framework, any task: ultralytics, cellpose, HuggingFace transformers, ONNX,
brightfield or fluorescence. Each entry names what loads it and carries its own
inference settings, preprocessing and acquisition assumptions, so a model can be
used correctly without anyone having to remember what it was trained on.

## Using them

The index is a release asset, so this URL always names the current set:

```
https://github.com/bgraedel/leishmania-models/releases/latest/download/index.json
```

In [trackanno](https://github.com/bgraedel/trackanno), paste it once per machine
under *Configure -> From registry...*. A model is then downloaded on first use,
verified against its sha256, cached, and its settings applied to the project.
Later runs are cache hits and never reach the network.

Anything else can read the same index: it is plain JSON, and each entry gives a
direct download URL and the digest to check it against.

To just run a model, `examples/` has a script per model that does that for you.
It fetches the weights, verifies them, runs the frames in tiles at the scale the
model was trained on, and writes an overlay, an ImageJ label stack or a RoiSet:

```bash
uv run examples/pose.py cells.tif
uv run examples/segment.py cells.tif --frames all --tiff labels.tif --rois masks.zip
```

Each script declares its own dependencies inline, so `uv run` needs nothing
installed first; any Python works too, once those are on it. Windows, macOS and
Linux — see [examples/README.md](examples/README.md).

## Layout

| | |
|---|---|
| `index.json` | every model, its digest, and the settings it needs |
| release assets | the weight files, one release per model version |
| `publish.py` | builds an index entry from a checkpoint |
| `examples/` | a script per model, taking its settings from the index |

**Weights are release assets, never git objects.** A 200 MB checkpoint committed
here would sit in every clone for ever, and git cannot forget it. `.gitignore`
stops that happening by accident.

## Adding a model

Run `publish.py` with **trackanno's own interpreter**, so it can read the
checkpoint for the task, keypoint count and training resolution. Any other
Python still works, but then those have to come from the flags.

```bash
git clone https://github.com/bgraedel/leishmania-models
cd leishmania-models

# Windows; on macOS or Linux it is .venv/bin/python
<trackanno>/.venv/Scripts/python.exe publish.py ...
```

```bash
# a detector
python publish.py best.pt --id leishmania-detect --version v1 \
    --framework ultralytics --modality brightfield --notes "promastigotes, 20x" \
    --imgsz 1280 --tile 640 --overlap 96

# a pose model, with the keypoint chain named
python publish.py best_pose.pt --id leishmania-pose --version v1 \
    --framework ultralytics --modality brightfield --notes "8-point flagellum, 20x" \
    --imgsz 1280 --tile 640 --overlap 96 --fps 100 --pixel-size 0.325 \
    --nodes Head Base Flag1 Flag2 Flag3 Flag4 Flag5 Tip --chain

# a cellpose model, trained across two magnifications
python publish.py CP_20240101 --id leishmania-cyto --version v1 \
    --framework cellpose --task segment --modality fluorescence \
    --preprocess cellpose.json --pixel-size 0.2 0.65

# a Mask2Former checkpoint from HuggingFace, zipped
python publish.py m2f.tar.gz --id leishmania-m2f --version v1 \
    --framework hf-transformers --task segment --modality fluorescence \
    --preprocess m2f.json
```

The script reads the checkpoint for what it can (task, keypoint count, the
`imgsz` it was trained at), hashes the file, and writes the entry with the
release URL already filled in. Then, on GitHub:

1. create a release tagged with the version;
2. upload **both** the weights and `index.json` as its assets;
3. commit `index.json` here as well, and push.

Step 2 is the one to get right. The registry URL points at
`releases/latest/download/index.json`, so `index.json` has to be a release
asset. A copy committed to the repository is a record of what was published,
and is not what anyone fetches.

## What an entry says

```json
{
  "id": "leishmania-pose", "version": "v1",
  "framework": "ultralytics", "task": "pose", "modality": "brightfield",
  "url": ".../releases/download/v1/best_pose.pt",
  "sha256": "9f3c...", "size": 52428800,
  "notes": "8-point flagellum, 20x",
  "config": {
    "detection": {"imgsz": 1280, "task": "pose"},
    "tiling": {"tile": 640, "overlap": 96},
    "keypoints": {"nodes": ["Head", "Base", "..."], "edges": [["Head", "Base"]]}},
  "preprocess": {},
  "assumes": {"fps": 100, "pixel_size_um": 0.325}
}
```

| field | meaning |
|---|---|
| `framework` | what loads the file: `ultralytics`, `cellpose`, `hf-transformers`, `onnx`, ... |
| `task` | `detect`, `pose`, `segment`, `classify`, `obb` |
| `modality` | `brightfield`, `fluorescence`, `phase`, ... |
| `sha256` | verified on download and pinned by the project that uses it |
| `config` | keyed by consumer; a reader takes its own key and ignores the rest |
| `preprocess` | how pixels reach the model: input size, normalisation, channels |
| `assumes` | the acquisition it was trained on |

`config` and `preprocess` are carried verbatim. Only the consumer interprets
them, so a cellpose entry can put `diameter` and `flow_threshold` under
`config.cellpose` and a trackanno entry puts `imgsz` under `config.detection`,
without either having to know about the other.

`assumes` takes a number or a list. A model trained across magnifications says
`"pixel_size_um": [0.2, 0.65]`, and a consumer checks whether the acquisition
falls inside that instead of comparing against one value. Nothing in `assumes`
is applied; it becomes a warning where a project disagrees.

**Any framework works.** The index is a name, a digest and some free-form
description, so what the bytes are is the consumer's problem. trackanno runs
`ultralytics` `detect` and `pose` models, lists everything else with what it is,
and declines to set it as a detector.

Weights that are a directory rather than a file (HuggingFace, some cellpose
setups) go up as a `.zip` or `.tar.gz`. It is fetched and verified like anything
else; unpacking is the consumer's job.

## Limits

**2 GB per release asset**, which is GitHub's cap. A YOLO pose model is
50-250 MB and a Swin-L Mask2Former around 800 MB, so this is rarely close.
Releases do not count towards the repository size, which is why weights go there
rather than into git.

## Versioning

One release per model version, and a tag is never moved. A consumer pins the
digest on first fetch, so weights that changed under an unchanged name are
refused rather than silently substituted. Publish `v2` rather than re-uploading
`v1`.
