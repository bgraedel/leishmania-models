# leishmania-models

Trained weights for *Leishmania* imaging, published as release assets with an
index that records how each model should be run.

Detection and pose models both live here. Each entry carries its own inference
settings, keypoint layout and acquisition assumptions, so a model can be used
correctly without anyone having to remember what it was trained on.

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

## Layout

| | |
|---|---|
| `index.json` | every model, its digest, and the settings it needs |
| release assets | the weight files, one release per model version |
| `publish.py` | builds an index entry from a checkpoint |

**Weights are release assets, never git objects.** A 200 MB checkpoint committed
here would sit in every clone for ever, and git cannot forget it. `.gitignore`
stops that happening by accident.

## Adding a model

```bash
# a detector
python publish.py best.pt --id leishmania-detect --version v1 \
    --notes "promastigotes, brightfield 20x" --imgsz 1280 --tile 640 --overlap 96

# a pose model, with the keypoint chain named
python publish.py best_pose.pt --id leishmania-pose --version v1 \
    --notes "8-point flagellum, brightfield 20x" \
    --imgsz 1280 --tile 640 --overlap 96 --fps 100 --pixel-size 0.325 \
    --nodes Head Base Flag1 Flag2 Flag3 Flag4 Flag5 Tip --chain
```

The script reads the checkpoint for what it can (task, keypoint count, the
`imgsz` it was trained at), hashes the file, and writes the entry. Then on
GitHub: create a release tagged with the version, upload the weights and
`index.json` as its assets, and commit `index.json` here so the repository
records what was published.

## What an entry says

```json
{
  "id": "leishmania-pose", "version": "v1", "task": "pose",
  "url": ".../releases/download/v1/best_pose.pt",
  "sha256": "9f3c...", "size": 52428800,
  "notes": "8-point flagellum, brightfield 20x",
  "config": {
    "detection": {"imgsz": 1280, "task": "pose"},
    "tiling": {"tile": 640, "overlap": 96},
    "keypoints": {"nodes": ["Head", "Base", "..."], "edges": [["Head", "Base"]]}},
  "assumes": {"fps": 100, "pixel_size_um": 0.325}
}
```

`config` is the reason this is a repository rather than a folder of checkpoints.
A checkpoint reports its task, keypoint count and class names, and those are read
from the file directly. It cannot report the keypoint **names**, the skeleton
**edges**, the resolution it was validated at, or a tile size that suits it, and
those are most of what is otherwise typed in by hand for every new project. Only
inference settings may appear there: a model says how it should be run, and
nothing about where anyone's data lives.

`assumes` is never applied. It becomes a warning where a project's movies were
acquired differently, since only the person knows whether a change of frame rate
or magnification is the experiment or a mistake.

## Versioning

One release per model version, and a tag is never moved. A consumer pins the
digest on first fetch, so weights that changed under an unchanged name are
refused rather than silently substituted. Publish `v2` rather than re-uploading
`v1`.
