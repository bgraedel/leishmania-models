# leishmania-models

Detection and pose weights for [trackanno](https://github.com/bgraedel/trackanno),
published as release assets with an index that says how each one should be run.

Point trackanno at this repository once per machine, in *Configure -> From
registry...*, or set it directly:

```
https://github.com/bgraedel/leishmania-models/releases/latest/download/index.json
```

trackanno then downloads a model on first use, verifies its sha256, caches it,
and applies the settings below. Later runs are cache hits and never touch the
network.

## Layout

| | |
|---|---|
| `index.json` | what models exist, their digests, and the settings each needs |
| release assets | the `.pt` files themselves, one release per model version |
| `publish.py` | builds an `index.json` entry from a checkpoint |

**Weights are release assets, never git objects.** A 200 MB `.pt` committed here
would sit in every clone for ever, and git cannot forget it. `.gitignore` stops
that happening by accident.

## Adding a model

```bash
python publish.py best_pose.pt --id leishmania-pose --version v1 \
    --notes "8-point flagellum, brightfield 20x" \
    --imgsz 1280 --tile 640 --overlap 96 --fps 100 --pixel-size 0.325 \
    --nodes Head Base Flag1 Flag2 Flag3 Flag4 Flag5 Tip --chain
```

It reads the checkpoint for what it can (task, keypoint count, training
`imgsz`), hashes the file, and writes the entry. Then on GitHub: create a
release tagged `v1`, upload the `.pt` and `index.json` as its assets, and commit
`index.json` here so the repository records what was published.

## What an entry says

```json
{
  "id": "leishmania-pose", "version": "v1", "task": "pose",
  "url": "https://github.com/bgraedel/leishmania-models/releases/download/v1/best_pose.pt",
  "sha256": "9f3c...", "size": 52428800,
  "notes": "8-point flagellum, brightfield 20x",
  "config": {
    "detection": {"imgsz": 1280, "task": "pose"},
    "tiling": {"tile": 640, "overlap": 96},
    "keypoints": {"nodes": ["Head", "Base", "Flag1", "..."],
                  "edges": [["Head", "Base"], ["Base", "Flag1"]]}},
  "assumes": {"fps": 100, "pixel_size_um": 0.325}
}
```

The `config` block is the reason this repository exists rather than a folder of
`.pt` files. A checkpoint reports its task, keypoint count and class names, and
trackanno reads those directly. It cannot report the keypoint **names**, the
skeleton **edges**, the `imgsz` it was validated at, or a tile size that suits
it, and those are most of what a person otherwise types in by hand. They are
applied to the project and reported, never changed silently.

`assumes` is not applied. It becomes a warning where a project's movies were
acquired differently, since only the person knows whether that is the experiment
or a mistake.

## Versioning

One release per model version, and a tag is never moved. trackanno pins the
digest into the project on first fetch, so weights that changed under an
unchanged name are refused rather than silently substituted. Publish `v2` rather
than re-uploading `v1`.
