#!/usr/bin/env python
"""Add a model to index.json, ready for a release.

    python publish.py best_pose.pt --id leishmania-pose --version v1 \
        --framework ultralytics --modality brightfield \
        --imgsz 1280 --tile 640 --overlap 0.15 --fps 100 --pixel-size 0.325 \
        --nodes Head Base Flag1 Flag2 Flag3 Flag4 Flag5 Tip --chain

Ultralytics checkpoints are read for their task, keypoint count and training
imgsz; everything else comes from the flags. The file is hashed, and that digest
is verified on every download and pinned by the project that uses it.

The weights are read where they are and never copied here. Re-running for an
existing id replaces that entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

INDEX = Path(__file__).with_name("index.json")
REPO = "bgraedel/leishmania-models"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def from_checkpoint(path: Path) -> dict:
    """What an ultralytics checkpoint says about itself, or {} if unreadable.

    Needs trackanno importable, which needs torch. Run this with trackanno's own
    interpreter, or pass --task, --imgsz and --nodes.
    """
    try:
        from trackanno.tiling import read_model_info
    except ImportError:
        sibling = Path(__file__).resolve().parents[1] / "trackanno" / "src"
        if sibling.is_dir():
            sys.path.insert(0, str(sibling))
        try:
            from trackanno.tiling import read_model_info
        except ImportError:
            print("  trackanno is not importable here; pass --task, --imgsz and"
                  "\n  --nodes yourself, or use trackanno's interpreter.")
            return {}
    try:
        return read_model_info(path)
    except Exception as exc:
        print(f"  could not read {path.name}: {exc}")
        return {}


def _extra_config(given: str | None) -> dict:
    """`--config`, as a JSON object. A leading brace is the object itself; anything
    else is the file holding it, which is the shape a long one wants.

    Every way of getting it wrong is answered with a sentence, the way the rest of this
    script answers: a missing file and a stray unquoted key are the two likely mistakes
    and neither is worth a traceback.
    """
    if not given:
        return {}
    if given.lstrip().startswith("{"):
        text = given
    else:
        try:
            text = Path(given).read_text(encoding="utf-8")
        except OSError as unreadable:
            raise SystemExit(f"--config {given}: {unreadable.strerror}. It takes the "
                             f"JSON object itself (starting with {{) or a file holding "
                             f"one.")
    try:
        extra = json.loads(text)
    except json.JSONDecodeError as malformed:
        raise SystemExit(f"--config is not valid JSON: {malformed}. Keys and strings "
                         f"both need double quotes, and a shell that strips them wants "
                         f"the object in a file instead.")
    if not isinstance(extra, dict):
        raise SystemExit(f"--config has to be a JSON object keyed by consumer, "
                         f"got {type(extra).__name__}")
    return extra


def _check_tiling(tiling: dict) -> None:
    """The same bounds the --tile and --overlap flags enforce.

    `--config` reaches the same keys by another road, and an entry published with an
    overlap of 0.95 is not caught here but by `tiling.settings` inside every run of it,
    which is every user rather than the one publishing.
    """
    overlap = tiling.get("overlap")
    if overlap is not None and not 0.0 <= overlap < 0.9:
        raise SystemExit(f"config.tiling.overlap is a fraction of the tile, 0 to below "
                         f"0.9; got {overlap}. Every run of the entry would refuse it.")
    tile = tiling.get("tile")
    if tile is not None and (not isinstance(tile, int) or tile < 0):
        raise SystemExit(f"config.tiling.tile is a size in pixels, or 0 for one pass "
                         f"over the whole frame; got {tile}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", type=Path,
                        help="the .pt wherever it already lives; it is not copied "
                             "here, and never committed")
    parser.add_argument("--id", required=True, help="the name projects will use")
    parser.add_argument("--version", required=True,
                        help="this model's version, independent of the release")
    parser.add_argument("--release", default="",
                        help="the release tag the asset is uploaded to; defaults "
                             "to --version. One release can hold several models")
    parser.add_argument("--notes", default="", help="one line, shown in the picker")
    parser.add_argument("--task", default="",
                        choices=["", "detect", "pose", "segment", "classify", "obb"],
                        help="usually read from the checkpoint; override if it is wrong")
    parser.add_argument("--modality", default="",
                        help="brightfield, fluorescence, phase, ... shown in the picker")
    parser.add_argument("--framework", default="ultralytics",
                        help="what loads the file: ultralytics, cellpose, "
                             "hf-transformers, onnx, torchscript, ...")
    parser.add_argument("--config", default=None,
                        help="extra settings as JSON, merged into config: the object "
                             'itself, or a path to a .json holding it. e.g. '
                             '\'{"cellpose": {"niter": 600, "flow_threshold": 0.6}}\'. '
                             "config is keyed by consumer and carried verbatim, so "
                             "this is where a setting goes that has no flag of its own")
    parser.add_argument("--preprocess", type=Path, default=None,
                        help="json describing how pixels reach the model (input "
                             "size, normalisation, channels). Needed for anything "
                             "ultralytics does not preprocess itself")
    parser.add_argument("--imgsz", type=int, default=0)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--overlap", type=float, default=0.0,
                        help="tile overlap as a FRACTION of the tile, 0 to 0.9. "
                             "Omit to leave the project default")
    parser.add_argument("--nodes", nargs="*", default=[], help="keypoint names, in model order")
    parser.add_argument("--chain", action="store_true",
                        help="wire the nodes 0-1, 1-2, ... which is a flagellum")
    parser.add_argument("--fps", type=float, nargs="*", default=None,
                        help="what it was trained at; give two for a range, or "
                             "several for discrete rates")
    parser.add_argument("--pixel-size", type=float, nargs="*", default=None,
                        help="um per pixel; give two for a range, e.g. "
                             "--pixel-size 0.2 0.65 for a multi-scale model")
    parser.add_argument("--magnification", type=float, nargs="*", default=None,
                        help="objectives it was trained on, e.g. 20 40 60 100")
    args = parser.parse_args(argv)

    if not args.weights.is_file():
        print(f"error: {args.weights} is not a file", file=sys.stderr)
        return 2

    release = args.release or args.version
    print(f"hashing {args.weights.name} ...")
    sha = digest(args.weights)
    size = args.weights.stat().st_size
    info = from_checkpoint(args.weights)

    task = args.task or info.get("task") or ""
    imgsz = args.imgsz or int(info.get("imgsz") or 0)
    nodes = list(args.nodes)
    if not nodes and info.get("n_keypoints"):
        nodes = [f"kp{i}" for i in range(int(info["n_keypoints"]))]
        print(f"  no --nodes given; using placeholders for {len(nodes)} keypoints")

    config: dict = {}
    detection = {k: v for k, v in (("imgsz", imgsz), ("conf", args.conf)) if v}
    if task:
        detection["task"] = task
    if detection:
        config["detection"] = detection
    if args.overlap and not 0.0 <= args.overlap < 0.9:
        print(f"error: --overlap is a fraction of the tile, got {args.overlap}",
              file=sys.stderr)
        return 2
    tiling = {k: v for k, v in (("tile", args.tile), ("overlap", args.overlap)) if v}
    if tiling:
        config["tiling"] = tiling
    if nodes:
        edges = [[nodes[i], nodes[i + 1]] for i in range(len(nodes) - 1)] if args.chain else []
        config["keypoints"] = {"nodes": nodes, "edges": edges}
    # Merged one key deep, so --config can add a cellpose setting without replacing
    # the tiling a flag put there, and can still override one key of it.
    for key, value in _extra_config(args.config).items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    _check_tiling(config.get("tiling") or {})

    # One value stays a scalar; several become a list, which is how a model
    # trained across magnifications says so. Anything reading this treats a
    # scalar as a range of zero width, so the two need no special casing.
    def scale(values):
        if not values:
            return None
        return values[0] if len(values) == 1 else sorted(values)

    assumes = {k: v for k, v in (("fps", scale(args.fps)),
                                 ("pixel_size_um", scale(args.pixel_size)),
                                 ("magnification", scale(args.magnification))) if v}

    preprocess = {}
    if args.preprocess:
        preprocess = json.loads(args.preprocess.read_text(encoding="utf-8"))
    elif args.framework != "ultralytics" and args.framework not in config:
        # ... unless config already carries a block for that framework, which is the
        # other place a consumer can be told how to run the thing. A cellpose 4 entry
        # takes that route: the network has no fixed input size to record, and what
        # normalisation it wants is one of the settings under config.cellpose.
        print(f"\nwarning: a {args.framework} model needs --preprocess, or whoever"
              "\n  runs it has to guess the input size and normalisation.")

    entry = {
        "id": args.id,
        "version": args.version,
        "task": task,
        "modality": args.modality,
        "framework": args.framework,
        "url": f"https://github.com/{REPO}/releases/download/{release}/{args.weights.name}",
        "sha256": sha,
        "size": size,
        "notes": args.notes,
        "config": config,
        "preprocess": preprocess,
        "assumes": assumes,
    }

    index = json.loads(INDEX.read_text(encoding="utf-8")) if INDEX.exists() else {"models": []}
    # Keyed by (id, version), so one id can offer several versions and a
    # project that pinned an older one keeps resolving to it.
    models = [m for m in index.get("models", [])
              if (m.get("id"), m.get("version")) != (args.id, args.version)]
    models.append(entry)
    index["models"] = sorted(models, key=lambda m: (m["id"], m.get("version", "")))
    INDEX.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {INDEX.name}: {args.id} {args.version}, {size / 1e6:.1f} MB")
    print(f"  sha256 {sha}")
    if args.framework != "ultralytics" or (task and task not in ("detect", "pose")):
        print("\nnote: trackanno runs ultralytics detect and pose models. This one "
              "is listed,\n  verified and downloadable there, and something else runs it.")

    if not task:
        print("\nwarning: no task recorded. The picker shows it, and trackanno uses"
              "\n  it to decide whether it can drive the model. Pass --task.")

    print("\nNext, on GitHub:")
    print(f"  1. create a release tagged {release}")
    print(f"  2. upload BOTH {args.weights.name} and index.json as its assets.")
    print("     index.json HAS to be an asset: the registry URL points at")
    print("     releases/latest/download/index.json, so a committed copy alone")
    print("     is not fetchable.")
    print("  3. commit index.json here as well, so the repository records what")
    print("     was published and the next publish starts from it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
