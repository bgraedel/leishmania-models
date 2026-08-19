#!/usr/bin/env python
"""Add a model to index.json, ready for a release.

    python publish.py best_pose.pt --id leishmania-pose --version v1 \
        --notes "8-point flagellum, brightfield 20x" \
        --imgsz 1280 --tile 640 --overlap 96 --fps 100 --pixel-size 0.325 \
        --nodes Head Base Flag1 Flag2 Flag3 Flag4 Flag5 Tip --chain

Reads the checkpoint for what it can (task, keypoint count, the imgsz it was
trained at), takes the rest from the flags, hashes the file, and writes the
entry. Then upload the .pt and index.json as assets of a release tagged with
``--version``.

The digest is the point of the exercise: trackanno verifies it on download and
pins it into the project, so weights cannot change under a name that stayed the
same. Re-running this for an existing id replaces that entry.
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
    """Whatever the .pt can be made to say about itself, or {} without torch."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trackanno" / "src"))
        from trackanno.tiling import read_model_info

        return read_model_info(path)
    except Exception as exc:                       # torch absent, or a odd file
        print(f"  (could not read the checkpoint: {exc})")
        return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--id", required=True, help="the name projects will use")
    parser.add_argument("--version", required=True, help="also the release tag")
    parser.add_argument("--notes", default="", help="one line, shown in the picker")
    parser.add_argument("--task", default="", choices=["", "detect", "pose"])
    parser.add_argument("--imgsz", type=int, default=0)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--nodes", nargs="*", default=[], help="keypoint names, in model order")
    parser.add_argument("--chain", action="store_true",
                        help="wire the nodes 0-1, 1-2, ... which is a flagellum")
    parser.add_argument("--fps", type=float, default=None, help="what it was trained at")
    parser.add_argument("--pixel-size", type=float, default=None, help="um per pixel")
    args = parser.parse_args(argv)

    if not args.weights.is_file():
        print(f"error: {args.weights} is not a file", file=sys.stderr)
        return 2

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
    tiling = {k: v for k, v in (("tile", args.tile), ("overlap", args.overlap)) if v}
    if tiling:
        config["tiling"] = tiling
    if nodes:
        edges = [[nodes[i], nodes[i + 1]] for i in range(len(nodes) - 1)] if args.chain else []
        config["keypoints"] = {"nodes": nodes, "edges": edges}

    assumes = {k: v for k, v in (("fps", args.fps), ("pixel_size_um", args.pixel_size)) if v}

    entry = {
        "id": args.id,
        "version": args.version,
        "task": task,
        "url": f"https://github.com/{REPO}/releases/download/{args.version}/{args.weights.name}",
        "sha256": sha,
        "size": size,
        "notes": args.notes,
        "config": config,
        "assumes": assumes,
    }

    index = json.loads(INDEX.read_text(encoding="utf-8")) if INDEX.exists() else {"models": []}
    models = [m for m in index.get("models", []) if m.get("id") != args.id]
    models.append(entry)
    index["models"] = sorted(models, key=lambda m: m["id"])
    INDEX.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {INDEX.name}: {args.id} {args.version}, {size / 1e6:.1f} MB")
    print(f"  sha256 {sha}")
    print("\nNext, on GitHub:")
    print(f"  1. create a release tagged {args.version}")
    print(f"  2. upload {args.weights.name} and index.json as its assets")
    print(f"  3. commit index.json here too, so the repo records what was published")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
