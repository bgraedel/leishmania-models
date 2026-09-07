#!/usr/bin/env python
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "torch",
#     # --guide places tiles with the detector, an ultralytics checkpoint.
#     "ultralytics>=8.4.142",
#     # transformers 5 builds its image processors on torchvision.
#     "torchvision",
#     "transformers",
#     "scipy",
#     "pillow",
#     "tifffile",
#     "numpy",
#     "roifile",
#     "opencv-python-headless",
#     "tqdm",
# ]
# ///
"""leishmania-m2f-1024 and leishmania-m2f-512-amodal: Mask2Former body and flagellum.

    python mask2former.py cells.tif
    python mask2former.py cells.tif --id leishmania-m2f-512-amodal --tiff labels.tif
    python mask2former.py cells.tif --frames 0-19 --color instance --rois masks.zip
    python mask2former.py cells.tif --tile 384        # smaller crops, cells larger

Two classes, body and flagellum.

By default the whole frame goes through in one pass: squared off to its long side and
resized to the checkpoint's input, so nothing is stretched and a larger frame arrives
scaled down; the run says by how much. `--tile N` cuts N px crops at the frame's own
resolution and resizes each up to that input, so cells arrive at the trained scale. The
run says which N does that -- `input_size / native_scale_factor`, 512 for the 1024
model -- and warns for any other. Each instance is kept by the one tile whose core
holds its centroid; `--stitch merge` unions pieces across neighbouring tiles instead.
`--guide` places the tiles on detected cells rather than on a grid; see guide.py.

  leishmania-m2f-1024         Swin-S, 512 px crops resized 2x to a 1024 px input
  leishmania-m2f-512-amodal   Swin-B, 512 px crops at native resolution, and amodal:
                              it completes a cell through whatever occludes it, so
                              masks overlap where two cells cross

The checkpoints ship `do_resize` off; the `size` beside it is already the network's
input, so turning it on lets a crop of any size reach that input at the right scale.

VRAM: post_process_instance_segmentation resamples all ~100 query masks to
`target_sizes` before dropping the low-scoring ones, so cost follows the area asked
back, not what survives. A 1392x1040 frame squares off to 1392 -- ~770 MB of float32
for that step; a 512 px tile is 100 MB, paid one tile at a time. Past the card's
memory the driver spills over PCIe, so --batch defaults to 1 here. On a small card try
`--tile 512` first, then the 512 model over the 1024 one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fetch import fetch, on_disk
from guide import add_guide_arguments, guide_from, tiling_on
from outputs import (Found, add_mask_arguments, add_output_arguments, cleanup_from,
                     frame_count, instance_mask, parse_frames, progress, runs_of,
                     resolve_classes, say, writers_for)
from run import pick_device, run_frames
from tiling import (DEFAULT_OVERLAP, add_stitch_argument, batches, resolve_stitch,
                    settings)

MODEL = "leishmania-m2f-1024"


def predict(model, processor, names, frame, boxes, conf, batch, device, cleanup,
            square: int):
    """Every instance in one frame, each mask in the frame's own coordinates.

    `square` is the size every crop is padded to before it goes in -- square because
    the processor resizes to the network's fixed square input, which would otherwise
    stretch a non-square crop. Masks come back at the crop's own resolution and are cut
    down to the real crop before placement, so the padding contributes nothing.
    """
    import torch

    found = []
    for first, group in progress(list(batches(boxes, batch)), "tiles", leave=False):
        crops, extents = [], []
        for y0, x0, y1, x1 in group:
            crop = frame[y0:y1, x0:x1]
            extents.append(crop.shape[:2])
            if crop.shape[:2] != (square, square):
                padded = np.zeros((square, square) + crop.shape[2:], crop.dtype)
                padded[:crop.shape[0], :crop.shape[1]] = crop
                crop = padded
            crops.append(crop)
        inputs = processor(images=crops, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # target_sizes is the crop, not the network's input: every query mask is
        # resampled to it, so the smaller of the two is most of the cost.
        # return_binary_maps keeps one mask per instance; the label-map form would
        # collapse overlapping amodal masks into one image.
        results = processor.post_process_instance_segmentation(
            outputs, target_sizes=[crop.shape[:2] for crop in crops],
            threshold=conf, return_binary_maps=True)
        for tile_index, ((y0, x0, _, _), (rows, cols), result) in enumerate(
                zip(group, extents, results), first):
            masks = result["segmentation"]
            if masks is None:
                continue
            masks = masks.cpu().numpy().astype(bool)
            for order, info in enumerate(result["segments_info"]):
                placed = instance_mask(masks[order][:rows, :cols], (y0, x0), *cleanup)
                if placed is None:
                    continue
                patch, origin = placed
                found.append(Found(label=names[int(info["label_id"])],
                                   score=float(info.get("score", 0.0)),
                                   mask=patch, origin=origin, source=tile_index))
    return found


def arguments() -> argparse.ArgumentParser:
    """This script's flags, as a parser gui.py can build a form out of."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path,
                        help="a .tif stack, a single image, or a FOLDER of images "
                             "taken in name order")
    parser.add_argument("--id", default=MODEL, help=f"which model; default {MODEL}")
    parser.add_argument("--version", default=None, help="index version; default is the newest")
    parser.add_argument("--weights", default=None,
                        help="a Mask2Former checkpoint folder on disk, or any file "
                             "inside one; skips the index")
    parser.add_argument("--conf", type=float, default=0.5, help="keep instances above this")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    tiling = parser.add_argument_group("tiling")
    tiling.add_argument("--tile", type=int, default=0,
                        help="crop size in frame px; 0 (default) is the whole frame in "
                             "one pass. N puts cells at the trained scale -- the run "
                             "says which N does that")
    tiling.add_argument("--overlap", type=float, default=None,
                        help=f"tile overlap as a fraction of the tile, 0 to 0.9 "
                             f"(default {DEFAULT_OVERLAP} where the index says nothing)")
    tiling.add_argument("--batch", type=int, default=1,
                        help="tiles per forward pass; raise it if the card has room")
    add_stitch_argument(tiling)
    add_guide_arguments(parser)
    add_mask_arguments(parser)
    add_output_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = arguments().parse_args(argv)

    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    from transformers.image_utils import PILImageResampling

    if args.weights:
        chosen = Path(args.weights)
        # A checkpoint here is a FOLDER; a file inside it names its folder.
        weights, entry = on_disk(chosen.parent if chosen.is_file() else chosen)
    else:
        weights, entry = fetch(args.id, args.version)
    tile, overlap = settings(entry, args.tile, args.overlap)
    name = args.id if not args.weights else weights.stem
    jobs = runs_of(args, name)

    device = pick_device(args.device)
    how = resolve_stitch(args.stitch, args.guide)
    guide = guide_from(args, device, args.batch)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(weights).to(device).eval()
    # The shipped processor carries the normalisation, size divisor and network input;
    # do_resize is the one thing off in it. Bicubic because that is what this model's
    # training frames were upsampled with.
    processor = AutoImageProcessor.from_pretrained(
        weights, do_resize=True, resample=PILImageResampling.BICUBIC)
    # The input size comes from the PROCESSOR, which does the resizing, not the index.
    shipped = dict(getattr(processor, "size", None) or {})
    recorded = int(((entry.get("preprocess") or {}).get("input_size") or [0])[0])
    size = int(shipped.get("height") or shipped.get("shortest_edge") or recorded)
    if not size:
        raise SystemExit(f"{weights}: neither the checkpoint's processor nor the "
                         f"index records an input size for this model")
    if recorded and size != recorded:
        say(f"warning: the index records a {recorded} px input for {args.id} and the "
            f"checkpoint's own processor says {size}; going with the checkpoint")
    # The crop that fills that input at the scale the model was trained on.
    native = max(1, size // int((entry.get("preprocess") or {})
                                .get("native_scale_factor", 1)))
    # Settled here because `native` is not known until the processor has been read.
    tile, overlap = tiling_on(args, entry, tile, overlap, native,
                              why=f", the crop that reaches a {size} px input at "
                                  f"this model's own scale")
    names = {int(key): value for key, value in model.config.id2label.items()}
    classes = resolve_classes(names, args.classes)
    names = dict(zip(sorted(names), classes))  # --classes reaches the instances too
    cleanup = cleanup_from(args)

    numbers: list = []  # the current run's frames; `announce` reads it

    def square(frame) -> int:
        """The size every crop is padded out to before it goes in.

        Untiled, the whole frame is the crop. Either way it is squared off before the
        resize to the network's fixed square input, which would otherwise stretch a
        non-square crop. Capped at the frame, so a --tile larger than the image does
        not spend the input on padding.
        """
        longest = max(frame.shape[:2])
        return min(tile, longest) if tile else longest

    def suffix(frame, boxes):
        crop = square(frame)
        return f" -> a {size} px input ({size / crop:.3g}x), on {device}"

    def announce(frame, boxes):
        crop = square(frame)
        if crop != native:
            say(f"warning: this model was trained on {native} px crops; {crop} px "
                f"presents cells at {native / crop:.3g}x the scale it knows"
                + ("" if tile else f" -- --tile {native} presents them as trained"))
        say(f"{len(numbers)} frame(s), classes: {', '.join(classes)}")

    def run(frame, boxes):
        return predict(model, processor, names, frame, boxes, args.conf, args.batch,
                       device, cleanup, square=square(frame))

    for job in progress(jobs, "images"):
        numbers = parse_frames(args.frames, frame_count(job.image))
        writers = writers_for(job, job.image, name, classes, numbers)
        run_frames(job.image, numbers, writers, run, tile=tile, overlap=overlap,
                   how=how, guide=guide, strict=args.guide_strict, suffix=suffix,
                   announce=announce, item=job.item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
