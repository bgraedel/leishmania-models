#!/usr/bin/env python
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "torch",
#     "transformers",
#     "scipy",
#     "pillow",
#     "tifffile",
#     "numpy",
#     "roifile",
#     "opencv-python-headless",
# ]
# ///
"""leishmania-m2f-1024 and leishmania-m2f-512-amodal: Mask2Former body and flagellum.

    python mask2former.py cells.tif
    python mask2former.py cells.tif --id leishmania-m2f-512-amodal --tiff labels.tif
    python mask2former.py cells.tif --frames 0-19 --color instance --rois masks.zip

Two classes, body and flagellum. `preprocess` in the index says the image processor
does not resize, so the frame is upsampled by native_scale_factor, cut into
input_size tiles at that scale, and every tile reaches the network at exactly the
size it was trained on. Nothing downstream corrects a wrong scale, which is why the
tile size here comes from the model rather than from a flag. Tiles overlap, and
pieces of the same cell found in neighbouring tiles are unioned back together.

  leishmania-m2f-1024         Swin-S, 1024 px tiles of a 2x upsampled frame
  leishmania-m2f-512-amodal   Swin-B, 512 px tiles at native resolution, and amodal:
                              it completes a cell through whatever occludes it, so
                              masks overlap where two cells cross

**A 1024 px tile wants a card with room to spare.** transformers' own
post_process_instance_segmentation resamples every one of the model's ~100 query
masks to the tile size before throwing the low-scoring ones away, so one tile peaks
around 2.0 GB of VRAM and two at once want 3.8. Past what the card has, the driver
spills to system memory over PCIe. Measured on a 2 GB card: one tile 5 s, two 38 s,
which is why --batch defaults to 1. The 512 model costs a quarter as much per tile,
and is the one to reach for on a small card -- it was still four times quicker there
than the CPU. Dropping the sub-threshold queries before the resample is the real fix
and means replacing the library's post-processing, which is a pipeline's job.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fetch import fetch, load_image
from outputs import (Found, Writers, add_mask_arguments, add_output_arguments,
                     frame_count, instance_mask, parse_frames, report, resolve_classes)
from tiling import DEFAULT_OVERLAP, batches, describe, merge_masks, slices

MODEL = "leishmania-m2f-1024"


def pick_device(name: str) -> str:
    """cuda, then Apple's mps, then the cpu -- unless the flag names one."""
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        # Some Mask2Former ops have no Metal kernel; PYTORCH_ENABLE_MPS_FALLBACK=1
        # sends those to the cpu rather than raising.
        return "mps"
    return "cpu"


def predict(model, processor, names, scaled, boxes, scale, conf, batch, device,
            cleanup):
    """Every instance in one frame, each mask put back into the original frame.

    A tile's mask is placed at the frame's own resolution rather than the upsampled
    one: native_scale_factor is a whole number, so undoing the upsample is a stride,
    and doing it here keeps a crowded 2x frame from carrying its masks twice the size
    they need to be.
    """
    import torch

    found = []
    for first, group in batches(boxes, batch):
        crops = [scaled[y0:y1, x0:x1] for y0, x0, y1, x1 in group]
        inputs = processor(images=crops, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # return_binary_maps keeps one mask per instance. The label-map form collapses
        # them into a single image, and amodal masks that overlap cannot survive that.
        results = processor.post_process_instance_segmentation(
            outputs, target_sizes=[crop.shape[:2] for crop in crops],
            threshold=conf, return_binary_maps=True)
        for tile_index, ((y0, x0, _, _), result) in enumerate(zip(group, results), first):
            masks = result["segmentation"]
            if masks is None:
                continue
            masks = masks.cpu().numpy().astype(bool)
            for order, info in enumerate(result["segments_info"]):
                placed = instance_mask(masks[order][::scale, ::scale],
                                       (y0 // scale, x0 // scale), *cleanup)
                if placed is None:
                    continue
                patch, origin = placed
                found.append(Found(label=names[int(info["label_id"])],
                                   score=float(info.get("score", 0.0)),
                                   mask=patch, origin=origin, source=tile_index))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="a .tif stack, or any single image")
    parser.add_argument("--id", default=MODEL, help=f"which model; default {MODEL}")
    parser.add_argument("--version", default=None, help="index version; default is the newest")
    parser.add_argument("--conf", type=float, default=0.5, help="keep instances above this")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP,
                        help="tile overlap as a fraction of the tile, 0 to 0.9")
    parser.add_argument("--batch", type=int, default=1,
                        help="tiles per forward pass; raise it if the card has room")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    add_mask_arguments(parser)
    add_output_arguments(parser)
    args = parser.parse_args(argv)

    from PIL import Image
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    weights, entry = fetch(args.id, args.version)
    # Not a flag: the processor does not resize, so a tile that is not input_size
    # reaches the network at the wrong scale and nothing downstream can tell.
    tile = int(entry["preprocess"]["input_size"][0])
    scale = int(entry["preprocess"].get("native_scale_factor", 1))
    if not 0.0 <= args.overlap < 0.9:
        raise SystemExit(f"--overlap is a fraction of the tile; got {args.overlap}")
    numbers = parse_frames(args.frames, frame_count(args.image))

    device = pick_device(args.device)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(weights).to(device).eval()
    # The processor ships with the weights and already carries do_resize False, the
    # 0.5/0.5 normalisation and the size divisor, so it needs no arguments here.
    processor = AutoImageProcessor.from_pretrained(weights)
    names = {int(key): value for key, value in model.config.id2label.items()}
    classes = resolve_classes(names, args.classes)
    names = dict(zip(sorted(names), classes))  # --classes reaches the instances too
    cleanup = (args.components, args.min_component_area, args.min_component_frac,
               args.min_area)

    out = args.out or args.image.with_suffix(f".{args.id}.png")
    writers = Writers(len(numbers), classes, args.color, out, args.tiff, args.rois)

    for position, number in enumerate(numbers):
        frame = load_image(args.image, number)
        height, width = frame.shape[:2]
        scaled = frame if scale == 1 else np.array(
            Image.fromarray(frame).resize((width * scale, height * scale), Image.BICUBIC))
        # Tile origins have to be whole multiples of the upsample, or a strided mask
        # would land half a pixel out. Snapping the overlap step keeps them so.
        boxes = [(y0 - y0 % scale, x0 - x0 % scale, y1, x1)
                 for y0, x0, y1, x1 in slices(*scaled.shape[:2], tile, args.overlap)]
        # The merge works in frame coordinates, so the seams have to be there too.
        tile_map = [(y0 // scale, x0 // scale, -(-y1 // scale), -(-x1 // scale))
                    for y0, x0, y1, x1 in boxes]
        if position == 0:
            print(describe(scaled.shape, boxes, tile, args.overlap)
                  + f" at {scale}x, on {device}")
            print(f"{len(numbers)} frame(s), classes: {', '.join(classes)}")

        found = predict(model, processor, names, scaled, boxes, scale,
                        args.conf, args.batch, device, cleanup)
        found = merge_masks(found, tile_map, (height, width))
        report(number, found, len(numbers))
        writers.add(position, number, frame, found)

    writers.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
