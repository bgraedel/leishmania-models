#!/usr/bin/env python
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "torch",
#     # --guide runs the detector to place the tiles, and that is an ultralytics
#     # checkpoint. Only --guide needs it, but a dependency that half the flags
#     # need is still a dependency.
#     "ultralytics",
#     # transformers 5 builds its image processors on torchvision, and raises on the
#     # first AutoImageProcessor without it. torch alone is not enough.
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

**By default the frame goes through whole, into the network at the input size the
checkpoint records** -- one plain pass. The frame is squared off to its long side and
the processor resizes that to the input, so nothing is stretched and a frame larger
than the input arrives scaled down; the run says by how much.

`--tile N` opts into tiling: an N px crop is cut at the frame's own resolution and
resized up to the same input, so a cell arrives at the scale the model was trained on
however large the frame is. The run says which N does that -- `input_size /
native_scale_factor`, 512 for the 1024 model -- and warns for any other, the default
whole-frame pass included. Tiles
overlap, and each instance is kept by the one tile whose core holds its centroid;
`--stitch merge` unions pieces across neighbouring tiles instead. `--guide` puts the
tiles where the detector found cells rather than on a grid, so that each cell is
inside one tile whole and nothing has to be rejoined; see guide.py.

  leishmania-m2f-1024         Swin-S, 512 px crops resized 2x to a 1024 px input
  leishmania-m2f-512-amodal   Swin-B, 512 px crops at native resolution, and amodal:
                              it completes a cell through whatever occludes it, so
                              masks overlap where two cells cross

The checkpoints ship `do_resize` off, which is what made the tile size a constant
here before. The `size` beside it is already the input the network wants, so turning
it on is all it takes for a crop of any size to reach that input at the right scale.

**This one wants a card with room to spare, and the default pass wants the most.**
transformers' own post_process_instance_segmentation resamples every one of the
model's ~100 query masks to `target_sizes` before throwing the low-scoring ones away,
so the cost is set by the area asked back, not by what survives. A whole 1392x1040
frame squares off to 1392, which is ~770 MB of float32 for that one step; a 512 px tile
is 100 MB, and a tiled run pays it one tile at a time. Past what the card has, the driver spills to system
memory over PCIe, which is why --batch defaults to 1 in this script alone. `--tile
512` is the first thing to reach for on a small card, and the 512 model over the 1024
one after that.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fetch import fetch, frame_sizes, load_image
from guide import add_guide_arguments, guide_from
from outputs import (Found, Writers, add_mask_arguments, add_output_arguments,
                     default_output, frame_count, instance_mask, parse_frames,
                     progress, report, resolve_classes, say)
from tiling import (DEFAULT_OVERLAP, add_stitch_argument, batches, describe,
                    resolve_stitch, slices, stitch, warn_if_longer_than_overlap)

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


def predict(model, processor, names, frame, boxes, conf, batch, device, cleanup,
            square: int):
    """Every instance in one frame, each mask in the frame's own coordinates.

    `square` is the size every crop is padded out to before it goes in: the tile when
    tiling, the frame's long side when not. It has to be square because the processor
    resizes to the network's fixed square input, and a 300x700 crop sent into that
    would come out with every cell stretched.

    The padded size is also what `target_sizes` asks the masks back at, so they arrive
    at the resolution they are stored at rather than at the network's, and each is cut
    down to the real crop before being placed so the padding contributes nothing.
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
        # resampled to it before the low-scoring ones are dropped, so asking for the
        # smaller of the two is most of what this costs.
        # return_binary_maps keeps one mask per instance. The label-map form collapses
        # them into a single image, and amodal masks that overlap cannot survive that.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path,
                        help="a .tif stack, any single image, or a FOLDER of "
                             "images taken in name order as one sequence")
    parser.add_argument("--id", default=MODEL, help=f"which model; default {MODEL}")
    parser.add_argument("--version", default=None, help="index version; default is the newest")
    parser.add_argument("--conf", type=float, default=0.5, help="keep instances above this")
    parser.add_argument("--tile", type=int, default=0,
                        help="crop size in frame px. The default is 0: the whole frame "
                             "in one pass, squared off and resized to the network's "
                             "input. --tile N cuts N px crops and resizes each of those "
                             "instead, which is what puts cells at the scale the model "
                             "was trained on; the run says which N does that")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP,
                        help="tile overlap as a fraction of the tile, 0 to 0.9")
    parser.add_argument("--batch", type=int, default=1,
                        help="tiles per forward pass; raise it if the card has room")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    add_stitch_argument(parser)
    add_guide_arguments(parser)
    add_mask_arguments(parser)
    add_output_arguments(parser)
    args = parser.parse_args(argv)

    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    from transformers.image_utils import PILImageResampling

    weights, entry = fetch(args.id, args.version)
    tile = int(args.tile)
    if not 0.0 <= args.overlap < 0.9:
        raise SystemExit(f"--overlap is a fraction of the tile; got {args.overlap}")
    numbers = parse_frames(args.frames, frame_count(args.image))

    device = pick_device(args.device)
    how = resolve_stitch(args.stitch, args.guide)
    guide = guide_from(args, device)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(weights).to(device).eval()
    # The processor ships with the weights and already carries the 0.5/0.5
    # normalisation, the size divisor, and a `size` that is the network's input.
    # do_resize is off in it, so it is the one thing that has to be turned on.
    #
    # resample is pinned to bicubic rather than left at the processor's bilinear,
    # because bicubic is what this model's training frames were upsampled with and the
    # resize has to present cells the way it knows them.
    processor = AutoImageProcessor.from_pretrained(
        weights, do_resize=True, resample=PILImageResampling.BICUBIC)
    # The input size comes from the PROCESSOR, which is the thing that will actually do
    # the resizing, and not from the index, which only describes it. They agree today;
    # reading the index instead would report a scale nobody was shown the moment they
    # stopped agreeing, and `do_resize=True` is what made that field load-bearing.
    shipped = dict(getattr(processor, "size", None) or {})
    size = int(shipped.get("height") or shipped.get("shortest_edge")
               or entry["preprocess"]["input_size"][0])
    recorded = int(entry["preprocess"]["input_size"][0])
    if size != recorded:
        say(f"warning: the index records a {recorded} px input for {args.id} and the "
            f"checkpoint's own processor says {size}; going with the checkpoint")
    # The crop that fills that input at the scale the model was trained on.
    native = max(1, size // int(entry["preprocess"].get("native_scale_factor", 1)))
    # Settled here rather than beside `--tile`, because it is `native` that says what
    # size to place, and `native` is not known until the processor has been read.
    if args.guide and not tile:
        tile = native
        say(f"--guide places tiles, so it turns tiling on: {tile} px crops, the size "
            f"that reaches a {size} px input at this model's own scale")
    names = {int(key): value for key, value in model.config.id2label.items()}
    classes = resolve_classes(names, args.classes)
    names = dict(zip(sorted(names), classes))  # --classes reaches the instances too
    cleanup = (args.components, args.min_component_area, args.min_component_frac,
               args.min_area)

    out = args.out or default_output(args.image, args.id)
    sizes = frame_sizes(args.image)
    writers = Writers(len(numbers), classes, args.color, out, args.tiff,
                      args.rois, [sizes[n] for n in numbers])

    for position, number in progress(list(enumerate(numbers)), "frames"):
        frame = load_image(args.image, number)
        height, width = frame.shape[:2]
        # Crops and seams are both in frame coordinates, so the stitching takes these
        # unchanged -- there is no upsampled frame for them to be converted out of.
        # Untiled, the whole frame is the crop. Either way it is squared off before
        # the resize, because that resize is to the network's fixed square input and
        # would otherwise stretch every cell in a non-square crop out of shape. Capped
        # at the frame: a --tile larger than the image is a square of mostly black,
        # and the network spends its input on the padding.
        crop = min(tile, max(height, width)) if tile else max(height, width)
        boxes = slices(height, width, tile, args.overlap)
        planned, note, plan = (guide.plan(frame, tile, args.overlap) if guide
                               else (None, None, None))
        boxes = planned or boxes
        if position == 0:
            say((note or describe(frame.shape, boxes, crop, args.overlap))
                  + f" -> a {size} px input ({size / crop:.3g}x), on {device}")
            if crop != native:
                say(f"warning: this model was trained on {native} px crops; {crop} px "
                      f"presents cells at {native / crop:.3g}x the scale it knows"
                    + ("" if tile else f" -- --tile {native} presents them as trained"))
            say(f"{len(numbers)} frame(s), classes: {', '.join(classes)}")

        found = predict(model, processor, names, frame, boxes,
                        args.conf, args.batch, device, cleanup, square=crop)
        if position == 0 and not planned:
            # Before stitching, so the lengths are the ones the tiles saw.
            warn_if_longer_than_overlap(found, tile, args.overlap)
        found = stitch(found, boxes, (height, width), how, plan=plan)
        # Only where there IS a plan: a frame the detector found nothing in has fallen
        # back to the grid, and there is no register there to be strict about.
        loose = 0
        if plan and args.guide_strict:
            loose = sum(1 for item in found if not item.cell)
            found = [item for item in found if item.cell]
        report(number, found, len(numbers), loose)
        writers.add(position, number, frame, found)

    writers.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
