"""Fresh checkpoint inference on selected test patches; no prediction-cache inputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import os
import shlex
import webbrowser

import numpy as np
import pandas as pd
from PIL import Image
import segmentation_models_pytorch as smp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import load_and_preprocess

THRESHOLD = 0.5


def first_available(*paths):
    return next((p for p in paths if p.exists()), paths[0])


def patch_metrics(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Expected matching H x W binary masks")
    prediction, target = prediction.astype(bool), target.astype(bool)
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return {"iou": intersection / (union + 1e-7),
            "dice": 2 * intersection / (int(prediction.sum() + target.sum()) + 1e-7)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 113])
    parser.add_argument("--data-root", type=Path, default=first_available(
        ROOT if (ROOT / "data/processed/test").is_dir() else ROOT / "verification/handoff-evaluation-2026-11-09",
        ROOT))
    parser.add_argument("--baseline-checkpoint", type=Path, default=first_available(
        ROOT / "results/baseline_unet_resnet18.pt",
        ROOT / "verification/baseline-drive-evaluation-2026-11-09/downloads/baseline_unet_resnet18.pt"))
    parser.add_argument("--comparison-checkpoint", type=Path, default=first_available(
        ROOT / "results/comparison_unet_resnet18.pt",
        ROOT / "verification/2026-11-09/downloads/comparison_unet_resnet18.pt"))
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--open", action="store_true", help="Open the new local report in the default browser after inference.")
    args = parser.parse_args(argv)
    started = datetime.now(timezone.utc)
    output = (args.output_dir or ROOT / "verification/demo" / started.strftime("run-%Y%m%dT%H%M%S-%fZ")).resolve()
    if output.exists():
        parser.error("Output directory already exists; choose a new directory. Nothing was overwritten.")
    frame = pd.read_csv(ROOT / "data/splits/test.csv")
    if len(set(args.indices)) != len(args.indices) or any(i < 0 or i >= len(frame) for i in args.indices):
        parser.error(f"Choose distinct test indices between 0 and {len(frame) - 1}.")
    for checkpoint in [args.baseline_checkpoint, args.comparison_checkpoint]:
        if not checkpoint.is_file():
            parser.error(f"Missing checkpoint: {checkpoint}. Supply the corresponding --*-checkpoint path.")
    selected = frame.iloc[args.indices].reset_index(drop=True)
    samples, tensors, targets = [], [], []
    for index, row in zip(args.indices, selected.itertuples(index=False)):
        for relative in [row.image_path, row.mask_path]:
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                parser.error(f"Expected repository-relative data path: {relative}")
            if not (args.data_root / relative).is_file():
                parser.error(f"Missing {args.data_root / relative}; supply --data-root containing data/processed/.")
        image, mask = load_and_preprocess(args.data_root / row.image_path, args.data_root / row.mask_path)
        if image.shape != (3, 256, 256) or mask.shape != (1, 256, 256):
            parser.error(f"Test {index}: expected a 256 x 256 image/mask pair")
        tensors.append(image)
        targets.append(mask[0].numpy().astype(bool))
        samples.append({"test_index": index, "source_image": row.source_image, "patch_index": int(row.patch_idx),
                        "image_path": row.image_path, "mask_path": row.mask_path})
    output.mkdir(parents=True, exist_ok=False)
    batch = torch.stack(tensors).to(args.device)
    manifest = {"started_utc": started.isoformat(), "device": args.device, "threshold": THRESHOLD,
                "metric_scope": "Individual 256 x 256 patch, NOT the whole test set",
                "metric_formula": "I/(U+1e-7), 2I/(P+T+1e-7)", "samples": samples, "models": {}}
    for key, label, checkpoint in [("baseline", "Baseline (BCE)", args.baseline_checkpoint),
                                    ("bce_dice", "BCE + soft Dice", args.comparison_checkpoint)]:
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.to(args.device).eval()
        print(f"Loaded {label}: {checkpoint}\nSHA-256: {digest}", flush=True)
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(batch)).cpu().numpy()
        if probabilities.shape != (len(samples), 1, 256, 256) or not np.isfinite(probabilities).all():
            raise ValueError("Invalid fresh model output")
        print(f"Fresh forward: {label}, batch {tuple(batch.shape)}, threshold {THRESHOLD}", flush=True)
        np.savez_compressed(output / f"{key}_probabilities.npz", probabilities=probabilities,
                            test_indices=np.array(args.indices), image_paths=selected.image_path.to_numpy(dtype=str),
                            mask_paths=selected.mask_path.to_numpy(dtype=str))
        rows = []
        for sample, probability, target in zip(samples, probabilities[:, 0], targets):
            prediction = probability >= THRESHOLD
            metrics = patch_metrics(prediction, target)
            rows.append({"test_index": sample["test_index"], **metrics})
            Image.fromarray((prediction * 255).astype(np.uint8)).save(output / f"{key}_test_{sample['test_index']:03d}.png")
            print(f"Test {sample['test_index']}: PATCH IoU={metrics['iou']:.6f}, PATCH Dice={metrics['dice']:.6f}", flush=True)
        manifest["models"][key] = {"label": label, "checkpoint": str(checkpoint.resolve()), "sha256": digest,
                                   "fresh_forward_calls": 1, "patch_metrics": rows}
        del model
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    from scripts.demo_view import render_report
    page = render_report(output, manifest, args.data_root)
    print("Saved fresh inference:", output, flush=True)
    print("Open on Mac: open " + shlex.quote(str(page)), flush=True)
    if args.open:
        webbrowser.open(page.as_uri())
    return output


if __name__ == "__main__":
    main()
