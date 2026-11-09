from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_and_preprocess


BATCH_SIZE = 8

CHECKPOINT_PATH = PROJECT_ROOT / "results" / "baseline_unet_resnet18.pt"
OUTPUT_DIR = PROJECT_ROOT / "results" / "baseline_probabilities"


class BuildingDataset(Dataset):
    def __init__(self, dataframe, data_root=PROJECT_ROOT):
        self.dataframe = dataframe.reset_index(drop=True)
        self.data_root = data_root

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        image_path = self.data_root / row["image_path"]
        mask_path = self.data_root / row["mask_path"]

        image, _ = load_and_preprocess(image_path, mask_path)

        return image


def main():
    parser = argparse.ArgumentParser(description="Export U-Net + ResNet18 probabilities without training.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data/splits")
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    parser.add_argument("--limit", type=int, help="Smoke test only; exports the first N rows, not full-split metrics.")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"Missing checkpoint: {args.checkpoint}. See README.md for artifact handoff.")
    if any((args.output_dir / f"{split}.npz").exists() for split in args.splits):
        parser.error("Output already exists; choose a new --output-dir to preserve previous results.")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_name)

    print("Device:", device)

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    state_dict = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print("Checkpoint loaded:", args.checkpoint)

    for split in args.splits:
        df = pd.read_csv(
            args.splits_dir / f"{split}.csv"
        )
        if args.limit is not None:
            df = df.head(args.limit)

        dataset = BuildingDataset(df, args.data_root)

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        all_probabilities = []

        with torch.inference_mode():
            for images in loader:
                images = images.to(device)

                logits = model(images)
                probabilities = torch.sigmoid(logits)

                all_probabilities.append(
                    probabilities.cpu().numpy()
                )

        probabilities = np.concatenate(
            all_probabilities,
            axis=0,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / f"{split}.npz"

        np.savez_compressed(
            output_path,
            probabilities=probabilities,
            image_paths=df["image_path"].to_numpy(dtype=str),
            mask_paths=df["mask_path"].to_numpy(dtype=str),
        )

        print(
            f"{split.upper()}: "
            f"{probabilities.shape}, "
            f"saved to {output_path}"
        )


if __name__ == "__main__":
    main()
