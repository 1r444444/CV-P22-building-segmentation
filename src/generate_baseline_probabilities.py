from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_and_preprocess


DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

BATCH_SIZE = 8

CHECKPOINT_PATH = PROJECT_ROOT / "results" / "baseline_unet_resnet18.pt"
OUTPUT_DIR = PROJECT_ROOT / "results" / "baseline_probabilities"


class BuildingDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        image_path = PROJECT_ROOT / row["image_path"]
        mask_path = PROJECT_ROOT / row["mask_path"]

        image, _ = load_and_preprocess(image_path, mask_path)

        return image


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    print("Baseline checkpoint loaded.")

    for split in ["val", "test"]:
        df = pd.read_csv(
            PROJECT_ROOT / "data" / "splits" / f"{split}.csv"
        )

        dataset = BuildingDataset(df)

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        all_probabilities = []

        with torch.inference_mode():
            for images in loader:
                images = images.to(DEVICE)

                logits = model(images)
                probabilities = torch.sigmoid(logits)

                all_probabilities.append(
                    probabilities.cpu().numpy()
                )

        probabilities = np.concatenate(
            all_probabilities,
            axis=0,
        )

        output_path = OUTPUT_DIR / f"{split}.npz"

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