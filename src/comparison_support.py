"""Read-only checks and provenance for Person 2's comparison notebook."""

import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

from PIL import Image


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_sha256(path):
    """Notebook provenance must not depend on autosaved execution outputs."""
    path = Path(path)
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        sources = [(cell["cell_type"], "".join(cell["source"])) for cell in notebook["cells"]]
        content = json.dumps(sources, ensure_ascii=False).encode("utf-8")
    else:
        content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def validate_comparison_data(root, frames):
    """Require the complete baseline splits and existing 256px image-mask pairs."""
    expected = {"train": (4932, 137), "val": (144, 4), "test": (360, 10)}
    sources = {}
    summary = {}
    for split, (patch_count, source_count) in expected.items():
        frame = frames[split]
        if len(frame) != patch_count or frame.source_image.nunique() != source_count:
            raise ValueError(f"{split}: expected {patch_count} patches / {source_count} sources")
        if set(frame["split"]) != {split}:
            raise ValueError(f"{split}: incorrect split labels")
        if frame.image_path.duplicated().any() or frame.mask_path.duplicated().any():
            raise ValueError(f"{split}: duplicate paths")
        for source, group in frame.groupby("source_image"):
            if len(group) != 36 or set(group.patch_idx) != set(range(36)):
                raise ValueError(f"{split}/{source}: expected all 36 baseline patches")
        sources[split] = set(frame.source_image)
        for row in frame.itertuples(index=False):
            for column in ("image_path", "mask_path"):
                relative = Path(getattr(row, column))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Expected repository-relative path: {relative}")
                path = root / relative
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing {path}. Run the existing 02_prepare_subset.ipynb first."
                    )
                with Image.open(path) as image:
                    if image.size != (256, 256):
                        raise ValueError(f"Unexpected patch dimensions: {path}: {image.size}")
                    image.verify()
        summary[split] = {
            "patches": len(frame), "original_images": len(sources[split]),
            "split_csv_sha256": sha256(root / "data" / "splits" / f"{split}.csv"),
        }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if sources[left] & sources[right]:
            raise ValueError(f"Source-image leakage between {left} and {right}")
    return summary


def runtime_environment(root):
    import torch

    packages = ["torch", "torchvision", "segmentation-models-pytorch", "numpy",
                "pandas", "Pillow", "matplotlib", "tqdm"]
    result = {"packages": {name: version(name) for name in packages}}
    import platform
    result.update(python=platform.python_version(), platform=platform.platform(),
                  cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    for key, args in (("git_commit", ["rev-parse", "HEAD"]),
                      ("git_status", ["status", "--porcelain"])):
        proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        result[key] = proc.stdout.strip() if proc.returncode == 0 else "unavailable"
    result["source_sha256"] = {
        name: source_sha256(root / name) for name in (
            "src/losses.py", "src/comparison_support.py", "src/preprocessing.py",
            "notebooks/04_baseline.ipynb", "notebooks/05_comparison_bce_dice.ipynb",
        )
    }
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
