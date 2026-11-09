"""Render only the masks created in the current, unique demo output directory."""
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def render_report(output, manifest, data_root):
    sections = []
    links = []
    for position, sample in enumerate(manifest["samples"]):
        index = sample["test_index"]
        with Image.open(data_root / sample["image_path"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        with Image.open(data_root / sample["mask_path"]) as image:
            target = np.asarray(image.convert("L")) > 0
        Image.fromarray(rgb).save(output / f"input_test_{index:03d}.png")
        Image.fromarray((target * 255).astype(np.uint8)).save(output / f"ground_truth_test_{index:03d}.png")
        panels = [(rgb, "Input RGB · 256 × 256"), (target, "Ground truth")]
        for key in ["baseline", "bce_dice"]:
            model = manifest["models"][key]
            metrics = model["patch_metrics"][position]
            with Image.open(output / f"{key}_test_{index:03d}.png") as image:
                mask = np.asarray(image) > 0
            panels.append((mask, f"{model['label']}\nPatch IoU {metrics['iou']:.4f} · Dice {metrics['dice']:.4f}"))
        fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), dpi=160)
        for ax, (array, title) in zip(axes, panels):
            ax.imshow(array, interpolation="nearest", **({} if array.ndim == 3 else {"cmap": "gray", "vmin": 0, "vmax": 1}))
            ax.set_title(title, fontsize=13)
            ax.axis("off")
        fig.suptitle(f"Test {index} · {sample['source_image']} · patch {sample['patch_index']:02d}", fontsize=15)
        fig.text(0.5, 0.025, "IoU/Dice describe THIS PATCH, not the whole test set · threshold 0.5 for both models", ha="center", fontsize=12)
        fig.tight_layout(rect=(0, 0.07, 1, 0.93))
        name = f"test_{index:03d}_comparison.png"
        fig.savefig(output / name)
        plt.close(fig)
        title = f"Test {index}" + (" — failure example" if index == 113 else "")
        links.append(f'<a href="#test-{index}">{escape(title)}</a>')
        note = ""
        if index == 113:
            note = ("The roof is truncated at the source-image boundary. Limited context is an untested "
                    "hypothesis; it does not establish why BCE succeeds and the comparison model fails.")
        sections.append(f'<section id="test-{index}"><h2>{escape(title)}</h2>'
                        f'<img src="{name}" alt="Input, ground truth and fresh predictions for test {index}">'
                        f'<p>{escape(note)}</p></section>')
    html = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>P22 — Fresh model inference demo</title>
<style>body{margin:0;background:#f1f3f5;color:#19232e;font:18px system-ui,sans-serif}
header,main{max-width:1450px;margin:auto;padding:22px}header{background:white}
nav{position:sticky;top:0;background:#15384c;padding:15px 22px;z-index:1}
nav a{color:white;margin-right:30px;font-weight:600}section{background:white;padding:18px;margin:20px 0;scroll-margin-top:70px}
img{width:100%;height:auto}h1{font-size:28px;margin-top:0}h2{font-size:23px}
.scope{font-weight:700;color:#15384c}footer{font-size:14px;overflow-wrap:anywhere}</style>
<header><h1>Building segmentation — fresh checkpoint inference</h1>
<p class="scope">Each IoU/Dice value belongs to one 256 × 256 patch, NOT the whole test set.</p>
<p>RGB input → shared ImageNet preprocessing → U-Net + ResNet18 → sigmoid ≥ 0.5.<br>
Ground truth is used for display and scoring; only RGB tensors enter the models.</p>
"""
    html += f'<p>Fresh run: {escape(manifest["started_utc"])} · device: {escape(manifest["device"])}</p></header>'
    html += '<nav>' + "".join(links) + '</nav><main>' + "".join(sections)
    html += '<footer>Both checkpoints were loaded and forward passes completed during this run. '
    html += 'Run details and checkpoint SHA-256: <a href="run.json">run.json</a>.</footer></main></html>'
    path = output / "index.html"
    path.write_text(html, encoding="utf-8")
    return path
