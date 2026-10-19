# Person 2 — Comparison Experiment

Owner: Elena Tuturina. The comparison is the **same U-Net + ResNet18 with BCE +
soft Dice loss**, as specified in the project plan and first-working-model report.
This contribution implements, trains, and evaluates that condition and saves its
artifacts. It includes a brief initial comparison required by the milestone.
Person 3 owns threshold sensitivity, the final evaluation table and failure
analysis; Person 4 owns the combined technical summary, integration and demo.

## Experiment

Question: does adding soft Dice to BCE improve building overlap on the same data?
BCE optimizes pixel classification, whereas soft Dice directly rewards overlap
with the foreground. This may help with foreground/background imbalance; better
IoU/Dice must be established experimentally.

The new objective is:

    loss = mean_pixel_BCEWithLogits(logits, target)
           + mean_patch(1 - (2*sum(p*target) + 1)/(sum(p) + sum(target) + 1))
    p = sigmoid(logits)

Both coefficients are 1. The soft Dice term is computed independently for every
patch, including background-only patches, then averaged across the batch.
Smoothing is 1 in numerator and denominator. There is no hard threshold in the
loss. No loss-weight search or threshold sweep is performed.

| Setting | Baseline and comparison |
| --- | --- |
| Data | Massachusetts Buildings; all 151 original images |
| Split | Original 137/4/10 source images; 4932/144/360 patches |
| Patch preparation | Zero-pad 1500 to 1536; non-overlapping 256 × 256; keep all patches |
| Preprocessing | Existing `src/preprocessing.py`: RGB [0,1], ImageNet mean/std, binary mask |
| Model | `smp.Unet`, ResNet18, ImageNet encoder, 3 inputs, 1 output logit |
| Optimizer | Adam, lr 0.001, library defaults |
| Training | 10 epochs, batch size 8, seed 42, no augmentation, full precision |
| DataLoader | 0 workers; shuffled train with generator seed 42; unshuffled val/test |
| Checkpoint | Lowest validation loss under each condition's own objective |
| Metrics | Global/micro pixel IoU and Dice; sigmoid ≥ 0.5; epsilon 1e-7 |

The comparison notebook derives its Dataset, training/evaluation loops and metric
functions from `04_baseline.ipynb` at commit `2c418ba`. It reuses preprocessing
directly. It retains both diagnostic train-loader reads and the training-mode
shape-check forward from the baseline, preserving their effect on sampler and
BatchNorm state. The complete existing split CSVs are read without modification.

The saved baseline is a historical T4 run; the local comparison uses the available
GPU. Matching seeds and direct dependencies do not guarantee bitwise equality
across hardware or unrecorded transitive versions. The notebook records the
comparison environment and source/split/checkpoint hashes. Do not describe this
single-run comparison as proof of statistical significance.

## Local environment

Use Python 3.12 and a working NVIDIA driver. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-comparison.txt
.venv/bin/python -m ipykernel install --prefix .venv --name p22-comparison --display-name "P22 Comparison (.venv)"
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
nvidia-smi
.venv/bin/python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

The baseline pins `torch==2.11.0+cu130` and `torchvision==0.26.0+cu130`.
`requirements-comparison.txt` adds the official CUDA 13.0 wheel index rather than
replacing those versions. CPU checks of the loss are not training results.

## Data prerequisites

The provided Kaggle URL is a notebook. Its public metadata identifies its input
as [`balraj98/massachusetts-buildings-dataset`](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset),
version 2. The archive is approximately 1.6 GB. Download/extract locally if absent:

```bash
curl -L --fail --retry 3 --output data/massachusetts-buildings-v2.zip \
  'https://www.kaggle.com/api/v1/datasets/download/balraj98/massachusetts-buildings-dataset?datasetVersionNumber=2'
.venv/bin/python -m zipfile -e data/massachusetts-buildings-v2.zip data/raw
for split in train train_labels val val_labels test test_labels; do
  mv "data/raw/png/$split" "data/raw/$split"
done
```

The archive contains both PNG and TIFF trees. The existing preparation notebook
expects the PNG split folders directly under `data/raw/`; the move above adapts
the layout without changing any image contents. Perform it only once after a
fresh extraction. The TIFF copy is unused.

Verified archive SHA-256:
`bd271292c60033b6bcbfbe12ebbf30dc17dcd1cefc4060c7542fb2b73374875e`.

Expected layout:

```text
data/raw/train/          137 PNG images
data/raw/train_labels/   137 PNG masks
data/raw/val/              4 PNG images
data/raw/val_labels/       4 PNG masks
data/raw/test/            10 PNG images
data/raw/test_labels/     10 PNG masks
```

Reuse the existing `02_prepare_subset.ipynb` to populate `data/processed/` if
needed. No new subset, filtering, resizing or split is introduced. Its generated
split CSVs should remain identical to the committed baseline CSVs.
The comparison notebook checks file existence, readability, 256 × 256 dimensions,
expected counts, patch uniqueness and source-image separation before training.

## Run and save outputs

```bash
.venv/bin/python -m jupyterlab --no-browser
```

Open the local URL shown by Jupyter in your browser. Select the project's Python
kernel (its executable must be `.venv/bin/python`). If processed patches are
missing, open `notebooks/02_prepare_subset.ipynb`, run all cells, and save.
Then open `notebooks/05_comparison_bce_dice.ipynb`, choose **Restart Kernel and Run
All Cells**, wait for completion, and save with outputs. The first model creation
may download the ImageNet encoder weights; internet is required for that cache miss.

Alternatively, execute and save from the repository root:

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebooks/05_comparison_bce_dice.ipynb
```

Do not rerun the training cell in isolation: this continues training from the
current weights and RNG state. Restart and run the whole notebook for a fresh
experiment. A full rerun replaces the comparison artifacts; copy an earlier run
elsewhere first if it must be retained. Baseline artifacts are not overwritten.
The test set is evaluated only after checkpoint selection, using threshold 0.5.

## Produced artifacts

| Artifact | Purpose |
| --- | --- |
| `notebooks/05_comparison_bce_dice.ipynb` | Saved execution and displayed results |
| `results/metrics/comparison_training_history.csv` | Per-epoch train/validation loss, IoU, Dice and time |
| `results/metrics/comparison_metrics.csv` | Best epoch, selected validation and final test metrics |
| `results/metrics/comparison_initial_comparison.csv` | Brief baseline/comparison metrics and deltas in percentage points |
| `results/metrics/comparison_selected_predictions.csv` | Exact identities of eight selected test masks |
| `results/metrics/comparison_run.json` | Settings, source/split/checkpoint hashes, environment and timing |
| `results/metrics/comparison_environment.txt` | Exact `pip freeze` from the actual run |
| `results/predictions/comparison_prediction_00.png` … `_07.png` | Same first eight test CSV rows used by baseline |
| `figures/comparison_examples.png` | Basic visual check of the first four examples |
| `results/comparison_unet_resnet18.pt` | Local best state dict; ignored by Git |
| `results/comparison_probabilities/{val,test}.npz` | Local float32 probabilities for every evaluation patch; ignored by Git |

Probability archives contain `probabilities` (N,1,256,256), `image_paths`,
`mask_paths`, `source_images` and `patch_indices` in CSV order. Person 3 can
recompute metrics, adjust thresholds and select failures from these outputs or
the checkpoint without retraining. Large artifacts need a separate handoff;
they are not present in a Git clone. Download the checkpoint and caches from the
[comparison release](https://github.com/1r444444/CV-P22-building-segmentation/releases/tag/comparison-bce-dice-2026-10-19):

```bash
gh release download comparison-bce-dice-2026-10-19 \
  --repo 1r444444/CV-P22-building-segmentation \
  --pattern comparison_unet_resnet18.pt --dir results
gh release download comparison-bce-dice-2026-10-19 \
  --repo 1r444444/CV-P22-building-segmentation \
  --pattern val.npz --pattern test.npz --dir results/comparison_probabilities
```

The release also includes `comparison_artifact_checksums.txt`. The checkpoint's
SHA-256 is recorded independently in `results/metrics/comparison_run.json`.

To load the recorded checkpoint without retraining or downloading ImageNet weights:

```python
from pathlib import Path
import torch
import segmentation_models_pytorch as smp

root = Path.cwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)
state = torch.load(root / "results/comparison_unet_resnet18.pt", map_location=device, weights_only=True)
model.load_state_dict(state)
model = model.to(device).eval()
```

Use the existing preprocessing before inference, then apply sigmoid to obtain
probabilities. For analysis of this recorded run, the supplied probability caches
avoid another inference pass and do not require a GPU.

Training time includes epoch validation and checkpoint writes. Test evaluation
time includes data loading, forward passes, loss and metrics; it is not isolated
GPU latency. Export/compression time is recorded separately. Timings come from
the actual run, with CUDA synchronization at boundaries.

## Recorded run — October 19, 2026

The notebook completed all 10 epochs on an NVIDIA GeForce RTX 5070. The selected
checkpoint is epoch 10, with validation combined loss 0.4535. Training including
epoch validation and checkpoint writes took 312.19 seconds; test evaluation over
360 patches took 1.49 seconds, including data loading, forward passes, loss and
metrics. Probability export and compression took another 4.36 seconds.

| Split | Metric | BCE baseline | BCE + Dice | Change (percentage points) |
| --- | --- | --- | --- | --- |
| Validation | IoU | 0.6338 | 0.6160 | -1.78 |
| Validation | Dice | 0.7759 | 0.7624 | -1.35 |
| Test | IoU | 0.6898 | 0.6681 | -2.17 |
| Test | Dice | 0.8164 | 0.8010 | -1.54 |

The expected improvement was not observed for this fixed configuration. These
single-run results do not establish that Dice loss is generally inferior. No
loss weights or thresholds were tuned after observing test performance. BCE-only
and combined loss values measure different objectives and are not compared.

All validation/test IoU and Dice values were independently recomputed from the
saved float32 probabilities and original masks and matched the recorded metrics.
A fresh load of the checkpoint reproduced all 504 validation/test probability maps
exactly, as well as the recorded combined losses and metrics. Every processed RGB
image and mask matched its expected padded raw crop across all 5,436 patches.
The loss passed an independent NumPy calculation and finite-difference gradient
check. The source, split, baseline-reference and checkpoint hashes were verified,
as were all eight selected masks, the complete notebook execution and four tests.

## Milestone report

The English report is [SecondApproachSubmission_P22.pdf](report/SecondApproachSubmission_P22.pdf),
with editable [LaTeX source](report/SecondApproachSubmission_P22.tex). It follows
the previous reports' centered project/team/date title, numbered sections,
compact tables and Latin Modern fonts on A4 pages. It covers the new approach,
motivation, implementation, actual initial results, limitations and contribution.
It is separate from the team's final technical summary.

To rebuild the report with Tectonic from the repository root:

```bash
tectonic --outdir report report/SecondApproachSubmission_P22.tex
```

The development branch is `experiment/bce-dice-comparison`; the team's integration
branch is `main`. A ready-to-forward Russian handoff is in
[COMPARISON_HANDOFF_RU.md](COMPARISON_HANDOFF_RU.md).
