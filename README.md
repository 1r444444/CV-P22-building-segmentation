# P22 - Building Segmentation in Satellite Images

Team project for **Computer Vision 2026**.

## Goal

Segment building contours in aerial or satellite images using a subset of a
public dataset. Evaluate segmentation quality with IoU and Dice, and examine
errors associated with shadows, small buildings, and dense urban regions.

Research question: **does adding soft Dice to BCE improve building IoU/Dice?**
The implemented experiment uses Massachusetts Buildings and U-Net + ResNet18.
The changed factor is the loss; the architecture, source-level split and nominal
training settings are shared. See [COMPARISON_RUN.md](COMPARISON_RUN.md).

Integration status (2026-11-09): installation, imports, loss tests and an eight-patch
CPU inference smoke test were checked in a fresh Python 3.11 environment on macOS.
Full-split metric reproduction and retraining have **not** been performed in this
audit. Recorded notebook outputs and CSVs remain historical experiment results.

## Project Requirements

Based on the official Computer Vision 2026 project guidelines, the project must:

- use a public dataset or a small custom dataset;
- define a clear research or engineering question;
- establish a baseline and compare it with a second condition or model;
- change one major factor at a time where possible;
- report at least two appropriate quantitative metrics;
- include a concise ablation or sensitivity experiment;
- inspect and explain at least three failure cases;
- document the train/validation/test split and prevent data leakage;
- provide reproducible code or notebooks with clear run instructions;
- document the environment and approximate training/inference time;
- include a two-page technical summary;
- demonstrate the task, baseline, main comparison, and one failure case, with
  no more than three presentation slides in addition to the live demo;
- prepare each member for an individual Q&A about the project;
- state each team member's contribution.

The official guidelines mention both a five-minute demo and a 10-minute demo
format. The final duration is TBD pending clarification from the instructors.

For this project, the primary quantitative metrics are **IoU** and **Dice**.

## Repository Structure

```text
CV-P22-building-segmentation/
|-- README.md
|-- REPRODUCIBILITY_CHECKLIST.md
|-- requirements.txt
|-- .gitignore
|-- data/                 # Data acquisition and split instructions; no datasets
|-- notebooks/            # Reproducible experiments or analysis notebooks
|-- src/                  # Project source code
|-- results/
|   |-- metrics/          # Final lightweight metric files
|   `-- predictions/      # Selected lightweight prediction artifacts
|-- figures/              # Final lightweight figures
`-- report/               # Technical summary template and final report
```

The internal structure of `src/` will be introduced only when the implemented
code makes those boundaries useful.

## Team Responsibilities

### Person 1 - Dataset and Baseline

- Select and prepare the public dataset and subset.
- Create the train/validation/test split and preprocessing pipeline.
- Implement the baseline and produce baseline predictions and metrics.

### Person 2 - Comparison Model or Condition

- Define the second model or condition.
- Run a controlled comparison with the baseline.
- Save predictions and metrics.

### Person 3 - Evaluation and Failure Analysis

- Calculate final IoU and Dice scores.
- Prepare comparison tables and figures.
- Run the sensitivity or ablation experiment.
- Analyze and explain at least three failure cases.

### Person 4 - Reproducibility, Integration, and Demo

- Maintain the repository structure, README, and environment description.
- Integrate team outputs and verify clean-environment reproducibility.
- Check paths and required files.
- Organize final metrics, figures, technical summary, and demo materials.
- Document the runtime environment and approximate training/inference time.

Team members named in the existing milestone report: Renata Salikhzyanova,
Kristina Soloveva, Elena Tuturina, Irina Bychkova. Elena's comparison contribution
is documented in `COMPARISON_RUN.md` and the milestone PDF. Irina owns integration,
reproducibility and demo. Members 1 and 3 should confirm their name-to-role mapping
and final contribution statements; the responsibility list above is planned work,
not evidence that every deliverable is complete.

## Dataset

Massachusetts Buildings, Kaggle repack version 2, all 151 original images.
The original source-level partition is retained: 137/4/10 images become
4932/144/360 non-overlapping patches. Source filenames do not overlap between the
committed splits; geographical or content duplication has not been independently
excluded. Preparation pads 1500 to 1536 with zeros, then crops 256 × 256 patches.
Padding and background-only patches are included in the recorded evaluation.
Inputs use ImageNet normalization; masks use grayscale `> 0`, without interpolation.

See [data/README.md](data/README.md) for acquisition, layout, counts and preparation.
The full archive is about 1.6 GB; allow additional space for extraction and patches.

## Setup

Run commands from the repository root. Do not use the system Python 3.14 with the
pinned NumPy/pandas versions. The recorded comparison used Python 3.12.3 on Linux;
the local CPU audit used Python 3.11.15 on macOS arm64.

CPU/macOS inference, evaluation and notebooks:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-notebooks.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m ipykernel install --prefix .venv --name p22-local --display-name "P22 Local"
.venv/bin/python -m jupyterlab --no-browser
```

`requirements.txt` alone installs runtime libraries without Jupyter. The notebook
file adds Jupyter tools. This CPU/macOS environment uses the same direct library
versions with platform-specific PyTorch builds; it is not the original CUDA runtime.
Transitive dependencies are not fully locked. The historical comparison freeze is
`results/metrics/comparison_environment.txt`; the local audit freeze is under
`verification/2026-11-09/environment.txt` (local, ignored).

For the recorded CUDA training configuration, use **Python 3.12 and Linux/NVIDIA**
and follow [COMPARISON_RUN.md](COMPARISON_RUN.md#local-environment).
`requirements-comparison.txt` retains `torch==2.11.0+cu130` and
`torchvision==0.26.0+cu130` with the CUDA 13.0 index. This profile is not for macOS.
A new installation/training run on that CUDA profile was not verified by this audit.

## Entry points and order

| File | Purpose and required inputs |
| --- | --- |
| `notebooks/01_dataset_explore.ipynb` | Raw PNG pair/count/value checks and optional metadata CSV inspection |
| `notebooks/02_prepare_subset.ipynb` | Raw PNGs → padded patches and split CSVs; writes `data/processed/` and `data/splits/` |
| `notebooks/03_preprocessing.ipynb` | Demonstrates preprocessing on a train patch; not a dataset generator |
| `src/preprocessing.py` | Shared runtime preprocessing used by both models |
| `notebooks/04_baseline.ipynb` | Full BCE training, checkpoint selection, test evaluation and eight prediction PNGs |
| `notebooks/05_comparison_bce_dice.ipynb` | Full BCE + Dice training/evaluation, figures and probability export; explicitly requires CUDA |
| `src/losses.py` | BCE + mean per-patch soft Dice, coefficients 1/1, smoothing 1 |
| `src/generate_baseline_probabilities.py` | Inference-only export from either compatible checkpoint; no ImageNet download |
| `notebooks/06_evaluation.ipynb` | Both models' val/test caches + processed data → threshold CSVs and failure/threshold figures |
| `src/dataset.py` | Empty placeholder; Dataset classes actually live in notebooks and exporter |

Open notebooks inside the repository and use **Restart Kernel and Run All Cells**.
Do not treat saved outputs as a fresh run. The baseline's execution counters are
non-sequential and several cells have outputs without execution counts, so its
historical training order is not established by the saved notebook. The two train
loader diagnostic reads and training-mode shape forward affect sampler/BatchNorm
state and are intentionally preserved. Do not rerun a training cell in isolation.

Training notebooks and preparation overwrite their normal output paths. To retrain
or regenerate splits, use a separate clone/output copy and retain the submitted
results. Full training is an explicit, separate operation, not required for inference:

```bash
# In a separate clone with prepared data and the recorded CUDA environment:
.venv/bin/python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=p22-comparison --ExecutePreprocessor.timeout=-1 \
  notebooks/04_baseline.ipynb
.venv/bin/python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=p22-comparison --ExecutePreprocessor.timeout=-1 \
  notebooks/05_comparison_bce_dice.ipynb
```

The first training model construction may download ImageNet encoder weights.
Inference below constructs `encoder_weights=None` and loads the complete state dict.

## Restore weights and run inference

The clone includes neither checkpoints nor full probability archives. The
[comparison release](https://github.com/1r444444/CV-P22-building-segmentation/releases/tag/comparison-bce-dice-2026-10-19)
contains a 57,423,827-byte checkpoint, 35,307,019-byte val cache and 87,927,103-byte
test cache, plus checksums. These assets were present when inspected on 2026-11-09;
this release is marked prerelease, not a final team submission. Obtain only what
you need. For checkpoint-only inference (requires `curl`):

```bash
mkdir -p verification/artifacts
curl -L --fail --output verification/artifacts/comparison_unet_resnet18.pt \
  https://github.com/1r444444/CV-P22-building-segmentation/releases/download/comparison-bce-dice-2026-10-19/comparison_unet_resnet18.pt
.venv/bin/python - <<'PYTHON'
import hashlib, json
from pathlib import Path
path = Path("verification/artifacts/comparison_unet_resnet18.pt")
expected = json.loads(Path("results/metrics/comparison_run.json").read_text())["checkpoint_sha256"]
assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
print("Checkpoint SHA-256 verified")
PYTHON
# Requires the first test image/mask patches prepared as described in data/README.md.
.venv/bin/python src/generate_baseline_probabilities.py \
  --checkpoint verification/artifacts/comparison_unet_resnet18.pt \
  --splits test --limit 1 --device cpu --output-dir verification/inference-smoke
```

`--limit` is a smoke test and cannot reproduce the full evaluation. Remove it only
when all referenced files are available. Use `--data-root` for a separate data copy;
paths in the split CSVs remain relative to that root. Existing output archives are
rejected to prevent accidental overwrite; choose a new output directory per run.

**Baseline handoff blocker:** `results/baseline_unet_resnet18.pt` and both
`baseline_probabilities/{val,test}.npz` are absent from the clone and the only listed
release. Obtain the historical checkpoint plus checksum from Person 1, or the full
ordered probability caches from Person 3. Retraining creates a new result and does
not restore the historical baseline. Once the checkpoint and processed data exist:

```bash
.venv/bin/python src/generate_baseline_probabilities.py \
  --checkpoint results/baseline_unet_resnet18.pt \
  --output-dir verification/eval-inputs/baseline_probabilities
```

Restore the comparison caches and checksum file using the `gh release download`
commands in [COMPARISON_RUN.md](COMPARISON_RUN.md#produced-artifacts), adapting their
`--dir` values to `verification/eval-inputs/comparison_probabilities` and checking
SHA-256 against `comparison_artifact_checksums.txt`. Or generate them from the
verified comparison checkpoint with the exporter and that output directory.
The release caches total about 123 MB; they were not downloaded by the local audit.

## Evaluation and artifact regeneration

After both complete val/test cache sets and processed image/mask patches are present,
execute notebook 06 with the current virtualenv kernel. The notebook verifies cache
shape, finite probabilities, sample identities and order against committed CSVs.
Its default outputs go to `verification/evaluation/`, preserving submitted artifacts.
To use the separate cache locations above:

```bash
P22_RESULTS_ROOT="$PWD/verification/eval-inputs" \
P22_EVAL_OUTPUT_ROOT="$PWD/verification/eval-rerun" \
.venv/bin/python -m nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=p22-local --ExecutePreprocessor.timeout=600 \
  --output-dir verification/eval-rerun notebooks/06_evaluation.ipynb
```

Set `P22_DATA_ROOT` if `data/processed/...` is under a different root.
Outputs include `results/metrics/{comparison_threshold_sensitivity,threshold_sensitivity_all_models}.csv`
and four figures under the output root's `figures/`. The standalone comparison
example figure is generated by notebook 05, cells 19–22. Failure examples in
notebook 06 use test indices 113, 216, 267 at **threshold 0.5**, for both models.

## Recorded results and metric conventions

These are saved historical values, **not full-split metrics reproduced by this audit**.
Main comparison: threshold **0.5 for both models**, global foreground pixel totals:

| Model | Val IoU | Val Dice | Test IoU | Test Dice |
| --- | --- | --- | --- | --- |
| BCE | 0.6338 | 0.7759 | 0.6898 | 0.8164 |
| BCE + Dice | 0.6160 | 0.7624 | 0.6681 | 0.8010 |

The separate sensitivity experiment evaluates 0.3/0.5/0.7 and selects thresholds
using validation IoU: BCE 0.5, BCE + Dice 0.3. At those thresholds comparison test
IoU/Dice are **0.6718/0.8037**. Do not present these as the fixed-0.5 experiment.

Predictions are `sigmoid(logits) >= threshold`. Global IoU is `I/(U+1e-7)`;
Dice is `2I/(P+T+1e-7)`, aggregated over the entire split, including padded and
empty patches. Per-image failure ranking adds epsilon to numerator too (empty/empty
scores 1); it is not the global metric. The soft Dice training loss is per patch,
not global. Loss values for different objectives must not be compared directly.
The two overlap metrics are related and do not provide independent evidence.

Recorded results live in `results/metrics/`, selected binary masks in
`results/predictions/`, and figures in `figures/`. The main comparison table is
`comparison_initial_comparison.csv`; sensitivity is `threshold_sensitivity_all_models.csv`.
The milestone PDF is `report/SecondApproachSubmission_P22.pdf`, **not** the final
technical summary. The final summary remains an unfilled template and is intentionally
not authored during this audit. No demo slides were created.

Comparison run: Python 3.12.3, Linux, RTX 5070; training including epoch validation
312.19 s; test evaluation including loading/loss/metrics over 360 patches 1.49 s.
See `comparison_run.json` for timing boundaries, source/split/checkpoint hashes and
package versions. Baseline ran on T4; no complete baseline environment/timing
manifest is committed. Hardware differences and one run per condition limit causal
claims and prevent a direct runtime comparison.

## Remaining handoff and verification

- Person 1: baseline checkpoint/hash, actual baseline environment and timing,
  original training execution order, data license and content/geographic leakage checks.
- Person 2: confirm the published comparison assets are the final intended run.
- Person 3: publish baseline caches or regenerate them from the historical checkpoint;
  add likely causes for at least three concrete failures, confirm fixed vs selected
  thresholds, and commit the existing results-checkpoint report if it is a team source.
- Person 4: integrate those artifacts, rerun complete evaluation from a clean
  environment, record actual commands/times, then assemble the final summary and demo.

Local audit evidence is isolated under `verification/2026-11-09/` and ignored by Git;
it must not replace `results/metrics/`. Successful installation, unit tests or a smoke
inference do not establish full metric or training reproducibility.
Use [REPRODUCIBILITY_CHECKLIST.md](REPRODUCIBILITY_CHECKLIST.md) before submission.
