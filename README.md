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

Integration status (2026-11-09): full checkpoint inference on 144 validation and
360 test patches and sequential evaluation notebook 06 were checked on macOS CPU.
Baseline IoU/Dice at threshold 0.5 and all eight published masks match exactly.
Comparison release probabilities reproduce the recorded metrics exactly; fresh CPU
comparison inference matches their four-decimal precision at threshold 0.5, with
small differences.
Training has **not** been rerun or independently reproduced. See the
[recorded-demo instructions](#recorded-demo-without-retraining) for fresh inference
on individual examples; their patch scores are separate from full-split results.

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

`scripts/` contains the selected-patch demo runner and HTML/image rendering code.

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

Confirmed contributions: **Kristina Soloveva** - dataset preparation, preprocessing
and BCE baseline; **Elena Tuturina** - BCE + Dice implementation/run, artifacts and
comparison milestone report; **Renata Salikhzyanova** - evaluation and failure
analysis; **Irina Bychkova** - environment/integration fixes, reproducibility
verification, technical summary, and development/testing of the demo script.
See the [technical summary](report/technical_summary_P22.pdf).

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
| `scripts/demo_inference.py` | Fresh inference from both checkpoints on selected test patches, with per-patch masks/metrics and a local HTML report |
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

Anonymous access was checked on 2026-11-09 with bounded HTTP requests, without
account credentials: the Kaggle v2 archive, comparison release assets, baseline
checkpoint and [recorded demo](https://drive.google.com/file/d/12eeFMhpknbHesvbJhntL6vVybsqNM__O/view?usp=sharing)
returned file bytes (HTTP 206). This supports teacher access without login at the
time checked; it is not a full repeat download or video playback test. Large Drive
files may first show a confirmation page: open the baseline link and choose
**Download / Download anyway**, then verify the received size and hash below.
See [artifact access checks](report/reproducibility_evidence/public_artifact_access.json).

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

**Baseline restore:** the checkpoint is now available from
[Kristina's Drive handoff](https://drive.google.com/file/d/1TNc2oXPQq4J8j96hvKxc_UvBhmieqRHE/view).
It is not included in Git or the comparison release. The received file has
57,427,223 bytes and measured SHA-256
`1459948c249d7133a1c5115379122c17e0a1433429f168bfffe8a1dfddec20b0`.
This fingerprints the checked file; no separate author-supplied checksum was provided.
Full inference from it reproduced baseline IoU/Dice at 0.5 exactly.
Save the checkpoint as `results/baseline_unet_resnet18.pt` and
restore processed data as described above. Then export without retraining:

```bash
.venv/bin/python - <<'PYTHON'
import hashlib
from pathlib import Path
path = Path('results/baseline_unet_resnet18.pt')
assert path.stat().st_size == 57427223
with path.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == \
        '1459948c249d7133a1c5115379122c17e0a1433429f168bfffe8a1dfddec20b0'
print('Baseline checkpoint size and SHA-256 verified')
PYTHON
.venv/bin/python src/generate_baseline_probabilities.py \
  --checkpoint results/baseline_unet_resnet18.pt \
  --device cpu --output-dir verification/eval-inputs/baseline_probabilities
```

For fresh full comparison inference, use the same exporter and processed data:

```bash
.venv/bin/python src/generate_baseline_probabilities.py \
  --checkpoint verification/artifacts/comparison_unet_resnet18.pt \
  --device cpu --output-dir verification/eval-inputs/comparison_probabilities
```

Both commands export all 144 validation and 360 test patches; neither trains.
As an alternative for the exact historical comparison scores, download its release
caches and checksum file (about 123 MB):

```bash
mkdir -p verification/eval-inputs/comparison_probabilities
curl -L --fail --output verification/eval-inputs/comparison_probabilities/val.npz \
  https://github.com/1r444444/CV-P22-building-segmentation/releases/download/comparison-bce-dice-2026-10-19/val.npz
curl -L --fail --output verification/eval-inputs/comparison_probabilities/test.npz \
  https://github.com/1r444444/CV-P22-building-segmentation/releases/download/comparison-bce-dice-2026-10-19/test.npz
curl -L --fail --output verification/eval-inputs/comparison_probabilities/comparison_artifact_checksums.txt \
  https://github.com/1r444444/CV-P22-building-segmentation/releases/download/comparison-bce-dice-2026-10-19/comparison_artifact_checksums.txt
.venv/bin/python - <<'PYTHON'
import hashlib
from pathlib import Path
folder = Path('verification/eval-inputs/comparison_probabilities')
expected = {
    'val.npz': 'edba5c9b24f556f920efe5ff38f928b6e28594e25582b4c2142e9e058962bf47',
    'test.npz': 'cbae3cef0b4064e233a6df93e3e26a8cca2b8335aa09d0dd0637a474c4cebced',
}
for name, digest in expected.items():
    with (folder / name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest, name
    print(name, 'SHA-256 verified')
PYTHON
```

Choose **either** fresh comparison inference **or** release cache restoration into
this directory; keep separate directories if comparing both. The local audit
checked both routes. Cached comparison metrics match exactly; fresh CPU inference
matches the reported four-decimal precision, with small numerical differences.

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

These are the saved run values, now independently checked on the full evaluation
splits: baseline checkpoint inference matches exactly; comparison cache evaluation
matches exactly and fresh CPU checkpoint inference agrees at four decimals.
Main comparison: threshold **0.5 for both models**, global foreground pixel totals:

| Model | Val IoU | Val Dice | Test IoU | Test Dice |
| --- | --- | --- | --- | --- |
| BCE | 0.6338 | 0.7759 | 0.6898 | 0.8164 |
| BCE + Dice | 0.6160 | 0.7624 | 0.6681 | 0.8010 |

The separate sensitivity experiment evaluates 0.3/0.5/0.7 and selects thresholds
using validation IoU: BCE 0.5, BCE + Dice 0.3. At those thresholds comparison test
IoU/Dice are **0.6718/0.8037**. Do not present these as the fixed-0.5 experiment.
For the historical sensitivity table, use the released comparison caches. Fresh
CPU probabilities give comparison validation Dice **0.7653325306650232** at 0.3,
versus **0.7653526677389473** from the release (0.7653 vs 0.7654 after rounding).
This is a small numerical difference; selected thresholds and reported selected-
threshold test IoU/Dice are unchanged at four decimals. All main 0.5 metrics match
at four decimals. Do not claim every fresh CPU sensitivity value rounds identically.

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
technical summary. The [technical summary draft](report/technical_summary_draft.md)
contains verified results, limitations and cautiously labelled failure hypotheses.
The [two-page summary PDF](report/technical_summary_P22.pdf) is typeset from that
source; both pages were visually checked. Rebuild with
`python3 report/build_technical_summary.py` (requires `pdflatex` and Poppler's
`pdfinfo`; these are report-build tools, not inference dependencies).
The agreed [recorded demo](https://drive.google.com/file/d/12eeFMhpknbHesvbJhntL6vVybsqNM__O/view?usp=sharing)
is linked in the PDF; no demo slides were created by this audit.
See [FINAL_HANDOFF.md](FINAL_HANDOFF.md) for the publication
file list and [verification evidence](report/reproducibility_evidence/) for the
lightweight audit records.

Comparison run: Python 3.12.3, Linux, RTX 5070; training including epoch validation
312.19 s; test evaluation including loading/loss/metrics over 360 patches 1.49 s.
See `comparison_run.json` for timing boundaries, source/split/checkpoint hashes and
package versions. Baseline ran on T4; no complete baseline environment/timing
manifest is committed. Hardware differences and one run per condition limit causal
claims and prevent a direct runtime comparison.

## Recorded demo without retraining

[Watch the agreed recorded pipeline demo (Google Drive)](https://drive.google.com/file/d/12eeFMhpknbHesvbJhntL6vVybsqNM__O/view?usp=sharing).
The commands below repeat the working pipeline without retraining.

After the setup/data/weight steps above, run from the repository root:

```bash
.venv/bin/python scripts/demo_inference.py --indices 0 113 --device cpu \
  --comparison-checkpoint verification/artifacts/comparison_unet_resnet18.pt --open
```

Every invocation creates a new `verification/demo/run-.../` directory and opens
its `index.html` in the default browser after inference. The page has links for
**Test 0** and **Test 113 — failure example**. Each shows **input RGB | ground truth |
Baseline (BCE) | BCE + soft Dice**. The shared threshold is fixed at **0.5**.
All displayed IoU/Dice values describe the **individual 256 × 256 patch**, not the
full test set. Ground truth is used for scoring/display; only RGB enters the model.

The script loads both state dicts with `weights_only=True` and strict architecture
matching on every run, creates the model with `encoder_weights=None`, calls
`model.eval()` and performs a fresh forward pass under `torch.inference_mode()`.
It does not read previous probability caches or prediction PNGs. The terminal
prints checkpoint paths/SHA-256, forward-pass messages and fresh patch metrics.
The HTML report reads only images created in the new run directory.

Each output contains the two probability NPZs (with test indices and image/mask
identities), individual masks, four-column PNGs and `run.json`. Existing output
directories are rejected. For a fixed new directory, use `--output-dir`; to avoid
opening a browser, omit `--open` and use the `open '.../index.html'` command printed
by the script. No extra dependencies or server are required.

The command uses `data/processed/test/`, `results/baseline_unet_resnet18.pt` and the
comparison checkpoint restored above. If both checkpoints are instead under
`results/*unet_resnet18.pt`, omit `--comparison-checkpoint`. Explicit alternatives are:

```bash
.venv/bin/python scripts/demo_inference.py --indices 0 113 --device cpu \
  --data-root /path/containing/data \
  --baseline-checkpoint /path/to/baseline_unet_resnet18.pt \
  --comparison-checkpoint /path/to/comparison_unet_resnet18.pt --open
```

Here `--data-root` contains `data/processed/test/{images,masks}/`; it is not the
images directory itself. The script reports missing inputs instead of downloading
files or training. To preview an input before starting the recorded command:

```bash
open data/processed/test/images/22828930_15_patch_00.png
```

Test 0 has clearly visible buildings and imperfect but useful predictions; it is
the first test row, not a selected best score. Test 113 demonstrates the comparison
model's large omission. Fresh CPU comparison on test 113 gives patch IoU about
**0.0842**, while the recorded release probabilities give **0.0848**: five binary
pixels differ at threshold 0.5. The demo agrees exactly with the previously checked
fresh CPU probabilities, and does not substitute the historical result.

Suggested recording flow: show input; run the command while narrating; show test 0
and its four panels; use the page link to show test 113 and explain the untested
boundary/context hypothesis. If presenting full-test numbers, label the separate
360-patch table clearly. Finish by stating that evaluation was verified from
weights and training was not rerun. On Mac, press Shift–Command–5, choose a recording
area and select your microphone in Options; see
[Apple's screen-recording instructions](https://support.apple.com/en-euro/guide/mac-help/mh26782/mac).

## Remaining handoff and verification

- Person 1: complete baseline environment/execution-order records, data license
  and content/geographic leakage checks; baseline timing remains a progress-bar estimate.
- Team: complete the final submission package, including any required slides.
- Person 4: publish the reviewed audit changes when authorized. The agreed video
  URL and contributions are already in the summary; clean-copy checks are complete
  within the scope below.

Clean-copy acceptance passed on macOS arm64 with a new `.venv`: installation,
12 tests, both full checkpoint exports, notebook 06 and demo test 0/113. Both
fresh-comparison and release-cache evaluation routes were checked separately.
External inputs were previously verified prepared validation/test files and
checkpoints copied into the documented paths; no author source files or venv were
reused. The full 1.6 GB raw download, full sequential preparation notebook with
training data and model training were not repeated. See the
[acceptance record](report/reproducibility_evidence/final_acceptance.json) and
[fresh environment](report/reproducibility_evidence/clean_copy_environment.txt).

Local audit evidence is under the dated `verification/` directories and ignored by
Git; it must not replace `results/metrics/`. The demo is a selected-patch inference
demonstration; it does not rerun full evaluation or training.
Use [REPRODUCIBILITY_CHECKLIST.md](REPRODUCIBILITY_CHECKLIST.md) before submission.
