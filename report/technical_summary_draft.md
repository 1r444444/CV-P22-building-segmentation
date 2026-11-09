# P22 — Building Segmentation in Satellite Images

**Computer Vision 2026 · Technical summary · DRAFT, 2026-11-09**

**Team:** Renata Salikhzyanova, Kristina Soloveva, Elena Tuturina, Irina Bychkova.

> Working skeleton for two pages; final pagination must be checked after figures
> and results are inserted. Implementation facts below were inspected in code;
> split metadata and a small inference were checked by execution. Full validation/
> test metrics and training have not been independently reproduced. Historical
> metric CSVs are not promoted to verified final results in this draft.

<!-- PAGE 1: problem, dataset, method, one compact result table. -->

## 1. Problem and question

We study binary building segmentation in aerial images. The engineering question
is: **does adding soft Dice to binary cross-entropy improve foreground IoU and
Dice with the same U-Net + ResNet18 and nominal training settings?**
The intended changed factor is the loss function.

Source: [experiment description](../COMPARISON_RUN.md#experiment);
training notebooks 04 (cells 11–13) and 05 (cells 12–15).

## 2. Dataset

The implementation targets Massachusetts Buildings, using the Kaggle version-2
repack. Committed split CSVs contain **137/4/10 source images** and
**4932/144/360 train/validation/test patches**. Preparation pads 1500 × 1500 images
to 1536 × 1536 and crops non-overlapping 256 × 256 image/mask pairs, retaining all
36 patches per source, including background-only patches.

Source filenames are disjoint across the committed splits; their counts, patch
IDs and manifest hashes were checked. Full dataset contents and geographical
overlap were not checked. RGB inputs use ImageNet normalization; masks become
single-channel binary tensors via grayscale values `> 0`.

**[PENDING: confirm dataset license and any further leakage checks.]**

Sources: [data instructions](../data/README.md), notebook 02 (cells 4–6, 13–16),
[shared preprocessing](../src/preprocessing.py), and
[split hashes](../results/metrics/comparison_run.json).

## 3. Method and experiment

Both implementations use `smp.Unet` with an ImageNet-pretrained ResNet18 encoder,
three input channels and one output logit. Baseline loss is `BCEWithLogitsLoss`;
the second condition adds mean per-patch soft Dice loss, coefficients 1:1,
smoothing 1. Shared nominal settings are Adam, learning rate 0.001, batch size 8,
10 epochs and seed 42, without augmentation. Each condition selects its checkpoint
by its own minimum validation loss.

The main comparison uses `sigmoid(logits) >= 0.5` for both models. Global foreground
metrics are `IoU = I/(U + 1e-7)` and `Dice = 2I/(P + T + 1e-7)` over each entire
split, including padded pixels. A separate sensitivity analysis uses thresholds
0.3/0.5/0.7; any threshold selection must use validation data and be labelled
separately. No metric definition or experiment setting is changed by this draft.

## 4. Results — awaiting full evaluation

**[INSERT verified full-split results only after artifact recovery and evaluation.]**

| Model, common threshold 0.5 | Val IoU | Val Dice | Test IoU | Test Dice |
| --- | --- | --- | --- | --- |
| BCE baseline | PENDING | PENDING | PENDING | PENDING |
| BCE + soft Dice | PENDING | PENDING | PENDING | PENDING |

**[Record evaluation commit, checkpoint/cache checksums, exact split and command.]**
**[INSERT sensitivity result, threshold-selection rule and supporting artifact.]**

<!-- PAGE 2: interpretation, one compact failure figure, reproducibility, contributions. -->

## 5. Interpretation and limitations

**[INSERT a conclusion answering the question, supported by the verified table;
state effect size and distinguish fixed from validation-selected thresholds.]**

The historical runs used different GPUs (T4 and RTX 5070). The baseline software
environment and historical cell order are incompletely documented. A single run
per condition and four validation source images limit generalization; a measured
difference would not by itself establish statistical significance or isolate loss
as its sole cause. BCE and combined-loss values are different objectives.

## 6. Failure analysis — figure and causes pending

**[INSERT FINAL FIGURE: one compact panel with input, ground truth and both models
for three identified test patches; include threshold, identities and provenance.]**

Notebook 06 (cells 11 and 16) selects the following examples at threshold 0.5.
The descriptions below summarize existing materials, not newly verified findings.

| Test CSV index and identity | Existing description | Required completion |
| --- | --- | --- |
| 113: `23429020_15`, patch 05 | Large building largely missed by comparison model | [Verify predictions; explain likely cause with visual evidence.] |
| 216: `23579005_15`, patch 00 | Small building only partly detected | [Verify predictions; explain likely cause with visual evidence.] |
| 267: `23729035_15`, patch 15 | False positives near water | [Verify predictions; explain likely cause with visual evidence.] |

Describe likely causes as hypotheses unless an experiment tests them. Existing
figures under `figures/` are candidates; none is designated final by this draft.

## 7. Reproducibility and runtime

Repository: https://github.com/1r444444/CV-P22-building-segmentation.
Audit base: `e61de6b`; setup, inference and evaluation commands are in
[README](../README.md#setup).

**Verified scope:** fresh macOS arm64 Python 3.11.15 installation, dependency
check, eight unit/integration tests, split metadata checks and CPU inference on
eight test patches from one original image. Comparison checkpoint SHA-256 matches
the recorded manifest. This is a smoke test, not full metric reproduction.

**Blockers:** recover the historical baseline checkpoint/checksum, complete
validation/test caches for both models and processed evaluation data; then rerun
evaluation. Training has not been reproduced. Recorded CUDA configuration is
documented separately and was not revalidated by the CPU audit.

**[INSERT final software/hardware environment and measured training/evaluation
times, with timing boundaries; label historical timings separately if retained.]**

## 8. Actual contributions — confirmation required

- **Member 1 — [confirm name]:** [INSERT actual dataset/baseline work and artifacts.]
- **Member 2 — Elena Tuturina:** [CONFIRM actual comparison work and final scope;
  existing milestone documents the BCE + Dice implementation and recorded run.]
- **Member 3 — [confirm name]:** [INSERT actual evaluation, sensitivity and failure analysis work.]
- **Member 4 — Irina Bychkova:** [CONFIRM integration/reproducibility work and
  artifact references; add summary/demo work only when actually completed.]

<!-- FINALIZATION: fill placeholders with evidence, select final figure(s), remove
     draft/editorial notes, and verify the rendered document is at most two pages. -->
