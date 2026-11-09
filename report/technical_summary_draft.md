# P22 — Building Segmentation in Satellite Images

**Computer Vision 2026 · Technical Summary · Draft, 9 November 2026**

**Team:** Renata Salikhzyanova, Kristina Soloveva, Elena Tuturina, Irina Bychkova.

<!-- Target: two pages after typesetting; final pagination is not yet verified.
     Structure follows Computer_Vision_Project_Guidelines.pdf, pp. 1–3. -->

## 1. Problem and question

We investigate binary building segmentation in aerial images: **does adding soft
Dice to binary cross-entropy improve foreground overlap with the same U-Net +
ResNet18 and nominal training settings?** The intended experimental variable is
the loss function.

## 2. Dataset and preprocessing

Massachusetts Buildings, Kaggle repack version 2, contains 151 original 1500 × 1500
image–mask pairs. The original train/validation/test split has **137/4/10 images**,
producing **4932/144/360 patches**. Pairs are zero-padded to 1536 × 1536 and divided
into 36 non-overlapping 256 × 256 patches; all patches are retained. RGB inputs use
ImageNet normalization; grayscale masks are binarized with `> 0`.
Source filenames are disjoint across splits; reconstructed validation/test CSVs
match the committed files byte-for-byte. Geographical/content duplication has not
been excluded. **Dataset license confirmation remains pending.**

## 3. Method and experiment

Both conditions use U-Net with an ImageNet-pretrained ResNet18 encoder. The baseline
uses `BCEWithLogitsLoss`; the comparison adds mean per-patch soft Dice loss, with
equal coefficients and smoothing 1. Shared settings are Adam, learning rate 0.001,
batch size 8, ten epochs, seed 42 and no augmentation. Checkpoints are selected by
minimum validation loss under each condition's own objective.

The main comparison thresholds sigmoid probabilities at **0.5 for both models**.
Metrics use global foreground pixel totals, including empty and padded regions:
`IoU = I/(U + 1e-7)`; `Dice = 2I/(P + T + 1e-7)`. These are not per-patch averages.

## 4. Results and sensitivity

| Model, common threshold 0.5 | Val IoU | Val Dice | Test IoU | Test Dice |
| --- | ---: | ---: | ---: | ---: |
| BCE baseline | 0.6338 | 0.7759 | 0.6898 | 0.8164 |
| BCE + soft Dice | 0.6160 | 0.7624 | 0.6681 | 0.8010 |

Adding soft Dice reduced test IoU by **2.17 percentage points** and Dice by
**1.54 percentage points** in this configuration.

Sensitivity analysis evaluated thresholds 0.3/0.5/0.7. Maximizing validation IoU
selects **0.5 for BCE** (0.6338) and **0.3 for BCE + Dice** (0.6199). With these
validation-selected thresholds, comparison test IoU/Dice are **0.6718/0.8037**;
baseline remains **0.6898/0.8164**. Threshold 0.7 reduces validation overlap for both.
This sensitivity result is separate from the common-threshold comparison.

## 5. Interpretation and limitations

The chosen Dice term did not improve overlap, even after threshold selection.
This does not establish that Dice loss is generally inferior. One training run per
condition, four validation source images, different GPUs (T4/RTX 5070), and incomplete
baseline environment/execution-order records limit causal and statistical claims.
BCE-only and combined-loss values are not directly comparable.

## 6. Failure analysis

Renata's results checkpoint report (pp. 2–3) and the
[failure comparison](../figures/failure_comparison.png) describe three test examples
at threshold 0.5, regenerated during verification:

- **113, `23429020_15`, patch 05 — Observation:** a large roof is truncated at
  the source-image boundary. BCE recovers most of it (IoU 0.833); BCE + Dice misses
  most of it (0.085). **Possible cause:** limited spatial context may contribute
  to the omission. This is an untested hypothesis and does not establish why the
  models differ, especially since BCE succeeds on the same input.
- **216, `23579005_15`, patch 00 — Observation:** the labelled building occupies
  553 pixels (0.844% of the patch); both models recover only part of it and produce
  false positives (IoU 0.147/0.172). **Possible cause:** its small spatial extent
  may make the complete shape difficult to recover at this image resolution.
- **267, `23729035_15`, patch 15 — Observation:** both models mark a bright object
  at the water's edge, while GT contains only 18 foreground pixels; IoU is zero.
  **Possible cause:** the false positives may reflect confusion between this
  object and building foreground. Its identity is uncertain; an annotation error
  has not been established.

These explanations are visually motivated hypotheses, not demonstrated causes.

## 7. Reproducibility and runtime

Repository: https://github.com/1r444444/CV-P22-building-segmentation.
**Evaluation was repeated from both saved checkpoints on all 144 validation and
360 test patches; training was not rerun or independently reproduced.** The audit
used macOS arm64, Python 3.11.15, PyTorch 2.11.0 and segmentation-models-pytorch 0.5.0.
Baseline IoU/Dice at 0.5 and eight published masks match exactly. Comparison release
caches reproduce recorded metrics exactly; fresh CPU inference matches four-decimal
precision, with small numerical differences. Notebook 06 completed all 14 code cells
sequentially. Sensitivity values match at four decimals; PNG raster dimensions differ.
Separate logs, identities and checksums are retained under `verification/`.

Kristina estimates **about 18 minutes of training on T4** and **3–4 seconds of
batched test inference over 360 patches**, batch size 8, 256 × 256 inputs, from
progress bars. These are approximate historical estimates, not a separate benchmark
or single-image latency. The comparison manifest records **312.19 s** for training
including epoch validation/checkpoint writes and **1.49 s** for test evaluation
including loading, forward passes, loss and metrics on RTX 5070. Different hardware
and timing boundaries prevent a direct speed comparison.

## 8. Members' contributions

- **Kristina Soloveva:** dataset preparation, preprocessing and BCE baseline.
- **Elena Tuturina:** BCE + Dice implementation/run, artifacts and comparison milestone report.
- **Renata Salikhzyanova:** evaluation, threshold sensitivity, failure visualizations and descriptions.
- **Irina Bychkova:** environment/integration fixes, reproducibility verification and this draft.

**For team agreement:** confirm shared authorship of intermediate reports and any
additional contributions. Demo preparation is not claimed as completed.

<!-- Evidence notes for review, excluded from the two-page body:
     Guidelines: /Users/irina/Downloads/Computer_Vision_Project_Guidelines.pdf, pp. 1–3;
     p. 2 requires three failure cases with likely causes; the draft now separates
     reproduced observations from visually motivated, untested explanations.
     Data: notebooks/02_prepare_subset.ipynb, cells 4–6, 13–16; src/preprocessing.py.
     Models: notebooks/04_baseline.ipynb, cells 11–16;
     notebooks/05_comparison_bce_dice.ipynb, cells 12–18; COMPARISON_RUN.md.
     Results: results/metrics/baseline_metrics.csv, comparison_metrics.csv,
     comparison_threshold_sensitivity.csv, threshold_sensitivity_all_models.csv.
     Runtime: Kristina's messages supplied by the user; results/metrics/comparison_run.json.
     Failures: /Users/irina/Downloads/ResultsCheckpointSubmission_P22.pdf, pp. 2–3;
     notebooks/06_evaluation.ipynb, cells 9–11, 16; its markdown adds no case-specific causes.
     New hypotheses: verification/failure-review-2026-11-09/proposed-wording.md,
     case-statistics.json and alignment-verification.json; checked against raw
     images/masks and both models' predictions. Do not attribute these hypotheses
     to Renata's original report or present them as tested mechanisms.
     Verification: verification/baseline-drive-evaluation-2026-11-09/verification-summary.json,
     notebook-execution.json, baseline-evaluation-verification.json;
     verification/handoff-evaluation-2026-11-09/results/comparison-metric-verification.json
     and fresh-inference-verification.json. Audit files are local and ignored by Git;
     package relevant evidence and artifact access instructions before final submission.
     Provenance: verification at audit code 5ec84aa, base e61de6b.
     Contributions: ProjectPlanSubmission_P22.pdf, Team Responsibilities; Kristina's handoff;
     commits b378de7/2c418ba (kriss), b247240 (Renata); COMPARISON_RUN.md and
     SecondApproachSubmission_P22.tex (Elena); audit history and verification (Irina).
     Finalize: agree on tentative failure interpretations, license and shared contributions; package evidence;
     then typeset and verify no more than two pages. No PDF or slides generated. -->
