# Reproducibility Checklist

Complete this checklist before final submission.

- [ ] A clean clone works without relying on untracked project files.
- [ ] The documented environment can be recreated.
- [ ] The dataset can be obtained and prepared using the README instructions.
- [ ] Source code and notebooks contain no hardcoded local paths.
- [ ] All required source, configuration, metric, figure, and report files exist.
- [ ] Random seeds and relevant deterministic settings are documented.
- [ ] Final reported metrics can be reproduced from the documented commands.
- [ ] Final figures can be reproduced from the documented commands.
- [ ] The runtime environment is documented, including relevant software and hardware.
- [ ] Approximate training time is documented.
- [ ] Approximate inference time is documented.


## Integration audit — 2026-11-09

Reviewed base: `origin/main` at `e61de6bc9e5fd0fe96de69380a041f9b56946737`.
Local fixes: `audit/reproducibility-2026-11-09`. Both remote experiment branches are
ancestors of main; the comparison release is a prerelease and is not a final tag.
No applicable AGENTS.md was found in repository/ancestor directories.

The submission checklist above remains unchecked intentionally: partial tests do
not establish a clean full reproduction.

- [x] Installed the portable runtime/notebook dependencies in a fresh macOS arm64
  Python 3.11.15 venv; `pip check` passes. Original CUDA-only requirements fail here.
- [x] Direct `python -m nbconvert` uses the venv correctly; the old
  `python -m jupyter nbconvert` dispatcher selected an unrelated system installation.
- [x] Four original loss tests and four cache alignment/completeness tests pass.
- [x] Parsed all notebook code cells; no syntax errors. No mandatory author-local
  absolute paths in executable sources were found; saved output paths are historical.
- [x] Split CSV counts/patch IDs/source disjointness and hashes match the recorded
  comparison manifest (4932/144/360 patches; 137/4/10 sources).
- [x] Manifest hashes of preprocessing, loss, comparison support and both training
  notebooks match; training code/conditions were not changed by this audit.
- [x] Downloaded comparison checkpoint and verified SHA-256. Prepared eight first
  test patches from one primary-source TIFF pair with the existing notebook 02
  functions; all eight foreground fractions match the committed CSV.
- [x] CPU inference on those eight patches succeeds with probabilities `(8,1,256,256)`.
  Binary masks differ from the historical CUDA-selected PNGs by 1–5 pixels per
  patch (18 pixels total). This is a smoke test, not exact/full reproduction.
- [x] Saved metric CSVs agree with training histories, the initial comparison,
  sensitivity values at threshold 0.5 and the IoU/Dice relationship. The separate
  validation-selected threshold experiment is explicitly distinguished in README.
- [x] Recreated the comparison table from saved scalar metrics and two threshold
  plots from saved CSVs in a separate verification directory. This does not
  recompute metrics from predictions or regenerate data-dependent failure images.
- [ ] Full notebook 06 evaluation: the real fresh-kernel run executes cells 1–3
  and stops at cell 4 with missing `results/comparison_probabilities/val.npz`.
  Both complete val/test cache sets and processed data are needed. Baseline
  checkpoint/caches have no published handoff location.
- [ ] All 144 validation / 360 test predictions and metrics reproduced from weights.
- [ ] Training reproduced: not attempted; needs the intended CUDA environment,
  complete dataset and confirmation of historical baseline execution order.
- [ ] Dataset license, content/geographic leakage checks and baseline runtime and
  complete software environment confirmed by Person 1.
- [ ] Final intended run/threshold convention and substantive likely causes for
  three failure cases confirmed by Persons 2–3.
- [ ] Final member contributions, final technical summary and demo assembled.
  The summary and slides were intentionally not created in this audit.

Local logs, freeze, smoke outputs and regenerated figures are in
`verification/2026-11-09/` (ignored). They must not replace submitted artifacts.
The existing `Downloads/ResultsCheckpointSubmission_P22.pdf` is an external
three-page results checkpoint, not a committed final two-page technical summary.
