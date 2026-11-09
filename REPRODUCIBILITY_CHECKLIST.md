# Reproducibility Checklist

Complete this checklist before final submission.

- [x] A clean local clone of the audit branch works with external evaluation data
  and weights in documented locations; no author source or venv is reused.
- [x] The documented CPU/macOS environment can be recreated.
- [ ] The dataset can be obtained and prepared using the README instructions.
- [x] Executable sources have no mandatory author-local absolute paths.
- [ ] All required source, configuration, metric, figure, and report files exist.
- [x] Random seeds and relevant deterministic settings are documented.
- [x] Main metrics at 0.5 reproduce exactly for baseline and to four decimals for
  fresh CPU comparison; historical comparison caches reproduce exactly.
- [x] Four evaluation figures regenerate separately; raster identity is not claimed.
- [x] The runtime environment is documented, including relevant software and hardware.
- [x] Approximate training time is documented.
- [x] Approximate inference time is documented.


## Integration audit — 2026-11-09

Reviewed base: `origin/main` at `e61de6bc9e5fd0fe96de69380a041f9b56946737`.
Local fixes: `audit/reproducibility-2026-11-09`. Both remote experiment branches are
ancestors of main; the comparison release is a prerelease and is not a final tag.
No applicable AGENTS.md was found in repository/ancestor directories.

The top-level checklist remains a final clean-clone acceptance checklist. The
dated checks below describe what actually ran; they do not claim training was
reproduced or that the final GitHub already contains these local changes.

- [x] Installed the portable runtime/notebook dependencies in a fresh macOS arm64
  Python 3.11.15 venv; `pip check` passes. Original CUDA-only requirements fail here.
- [x] Direct `python -m nbconvert` uses the venv correctly; the old
  `python -m jupyter nbconvert` dispatcher selected an unrelated system installation.
- [x] Twelve tests pass: four original loss tests, four cache alignment/completeness
  tests and four demo integration tests.
- [x] Parsed all notebook code cells; no syntax errors. No mandatory author-local
  absolute paths in executable sources were found; saved output paths are historical.
- [x] Split CSV counts/patch IDs/source disjointness and hashes match the recorded
  comparison manifest (4932/144/360 patches; 137/4/10 sources).
- [x] Manifest hashes of preprocessing, loss, comparison support and both training
  notebooks match; training code/conditions were not changed by this audit.
- [x] Downloaded comparison checkpoint and full release caches; SHA-256 checks
  pass. Restored all 4 validation and 10 test source PNG pairs from the recorded
  archive and regenerated 144/360 patches with existing preparation functions.
  Regenerated val/test split CSVs are byte-identical to the committed files.
- [x] Downloaded the baseline Drive checkpoint: 57,427,223 bytes, SHA-256
  `1459948c249d7133a1c5115379122c17e0a1433429f168bfffe8a1dfddec20b0`.
  All tensor values are finite and strict U-Net/ResNet18 state-dict loading passes.
- [x] Full baseline CPU inference on 144 validation and 360 test patches matches
  all four saved IoU/Dice values at 0.5 exactly; all eight published masks match.
- [x] Full comparison release-cache metrics match exactly. Full fresh CPU
  checkpoint inference matches four-decimal reported precision; small probability
  and binary-mask differences are recorded rather than hidden.
- [x] Saved metric CSVs agree with training histories, the initial comparison,
  sensitivity values at threshold 0.5 and the IoU/Dice relationship. The separate
  validation-selected threshold experiment is explicitly distinguished in README.
- [x] Notebook 06 executed all 14 code cells sequentially with complete caches and
  processed patches. Its two sensitivity tables agree at four decimals; four
  regenerated figures have small raster-dimension differences (1–3 pixels).
- [x] Failure cases 113, 216 and 267 checked against raw images, ground truth and
  both predictions. The summary separates observations from untested hypotheses;
  the boundary/context hypothesis does not establish why models differ on 113.
- [x] Two fresh demo runs on test 0 and 113 loaded both checkpoints and executed
  both models each time. Existing caches/old prediction PNGs were blocked during
  inference; results equal the earlier verified CPU probabilities exactly.
- [x] Two-page summary typeset from the existing Markdown with 11 pt body text;
  page count and rendered pages checked. Irina's demo development/testing included.
- [x] Final clean local source-copy acceptance: fresh dependency installation,
  `pip check`, 12 tests, both full checkpoint exports, notebook 06 (14 cells) and
  demo test 0/113 passed. External inputs were previously verified evaluation PNGs
  and weights copied to README locations; author source/venv were not reused.
- [x] Repeated unchanged notebook 02 preparation functions on all 14 available
  evaluation source pairs: all 1008 generated PNG files and val/test CSVs match
  byte-for-byte. Full raw download and sequential notebook 02 were not repeated.
- [x] Both notebook 06 routes checked: released comparison caches reproduce the
  historical sensitivity table at four decimals; with fresh CPU comparison outputs,
  val Dice at threshold 0.3 is 0.7653 versus historical 0.7654. Other sensitivity
  values agree at four decimals; validation-selected thresholds remain 0.5/0.3.
- [x] Anonymous range reads of raw data, weights, release caches and recorded demo
  pass (HTTP 206); full repeat downloads and video playback were not checked.
- [ ] Repeat the remote clean-clone acceptance after publication if the final
  submitted commit differs from the checked local executable sources.
- [ ] Training reproduced: not attempted; needs the intended CUDA environment,
  complete dataset and confirmation of historical baseline execution order.
- [ ] Dataset license, content/geographic leakage checks and baseline runtime and
  complete software environment confirmed by Person 1.
- [x] Agreed contribution wording and recorded-demo URL are in the summary and README.
  Renata's contribution is evaluation and failure analysis.
- [ ] Final repository publication and any slides required for the submission.
  No slides/video were created and nothing was pushed by this audit.

Full local logs, caches and regenerated figures remain in the dated `verification/`
directories (ignored). Lightweight copies with relative paths and provenance are
in `report/reproducibility_evidence/`; see `FINAL_HANDOFF.md` for publication scope.
They must not replace submitted experimental artifacts.
The existing `Downloads/ResultsCheckpointSubmission_P22.pdf` is an external
three-page results checkpoint, not a committed final two-page technical summary.
