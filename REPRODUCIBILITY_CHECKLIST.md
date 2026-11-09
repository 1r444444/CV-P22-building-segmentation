# Reproducibility checks

Full evaluation checks were recorded on 9 November 2026. Report layout and the
contribution table were checked on 10 November 2026. Exact results are linked from
[the evidence index](report/reproducibility_evidence/README.md).

## Confirmed checks

- [x] A clean source copy with a newly installed Python 3.11 CPU/macOS environment
  runs using external evaluation data and checkpoints in documented locations.
- [x] Dependency installation and `pip check` pass; all 12 tests pass.
- [x] Executable sources have no mandatory author-specific absolute paths.
- [x] Split counts and source filename separation match the recorded manifest:
  4932/144/360 patches from 137/4/10 source images.
- [x] Existing preparation functions regenerate all 144 validation and 360 test
  patches, with byte-identical PNG files and split CSVs.
- [x] Preprocessing, loss, model code and training notebooks retain their recorded
  experimental contents; submitted metrics, predictions and figures are preserved.
- [x] Both checkpoint hashes are checked; baseline tensors are finite and the
  U-Net/ResNet18 checkpoint loads with strict architecture matching.
- [x] Full baseline checkpoint inference matches all four saved IoU/Dice values
  at 0.5 exactly, together with all eight published masks.
- [x] Comparison release-cache metrics match exactly; fresh CPU inference matches
  the main 0.5 metrics at four decimals.
- [x] Notebook 06 completes all 14 code cells sequentially and generates two CSVs
  and four figures in separate output directories. PNG raster identity is not claimed.
- [x] Sensitivity with released comparison caches matches at four decimals.
  Fresh CPU comparison val Dice at 0.3 is 0.7653 instead of release 0.7654;
  validation-selected thresholds remain 0.5 for BCE and 0.3 for BCE + Dice.
- [x] Demo test 0/113 loads both checkpoints and performs fresh forward passes.
  Repeated outputs match earlier verified CPU probabilities exactly; existing
  caches and prediction PNGs were blocked during the fresh-inference check.
- [x] Failure cases 113, 216 and 267 are checked against images, ground truth and
  predictions. Observations and possible causes are separated; the boundary/context
  hypothesis for 113 does not establish why models differ.
- [x] Environment, metric aggregation, threshold conventions and approximate
  training/inference timing boundaries are documented.
- [x] Anonymous bounded reads of raw data, weights, release caches and demo succeed
  with HTTP 206; the evidence records the scope of these access checks.
- [x] The summary has exactly two readable A4 pages, a contribution table and a
  clickable recorded-demo link. Renata's contribution is evaluation and failure analysis.

## Limits of the verification

| Item | Scope |
| --- | --- |
| Training from scratch | Not repeated |
| New Linux/CUDA installation | Not performed; the recorded environment is documented |
| Full raw download and sequential notebook 02 with train data | Not repeated; evaluation functions and all evaluation patches were checked |
| Clean-copy inputs | Previously verified external evaluation PNGs and weights; no author source or venv reuse |
| Cross-hardware equality | Fresh comparison CPU inference is not bitwise equal to the release probabilities |
| Geographic/content duplication | Not independently excluded; source filename separation was checked |
| Historical baseline environment/order | Complete records are unavailable |
| Download/video checks | Anonymous byte-range access, not full repeat downloads or playback testing |

## Repeat the checks

From the repository root after installing the environment:

```bash
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
```

Follow [README.md](README.md) for complete checkpoint export, evaluation and demo
commands. New outputs belong in separate directories. The included
`report/reproducibility_evidence/` records do not replace the submitted
`results/metrics/` or `figures/`.
