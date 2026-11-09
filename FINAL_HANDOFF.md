# Final submission handoff - 9 November 2026

This is a publication inventory, not confirmation that anything was uploaded.
The inspected GitHub `main` is `e61de6bc9e5fd0fe96de69380a041f9b56946737`.
All audit changes remain local on `audit/reproducibility-2026-11-09`; no push was
made. The final PDF is based on the existing Markdown, not a new report.

## Transfer from the audit branch

Transfer the reviewed branch changes together, preserving existing code, split
CSVs, metrics and figures already on main. Do not copy only the PDF/README.

| Files changed or added locally versus main | Why transfer |
| --- | --- |
| `requirements.txt`, `requirements-notebooks.txt`, `requirements-comparison.txt` | Portable inference/evaluation setup, notebook tools and separately recorded CUDA environment |
| `src/generate_baseline_probabilities.py` | Checked checkpoint-only exporter for **either** model, complete sample identities and overwrite protection |
| `notebooks/06_evaluation.ipynb` | Separate input/output roots and cache order/completeness validation; evaluation definitions unchanged |
| `tests/test_integration.py` | Protect complete and correctly aligned evaluation caches |
| `README.md`, `data/README.md`, `COMPARISON_RUN.md` | Setup, dataset layout/version, weights, full inference, evaluation and artifact instructions |
| `.gitignore`, `REPRODUCIBILITY_CHECKLIST.md` | Keep heavy/local outputs separate and distinguish checks from unresolved requirements |
| `scripts/demo_inference.py`, `scripts/demo_view.py`, `tests/test_demo.py` | Repeatable demo with fresh forward passes and clearly labelled per-patch scores |
| `report/technical_summary_draft.md`, `report/build_technical_summary.py`, `report/technical_summary_P22.pdf` | Editable content, reproducible typesetting and visually checked two-page PDF |
| `report/reproducibility_evidence/README.md`, `audit_environment.json`, `baseline_checkpoint.json`, `baseline_evaluation.json`, `comparison_cached_metrics.json`, `comparison_fresh_inference.json`, `notebook_execution.json`, `notebook_output_comparison.json`, `demo_fresh_runs.json`, `provenance.json` (all JSONs in that directory) | Lightweight evidence for the report's verification claims; original protocols preserved locally |
| `FINAL_HANDOFF.md` | This inventory and remaining acceptance checks |

`git diff --name-status origin/main` plus `git ls-files --others --exclude-standard`
shows the final local inventory, including uncommitted additions. Existing
`src/preprocessing.py`, `src/losses.py`, `src/comparison.py`, training/preparation
notebooks, `data/splits/`, `results/metrics/`, `results/predictions/` and `figures/`
are already on main and still needed. Their experimental contents were not changed.

## External artifacts and independent launch

1. Install the README notebook environment in a fresh `.venv` and register its
   kernel. No `.venv-audit` or author-local path should be required.
2. Obtain the recorded Kaggle v2 raw archive and prepare patches using notebook 02
   in a separate working copy. Verify split CSVs against the committed originals.
   Evaluation needs 144 validation and 360 test patches and ground-truth masks.
   The full download is about 1.6 GB; archive extraction requires additional space.
3. Download the baseline checkpoint from README's Drive link and check its listed
   size/SHA-256. Current Drive permission metadata shows `anyone: writer`, so
   link sharing is enabled with editing rights. Prefer reader access for the
   submitted weight artifact; permissions were not changed by this audit.
   An actual logged-out download by the teacher has not been tested. The hash is
   a measured received-file fingerprint, not an independently supplied author hash.
4. Download the comparison checkpoint from the named GitHub release and verify
   its manifest hash. Generate both models' full caches with the checked exporter,
   or use the verified release caches for historical comparison results.
5. Execute notebook 06 using README's `P22_RESULTS_ROOT`/`P22_EVAL_OUTPUT_ROOT`
   command. Keep fresh exports/figures separate from `results/` and `figures/`.

There is **no requirement to publish baseline probability caches**: the documented
exporter regenerates them from the baseline checkpoint. Sharing them separately
could save time, but is not necessary. Weights/data/caches must remain available
through external links; neither this Git clone nor the evidence JSONs contain them.
Do not transfer `.venv*`, full `verification/`, raw/processed datasets or checkpoints
into normal Git history. The prepared-data Drive archive was not needed/downloaded
or validated by this audit and is not required by the documented raw-data route.

### Is `scripts/demo_inference.py` required?

It is **not required for full result reproduction**. The already checked
`src/generate_baseline_probabilities.py --checkpoint ... --output-dir ...`
accepts either state dict, applies common preprocessing and exports full validation/
test probabilities; notebook 06 then computes global metrics, sensitivity and
figures. README contains both checkpoint commands and the notebook launch command.

Include the demo script anyway to reproduce the recorded demonstration simply.
It requires its sibling `scripts/demo_view.py`; it loads both checkpoints afresh,
uses threshold 0.5, and displays test 0/113 with input, GT and both predictions.
The demo's patch IoU/Dice do not replace global evaluation.

## Remaining before submission

- Add the recorded-demo URL to README and the Markdown; verify teacher access,
  rebuild the PDF and recheck exactly two readable pages.
- Agree on shared contributions and tentative failure explanations; obtain the
  dataset license/usage terms from Person 1. Do not silently assert these resolved.
- Run the final clean-clone acceptance with only the intended published files and
  accessible artifacts. Full checkpoint evaluation already passed locally, but
  this does not establish a clean published-clone run or reproduction of training.
- Publish the reviewed branch and PDF only after authorization; provide the final
  repository link/commit. No final release/tag is currently established by this audit.
- Check the separate slide requirement (maximum three); this task creates no slides
  or video and does not claim an existing final slide deck.

PDF build: `python3 report/build_technical_summary.py` requires system `pdflatex`
and `pdfinfo`. These tools are not dependencies for running the vision models.
