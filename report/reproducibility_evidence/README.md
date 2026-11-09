# Verification evidence - 9 November 2026

These are lightweight copies of **already completed** local checks, not new
experimental results. Original logs, weights, probability archives and outputs
remain unchanged under the ignored `verification/` directories. Absolute project
paths in these JSON copies are made repository-relative; numbers and booleans are
unchanged. `provenance.json` records the original source paths and SHA-256 values.

| Record | Scope |
| --- | --- |
| `audit_environment.json` | Actual CPU audit runtime and evaluation code commit |
| `baseline_checkpoint.json` | Received checkpoint size/hash, finite tensors and strict architecture load |
| `baseline_evaluation.json` | Full 144/360 inference, exact IoU/Dice at 0.5, eight exact masks, cache identities/hashes |
| `comparison_cached_metrics.json` | Comparison-only full release-cache check; its `baseline_evaluation_reproduced: false` describes that earlier comparison-only step, not the later baseline check |
| `comparison_fresh_inference.json` | Fresh CPU comparison inference versus release probabilities/metrics |
| `notebook_execution.json` | All 14 notebook 06 code cells completed in order |
| `notebook_output_comparison.json` | Two sensitivity tables and four figures versus submitted artifacts |
| `demo_fresh_runs.json` | Two runs; four real loads/forwards; old caches/PNGs blocked; agreement with prior CPU inference |
| `final_acceptance.json` | Clean tracked-source copy, newly installed venv, full fresh inference, notebook 06, release-cache route, demo and preparation-function checks |
| `clean_copy_environment.txt` | Exact freeze of the fresh macOS acceptance environment; a record, not a portable requirements file |
| `public_artifact_access.json` | Anonymous bounded download probes for data, weights and recorded demo; no full file/video playback verification |

Baseline matches exactly at threshold 0.5. Comparison cached metrics match exactly;
fresh CPU inference matches four-decimal main-comparison precision at 0.5.
Sensitivity tables using released comparison caches match at four decimals;
fresh comparison CPU inference gives val Dice 0.7653 rather than release 0.7654
at threshold 0.3. Selected thresholds are unchanged. Rendered PNGs are not
byte-identical. **Training was not rerun.** The agreed video link is in README and
the summary; anonymous range access passed, but playback was not tested.

Reproduce with README's checkpoint exporter and notebook 06. Demo scripts are a
separate selected-patch demonstration and are not required for full evaluation.
