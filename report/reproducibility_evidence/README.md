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

Baseline matches exactly at threshold 0.5. Comparison cached metrics match exactly;
fresh CPU inference matches four-decimal report precision. Sensitivity tables match
at four decimals; rendered PNGs are not byte-identical. **Training was not rerun.**
The video is reported recorded by the team; this package does not verify its content
or substitute for a shareable recording link.

Reproduce with README's checkpoint exporter and notebook 06. Demo scripts are a
separate selected-patch demonstration and are not required for full evaluation.
