# P22 - Building Segmentation in Satellite Images

Team project for **Computer Vision 2026**.

## Goal

Segment building contours in aerial or satellite images using a subset of a
public dataset. Evaluate segmentation quality with IoU and Dice, and examine
errors associated with shadows, small buildings, and dense urban regions.

The exact dataset, models, hyperparameters, and experimental results are TBD.

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
- include a two-page technical summary and a demo;
- state each team member's contribution.

For this project, the primary quantitative metrics are **IoU** and **Dice**.

## Repository Structure

```text
CV-P22-project/
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

Team member names and final contribution statements: TBD.

## Dataset

TBD. The final documentation will include the dataset source and license,
subset selection, download procedure, preprocessing, split strategy, and
leakage-prevention checks. See [`data/README.md`](data/README.md).

## Setup

TBD. Exact environment creation and installation commands will be added after
the team code and dependencies are finalized.

## Experiments

TBD. This section will document the baseline, comparison condition, controlled
variables, and ablation or sensitivity experiment.

## Results

TBD. This section will contain final IoU and Dice values, a compact comparison,
runtime measurements, and links to saved figures and metrics.

## Reproducibility

TBD. Final instructions will cover dataset preparation, experiment commands,
random seeds and settings where relevant, metric/figure regeneration, hardware
and software environment, and approximate training/inference time.

Use [`REPRODUCIBILITY_CHECKLIST.md`](REPRODUCIBILITY_CHECKLIST.md) before the
final submission.
