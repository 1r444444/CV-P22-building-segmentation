# Technical summary files

The completed summary is available as [a two-page PDF](technical_summary_P22.pdf)
and [editable Markdown](technical_summary_draft.md).

Its eight sections cover the problem/question, dataset/preprocessing, method,
results/sensitivity, interpretation/limitations, three failure cases,
reproducibility/runtime and team contributions. The contribution table and
recorded-demo link are included in the completed report.

To rebuild from the repository root:

```bash
python3 report/build_technical_summary.py
```

System `pdflatex` and Poppler's `pdfinfo` are required. The build checks the
page count; inspect both rendered pages after editing the content.
