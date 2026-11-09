# Data Instructions

## Source and version

Massachusetts Buildings, originally published by Volodymyr Mnih:
https://www.cs.toronto.edu/~vmnih/data/ (building image and target-map split links).
The recorded experiment used Kaggle repack **version 2**:
https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset.
The exact license/usage terms still need confirmation by the dataset owner in the
team. Do not infer a license from public download access.

The full archive is approximately **1.6 GB** and contains both PNG and TIFF trees.
Extraction and generated patches require additional disk space. Download is an
explicit step; do not start it as part of a lightweight smoke test.

From the repository root, with the environment installed as in README.md:

```bash
curl -L --fail --retry 3 --output data/massachusetts-buildings-v2.zip \
  'https://www.kaggle.com/api/v1/datasets/download/balraj98/massachusetts-buildings-dataset?datasetVersionNumber=2'
.venv/bin/python - <<'PY'
import hashlib
from pathlib import Path
path = Path('data/massachusetts-buildings-v2.zip')
assert hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() == \
    'bd271292c60033b6bcbfbe12ebbf30dc17dcd1cefc4060c7542fb2b73374875e'
print('Recorded archive checksum verified')
PY
.venv/bin/python -m zipfile -e data/massachusetts-buildings-v2.zip data/raw
# Only after a fresh extraction, and only if the destinations do not yet exist:
for split in train train_labels val val_labels test test_labels; do
  if [ -e "data/raw/$split" ]; then
    echo "Stop: data/raw/$split already exists; inspect before moving files."
    break
  fi
  mv "data/raw/png/$split" "data/raw/$split"
done
```

A Kaggle login/terms acceptance may be required depending on the service response.
If the download returns HTML or the hash differs, stop and inspect it. The hash
above is from the recorded experiment; the full download was not repeated in the
2026-11-09 integration audit.

## Expected layout and preparation

```text
data/raw/
  train/         137 RGB PNG images, 1500 x 1500
  train_labels/  137 corresponding masks, same filenames
  val/             4 images
  val_labels/      4 masks
  test/           10 images
  test_labels/    10 masks
  metadata.csv             # used by notebook 01, cell 7
  label_class_dict.csv     # used by notebook 01, cell 7
```

Masks are RGB with identical channels and values 0/255. If metadata CSVs are
elsewhere in the extracted tree, put the supplied CSVs in `data/raw/` before running
notebook 01. They are used for inspection, not patch generation.

1. In a separate working copy when retaining an existing run, restart and run
   `notebooks/01_dataset_explore.ipynb` to inspect counts, pairs and mask values.
   Its pair checker can report success for empty folders, so check the expected
   counts too; a success message alone is insufficient.
2. Restart and run `notebooks/02_prepare_subset.ipynb`. It pads the bottom/right
   with zeros from 1500 to 1536 and crops both image and mask at identical coordinates
   into a row-major 6 x 6 grid of non-overlapping 256 x 256 patches.
3. All 36 patches per source are retained, including empty masks. No random
   resplit, augmentation, interpolation or coverage filtering is introduced.
4. Confirm `data/processed/{train,val,test}/{images,masks}/` exists and that
   generated split records match the committed CSVs. Notebook 02 **writes the
   CSVs**; keep original copies or use a separate clone and compare before accepting.
5. Runtime preprocessing is `src/preprocessing.py`: RGB float32 / 255, ImageNet
   mean/std, channels first; grayscale mask `> 0` → float32 single channel.

| Split | Original images | Patches | Image files + mask files |
| --- | ---: | ---: | ---: |
| train | 137 | 4932 | 4932 + 4932 |
| val | 4 | 144 | 144 + 144 |
| test | 10 | 360 | 360 + 360 |

The committed `data/splits/{train,val,test}.csv` files contain repository-relative
paths and patch IDs. Their SHA-256 values are in
`results/metrics/comparison_run.json`. Patches from one original image stay in one
split; the committed source filename sets are disjoint. This does not establish
geographical separation or absence of duplicate content, which still need checking.

The comparison notebook's `validate_comparison_data` checks counts, dimensions,
readability, uniqueness and source-level separation. It does not prove pixel-level
identity to the original dataset. Notebook 01 and the shared mask conversion were
inspected, but the complete raw data were not validated by the local audit.

## Small inference example without the full archive

The original source supplies individual TIFFs. The audit downloaded only the first
test pair, `22828930_15`, from the `mass_buildings/test/sat/` and `test/map/` links
on the source website (about 13.5 MB for this pair), then applied the **existing**
notebook 02 padding/cropping functions to the first eight patches. All eight mask
foreground fractions matched the committed test CSV. These inputs are under
`verification/2026-11-09/data/processed/test/` locally, not under the normal data tree.
This small check does not validate the entire PNG repack or split.

Raw data and generated patches are ignored by Git. The split CSVs are already
tracked; do not use broad `git add -f data/` to publish datasets.
