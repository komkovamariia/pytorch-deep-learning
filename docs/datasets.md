# Data used in these projects

The course data come from [ML4HD's data folder](https://drive.google.com/drive/folders/14xwb6leBeD3QMZH6wnnIIOi63D1D8xmS). Download the required files with `python scripts/download_data.py ecg`. The script verifies SHA-256 checksums against the files used for the recorded run. It fails if the course file has changed, so a new dataset cannot silently replace the measured one.

Data files and course solutions are kept outside Git. Their original access and licensing conditions still apply. `docs/datasets.json` records filenames and identifiers for reproducibility.

The digit experiments use the real handwritten digits included in scikit-learn: 1,797 images at 8 by 8 resolution. Resizing is stated in each notebook. Synthetic cells, sensor signals and controlled translation examples are labeled as synthetic. Scores on those examples test the stated task and generator.

The longer mode changes the training budget. It does not make a small dataset equivalent to a larger benchmark. Test data remain separate from threshold selection and checkpoint selection.
