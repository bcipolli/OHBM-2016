# OHBM-2016
Submission to OHBM 2016 on functional lateralization using the neurovault dataset.

### Installation

1. clone this repository
2. `pip install -r requirements.txt`

Note: if using `virtualenv`, run: `fix-osx-virtualenv env/` (if `env/` is your virtual env root)

### Analyses

Run each script with `--help` to view all script options.

* `analysis.py` - Runs full analysis suite (completed for SfN 2016)
* `match.py` - Downloads images, computes components, compares/matches & plots components.
* `qc.py` - Downloads images, visualizes them for quality control purposes.


### Outputs

For `analysis.py` / `match.py`:
* `ica_nii` - directory containing Nifti1 label maps for each of 20 ICA components when run on left-only, right-only, and both hemispheres.
* `ica_map` - Png images showing each component above (20 for each ICA run) when run on left-only, right-only, and both hemispheres.

For `qc.py`:
* `qc` - directory of images showing 16 nii files for review. To exclude images / collections, use [`fetch_neurovault`'s filtering procedures](https://github.com/bcipolli/nilearn/blob/neurovault-downloader/nilearn/datasets/func.py#L1505)
