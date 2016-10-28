# OHBM-2016
Submission to OHBM 2016 on functional lateralization using the neurovault dataset.

### Installation

1. clone this repository
2. `pip install -r requirements.txt`

Note: if using `virtualenv`, run: `fix-osx-virtualenv env/` (if `env/` is your virtual env root)

### Analyses

Run each script with `--help` to view all script options.

* `analysis.py` - Downloads images, computes components, runs sparsity analyses. Key metrics include:
    * HPI: (L-R)/L+R for # of voxels above a given threshold for each component
    * SAS: l1/l2/correlation measure to compare R and L in wb components (this is included in 2-c)
* `match.py` - Downloads images, computes components, compares/matches & plots components.
* `qc.py` - Downloads images, visualizes them for quality control purposes.


### Outputs

For `analysis.py` / `match.py`:
* `ica_nii` - directory containing Nifti1 label maps for each of 20 ICA components when run on left-only, right-only, and both hemispheres.
* `ica_map` - Png images showing each component above (20 for each ICA run) when run on left-only, right-only, and both hemispheres.

For `qc.py`:
* `qc` - directory of images showing 16 nii files for review. To exclude images / collections, use [`fetch_neurovault`'s filtering procedures](https://github.com/bcipolli/nilearn/blob/neurovault-downloader/nilearn/datasets/func.py#L1505)

For `match.py`:
* `ica_nii` - directory containing Nifti1 label maps for each of 20 ICA components when run on left-only, right-only, and both hemispheres.
* `ica_imgs` - Png images showing each component above (20 for each ICA run) when run on left-only, right-only, and both hemispheres.

For `analysis.py`:
* `ica_imgs/neurovault/analysis/1_wb_HPAI.png` - For each n_components, a scatter plot of HPI for each component. Size of dot = # voxels above chance.

* `ica_imgs/neurovault/analysis/2_vcSparsity_comparison_{{L/R}}.png` - Graph of the sparsity over a range of n_components for wb, R, and L components: if there is no asymmetric activity, the contrast should be similar in wb and R/L, resulting in roughly 2x sparsity values for wb (since there are double the amount of total voxels). Increased contrast in unilateral components suggest ‘masking’ of lateralized activity by wb analysis. (Show example image comparison)--requires ica images only

* `ica_imgs/neurovault/analysis/3_l1Sparsity_comparison_{{L/R}}.png` - ?

* `ica_imgs/neurovault/analysis/4_Matching_results_box.png` - Matching results: Graph of the change in matching performance (average dissimilarity score and proportion of unmatched unilateral components) over a range of n_components. High-quality matching suggests refinement via hemi-analysis; low-quality matching suggests masking via wb analysis. --requires matching

* `ica_imgs/neurovault/analysis/3_l1Sparsity_comparison_{{box/dots}}.png` - Graph of SAS over a range of n_components, comparing SAS of wb and the matched RL--requires matching

* `ica_imgs/neurovault/analysis/[n_components]/1_PosNegHPAI_{{n_components}}.png` - Relationship between positive and negative HPI, for each n_component. The robust presence of sparsity on the negative side indicates anticorrelated networks being pulled out as a single component; this graph shows the relationship between HPI of the anticorrelated networks.

* `ica_imgs/neurovault/analysis/[n_components]/2_HPAIvsSAS_{{n_components}}.png` - Graph of the relationship between HPI and SAS score for a particular n_component decomposition.
