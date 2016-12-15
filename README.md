# OHBM-2016
Submission to OHBM 2016 on functional lateralization using the neurovault dataset.

### Installation

1. clone this repository
2. `pip install -r requirements.txt`

Note: if using `virtualenv`, run: `fix-osx-virtualenv env/` (if `env/` is your virtual env root)

### Analyses

Run each script with `--help` to view all script options.

* `analysis.py` - Downloads images, computes components, runs sparsity analyses. Key metrics include:
    * HPAI (Hemispheric Participation Asymmetry Index): (L-R)/L+R for # of voxels above a given threshold for each component
    * SSS (Spatial Similarity Score): correlation measure to compare R and L in wb components
    * ACNI (Anti-correlated network index): proportion of anti-correlated network, or the voxels with negative weight (and with magnitude above a given threshold) in a given component.
* `match.py` - Downloads images, computes components, compares/matches & plots components.
* `qc.py` - Downloads images, visualizes them for quality control purposes.


### Outputs

For `analysis.py` / `match.py`:
* `ica_nii` - directory containing Nifti1 label maps for each of 20 ICA components when run on left-only, right-only, and both hemispheres.
* `ica_map` - Png images showing each component above (20 for each ICA run) when run on left-only, right-only, and both hemispheres.

For `qc.py`:
* `qc` - directory of images showing 16 nii files for review. To exclude images / collections, use [`fetch_neurovault`'s filtering procedures](https://github.com/bcipolli/nilearn/blob/neurovault-downloader/nilearn/datasets/func.py#L1505)

For `match.py`:
* `ica_nii/[dataset]/` - directory containing Nifti1 image maps for n ICA components when run on left-only (L), right-only (R), and both hemispheres (wb) for the given dataset.
* `ica_nii/{dataset}/[n]` - directory containing PNG images for each of n ICA component
* `ica_imgs/[dataset]/analyses/[n]` - Matching matrices (wb_{{L/R}}_simmat.png) showing match scores between wb and L/R components for n ICA decomposition.
* `ica_imgs/[dataset]/analyses/n/{{forced/unforced}}-match/` - directory containing PNG images for pairs of matched wb and R/L, and combined RL. Forced-match force one-to-one matching, while unforced-match pairs best-matching R/L/RL for every wb component. For each pair, term comparison is also saved (each comparison as PNG and summary as csv).

For `analysis.py`:
* `ica_imgs/[dataset]/analysis/1_wb_HPAI.png` - Graph of the HPAI in wb ICA images for a given range of n_components. Size of dot = # voxels above a global threshold.

* `ica_imgs/[dataset]/analysis/2_vcSparsity_comparison_{{L/R}}.png` - Graph of the sparsity over a range of n_components for wb, R, and L components: if there is no asymmetric activity, the contrast should be similar in wb and R/L, resulting in similar voxel count when comparing for each hemisphere (confirm this by performing the same analysis with dummy half-brain composed of randomly selected voxels). Increased contrast in unilateral components suggest ‘masking’ of lateralized activity by wb analysis.

* `ica_imgs/[dataset]/analysis/3_l1Sparsity_comparison_{{L/R}}.png` - Same idea as above but calculate sparsity using L1 rather than voxel count with an arbitrary threshold.

* `ica_imgs/[dataset]/analysis/4_Matching_results_box.png` - Matching results: Graph of the change in matching performance (average dissimilarity score and proportion of unmatched unilateral components) over a range of n_components. High-quality matching suggests refinement via hemi-analysis; low-quality matching suggests masking via wb analysis.

* `ica_imgs/[dataset]/analysis/5_wb_RL_SSS_{{box/dots}}.png` - Graph of SSS over a range of n_components, comparing SSS of wb and the matched RL. Decrease suggests asymmetric organization revealed by unilateral ICA.

* `ica_imgs/[dataset]/analysis/6_ACNI_comparison.png` - Graph of ACNI over a range of n_components for L/R and wb analyses.

* `ica_imgs/[dataset]/analysis/[n_components]/1_PosNegHPAI_{{n_components}}.png` - Relationship between positive and negative HPAI, for each n_component. The robust presence of sparsity on the negative side indicates anticorrelated networks being pulled out as a single component; this graph shows the relationship between HPAI of the anticorrelated networks.

* `ica_imgs/[dataset]/analysis/[n_components]/2_HPAIvsSAS_{{n_components}}.png` - Graph of the relationship between HPAI and SAS score for a particular n_component decomposition.
