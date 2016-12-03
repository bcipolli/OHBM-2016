# *- encoding: utf-8 -*-
# Author: Ami Tsuchida
# License: BSD
"""
"""

import os.path as op

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nilearn.masking import apply_mask
from scipy import stats

from .sparsity import SPARSITY_SIGNS
from nilearn_ext.masking import flip_img_lr, get_hemi_gm_mask
from nilearn_ext.plotting import save_and_close


def calculate_hpai(wb_img, percentile=95.0):
    """
    Compute HPAI for each component image of the given WB ICA image.

    It is calculated by first taking the voxels whose magnitude is above
    a given percentile (default 95.0), and calculating (R-L)/(R+L) for
    the number of voxels. The L grey matter mask is applied to both sides to
    keep the total numnber of voxels in the hemispheres eaual.

    HPAI is calculated separately for positive, negative, and absolute values,
    and returned as a dictionary with SPARSITY_SIGNS as keys.
    """
    n_components = wb_img.shape[3]

    hpai_d = {}

    # Get threshold values for each image based on the given percentile val.
    gm_mask = get_hemi_gm_mask(hemi="wb")
    wb_masked = apply_mask(wb_img, gm_mask)
    thr = stats.scoreatpercentile(np.abs(wb_masked), percentile, axis=1)
    reshaped_thr = thr.reshape((n_components, 1))

    # Count the number of voxels above the threshold in each hemisphere.
    # Use only lh_masker to ensure the same size
    hemi_mask = get_hemi_gm_mask(hemi="L")
    masked_r = apply_mask(flip_img_lr(wb_img), hemi_mask)
    masked_l = apply_mask(wb_img, hemi_mask)
    for sign in SPARSITY_SIGNS:
        if sign == "pos":
            voxel_r = np.sum(masked_r > reshaped_thr, axis=1)
            voxel_l = np.sum(masked_l > reshaped_thr, axis=1)
        elif sign == "neg":
            voxel_r = np.sum(masked_r < -reshaped_thr, axis=1)
            voxel_l = np.sum(masked_l < -reshaped_thr, axis=1)
        elif sign == "abs":
            voxel_r = np.sum(np.abs(masked_r) > reshaped_thr, axis=1)
            voxel_l = np.sum(np.abs(masked_l) > reshaped_thr, axis=1)

        hpai_d[sign] = np.divide((voxel_r - voxel_l), (voxel_r + voxel_l).astype(float))

    return hpai_d


def plot_hpai(wb_master, sparsity_threshold, out_dir):
    components = np.unique(wb_master['n_comp'])

    # 1) HPAI-for pos, neg, and abs in wb components
    print "Plotting HPAI of wb components"
    out_path = op.join(out_dir, '1_wb_HPAI.png')

    fh, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(18, 6))
    fh.suptitle("Hemispheric Participation Asymmetry Index for each component", fontsize=16)
    colors = sns.color_palette("Paired", 6)
    hpai_styles = {'pos': (colors[4], colors[5], 'for correlated network'),
                   'neg': (colors[0], colors[1], 'for anti-correlated network'),
                   'abs': (colors[2], colors[3], 'overall')}
    by_comp = wb_master.groupby("n_comp")
    for ax, sign in zip(axes, SPARSITY_SIGNS):
        mean, sd = by_comp.mean()["%sHPAI" % sign], by_comp.std()["%sHPAI" % sign]
        ax.fill_between(components, mean + sd, mean - sd, linewidth=0,
                        facecolor=hpai_styles[sign][0], alpha=0.5)
        size = wb_master['rescaled_vc_%s' % (sign)]
        ax.scatter(wb_master.n_comp, wb_master["%sHPAI" % sign], label=sign,
                   c=hpai_styles[sign][1], s=size, edgecolors="grey")
        ax.plot(components, mean, c=hpai_styles[sign][1])
        ax.set_xlim((0, components[-1] + 5))
        ax.set_ylim((-1, 1))
        ax.set_xticks(components)
        ax.set_ylabel("HPAI((R-L)/(R+L) %s" % (hpai_styles[sign][2]))
    fh.text(0.5, 0.04, "Number of components", ha="center")

    save_and_close(out_path, fh=fh)
