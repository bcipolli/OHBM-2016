# *- encoding: utf-8 -*-
# Author: Ami Tsuchida
# License: BSD
"""
"""
import os.path as op

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nilearn.image import iter_img
from nilearn.masking import apply_mask
from scipy import stats

from nilearn_ext.masking import get_mask_by_key
from nilearn_ext.plotting import save_and_close


SPARSITY_SIGNS = ['pos', 'neg', 'abs']


def get_sparsity_threshold(images, global_percentile=99.9):
    """
    Given the list of images, get global (across images) sparsity threshold
    using the specified percentile values.

    The global_percentile for each image in each component are obtained,
    and the minimum value is returned.
    """
    global_thr = []
    for image in images:
        g_thr = []
        for component_img in iter_img(image):
            dat = component_img.get_data()
            nonzero_dat = dat[np.nonzero(dat)]
            g = stats.scoreatpercentile(np.abs(nonzero_dat), global_percentile)
            g_thr.append(g)
        global_thr.append(min(g_thr))
    thr = min(global_thr)

    return thr


def get_hemi_sparsity(img, hemi, thr=0.000005):
    """
    Calculate sparsity of the image for the given hemisphere.
    Sparsity is calculated using 1) l1norm ("l1") value of the image, and
    2) voxel count ("vc") for # of voxels above a threshold.

    The vc method is calculated separately for pos, neg side of the image
    and for absolute values, to detect any anti-correlated netowrks.

    It assumes the values of the img is normalized.

    Returns a dict containing arrays for l1, vc-pos, vc-neg, vc-abs, each
    1-vector array with the length (n_component) of the img.
    The dict also contains n_voxels for the given hemi.
    """
    # Transform img to vector for the specified hemisphere
    gm_mask = get_mask_by_key(hemi)
    masked = apply_mask(img, gm_mask)
    sparsity_dict = {}
    sparsity_dict["l1"] = np.linalg.norm(masked, axis=1, ord=1)
    sparsity_dict["vc-pos"] = (masked > thr).sum(axis=1)
    sparsity_dict["vc-neg"] = (masked < -thr).sum(axis=1)
    sparsity_dict["vc-abs"] = (np.abs(masked) > thr).sum(axis=1)

    return sparsity_dict


def plot_sparsity(wb_master, R_master, L_master, out_dir):
    # 2) VC and 3) L1 Sparsity comparison between wb and hemi components
    print "Plotting sparsity for WB and hemi-components"
    pastel2 = sns.color_palette("Pastel2")
    set2 = sns.color_palette("Set2")
    hemi_colors = {"R": [set2[2], pastel2[2]], "L": [set2[0], pastel2[0]]}
    # Prepare summary of sparsity for each hemisphere
    for hemi, hemi_df in zip(("R", "L"), (R_master, L_master)):
        wb_sparsity = wb_master[hemi_df.columns]
        wb_sparsity["decomposition_type"] = "wb"
        hemi_df["decomposition_type"] = hemi
        sparsity_summary = wb_sparsity.append(hemi_df)

        # First plot voxel count sparsity
        out_path = op.join(out_dir, '2_vcSparsity_comparison_%s.png' % hemi)

        fh, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(18, 6))
        fh.suptitle("Voxel Count Sparsity of each component: Comparison of WB "
                    "and %s-only decomposition" % hemi, fontsize=16)
        colors = sns.color_palette("Paired", 6)
        sparsity_styles = {'pos': [colors[5], colors[4]],
                           'neg': [colors[1], colors[0]],
                           'abs': [colors[3], colors[2]]}
        for ax, sign in zip(axes, SPARSITY_SIGNS):
            sns.boxplot(x="n_comp", y="vc-%s_%s" % (sign, hemi), ax=ax,
                        hue="decomposition_type", data=sparsity_summary,
                        palette=sparsity_styles[sign])
            ax.set_title("%s" % sign)
        fh.text(0.04, 0.5, "Voxel Count Sparsity values", va='center',
                rotation='vertical')
        fh.text(0.5, 0.04, "Number of components", ha="center")

        save_and_close(out_path, fh=fh)

        # Next L1 norm sparsity
        out_path = op.join(out_dir, '3_l1Sparsity_comparison_%s.png' % hemi)

        fh = plt.figure(figsize=(10, 6))
        ax = fh.gca()
        plt.title("L1 Sparsity of each component: Comparison of WB "
                  "and %s-only decomposition" % hemi, fontsize=16)
        sns.boxplot(x="n_comp", y="l1_%s" % hemi, ax=ax,
                    hue="decomposition_type", data=sparsity_summary,
                    palette=hemi_colors[hemi])
        ax.set_xlabel("Number of components")
        ax.set_ylabel("L1 sparsity values")

        save_and_close(out_path, fh=fh)
