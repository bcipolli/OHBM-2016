# *- encoding: utf-8 -*-
# Author: Ami Tsuchida
# License: BSD
"""
"""

import os.path as op

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn.masking import apply_mask
from scipy import stats
from textwrap import wrap

from nilearn_ext.masking import get_hemi_gm_mask
from nilearn_ext.plotting import save_and_close


def calculate_acni(img, hemi, percentile=95.0):
    """
    For each component image in the give ICA image, calculate Anti-Correlated
    Network Index (ACNI), which is simply a proportion of negative activation
    out of all the voxels whose magnitude is above a given percentile value.

    i.e. the component with value close to 0.5 has strong ACN, with positive and
    negative side of the activation equally balanced, while a value
    close to 0 indicates the component has very little ACN.

    Returns an array of length equal to the n_component of the given image.
    """
    n_components = img.shape[3]

    # Get threshold values for each image based on the given percentile val.
    gm_mask = get_hemi_gm_mask(hemi=hemi)
    masked = apply_mask(img, gm_mask)
    thr = stats.scoreatpercentile(np.abs(masked), percentile, axis=1)
    reshaped_thr = thr.reshape((n_components, 1))

    neg_voxels = np.sum(masked < -reshaped_thr, axis=1)
    abs_voxels = np.sum(np.abs(masked) > reshaped_thr, axis=1)

    acni = np.divide(neg_voxels, abs_voxels.astype(float))

    return acni


def plot_acni(wb_master, R_master, L_master, out_dir):
    # 6) Plot ACNI for wb and hemi-components
    print "Generating plots of ACNI for wb and hemi-components"
    set2 = sns.color_palette("Set2")
    palette = [set2[2], set2[0], set2[1]]
    # Prepare ACNI for wb and hemi-components
    acni_cols = ["n_comp", "ACNI", "decomposition_type"]
    acni_summary = pd.DataFrame(columns=acni_cols)
    hemis = ("wb", "R", "L")
    master_DFs = (wb_master, R_master, L_master)
    for hemi, df in zip(hemis, master_DFs):
        acni = df[["n_comp", "ACNI_%s" % hemi]]
        acni["decomposition_type"] = hemi
        acni.columns = acni_cols
        acni_summary = acni_summary.append(acni)
    acni_summary["n_comp"] = acni_summary.n_comp.astype(int)
    out_path = op.join(out_dir, "6_ACNI_comparison.png")

    fh = plt.figure(figsize=(10, 6))
    ax = fh.gca()
    title = "\n".join(wrap("Anti-Correlated Network Index of each component: "
                           "Comparison of WB and R- or L-only decomposition", 60))
    plt.title(title, fontsize=16)
    sns.boxplot(x="n_comp", y="ACNI", ax=ax, hue="decomposition_type",
                data=acni_summary, palette=palette)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Proportion of Anti-Correlated Network")

    save_and_close(out_path, fh=fh)
