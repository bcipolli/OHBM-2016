# *- encoding: utf-8 -*-
# Author: Ami Tsuchida
# License: BSD
"""
"""

import os
import os.path as op

import pandas as pd
from sklearn.externals.joblib import Memory

from .acni import calculate_acni
from .hpai import calculate_hpai
from .match import do_match_analysis
from .sparsity import get_hemi_sparsity, SPARSITY_SIGNS
from nilearn_ext.decomposition import compare_RL
from nilearn_ext.utils import get_match_idx_pair


def load_or_generate_summary(images, term_scores, n_components, scoring, dataset,
                             sparsity_threshold, acni_percentile=95.0, hpai_percentile=95.0,
                             force=False, plot=True, out_dir=None,
                             memory=Memory(cachedir='nilearn_cache')):
    """
    For a given n_components, load summary csvs if they already exist, or
    run main.py to get and save necessary summary data required for plotting.

    Returns (wb_summary, R_summary, L_summary), each of which are DataFrame.
    """
    # Directory to find or save the summary csvs
    out_dir = out_dir or op.join('ica_imgs', dataset, 'analyses', str(n_components))
    summary_csvs = ["wb_summary.csv", "R_summary.csv", "L_summary.csv"]

    # If summary data are already saved as csv files, simply load them
    if not force and all([op.exists(op.join(out_dir, csv)) for csv in summary_csvs]):
        (wb_summary, R_summary, L_summary) = (pd.read_csv(op.join(out_dir, csv))
                                              for csv in summary_csvs)

    # Otherwise run match analysis and save them as csv files
    else:
        # Initialize summary DFs
        (wb_summary, R_summary, L_summary) = (pd.DataFrame(
            {"n_comp": [n_components] * n_components}) for i in range(3))
        if not op.exists(out_dir):
            os.makedirs(out_dir)

        # Use wb matching in match analysis to get component images and
        # matching scores
        match_method = 'wb'
        img_d, score_mats_d, sign_mats_d = do_match_analysis(
            dataset=dataset, images=images, term_scores=term_scores,
            key=match_method, force=False, plot=plot,
            plot_dir=out_dir, n_components=n_components, scoring=scoring)

        # 1) For each of "wb", "R", and "L" image, get sparsity and ACNI
        # (Anti-Correlated Network index). For "wb", also get HPAI
        # (Hemispheric participation asymmetry index).
        hemis = ("R", "L", "wb")
        sparsityTypes = ("l1", "vc-pos", "vc-neg", "vc-abs")

        # Dict of DF and labels used to get and store results
        label_dict = {"wb": (wb_summary, hemis),
                      "R": (R_summary, ["R"]),
                      "L": (L_summary, ["L"])}

        for key in label_dict:
            (df, labels) = label_dict[key]

            # 1-1) Sparsity
            # sparsity_results = {label: sparsity_dict}
            sparsity_results = {label: get_hemi_sparsity(img_d[key], label,
                                thr=sparsity_threshold) for label in labels}

            for s_type in sparsityTypes:
                for label in labels:
                    df["%s_%s" % (s_type, label)] = sparsity_results[label][s_type]

            # 1-2) ACNI
            for label in labels:
                df["ACNI_%s" % label] = calculate_acni(
                    img_d[key], hemi=label, percentile=acni_percentile)

            # 1-3) For wb only, also compute HPAI
            if key == "wb":
                hpai_d = calculate_hpai(img_d[key], percentile=hpai_percentile)
                for sign in SPARSITY_SIGNS:
                    df["%sHPAI" % sign] = hpai_d[sign]

        # Save R/L_summary DFs
        R_summary.to_csv(op.join(out_dir, "R_summary.csv"))
        L_summary.to_csv(op.join(out_dir, "L_summary.csv"))

        # 2) Get SSS of wb component images as well as matched RL images
        col_img_pairs = [("wb_SSS", img_d["wb"]),
                         ("matchedRL_SSS", img_d["RL-unforced"])]
        for (col, img) in col_img_pairs:
            score_arr = compare_RL(img)
            wb_summary[col] = score_arr

        # 3) Finally store indices of matched R, L, and RL components, and the
        # respective match scores against wb
        comparisons = [('wb', 'R'), ('wb', 'L'), ('wb', 'RL-unforced')]
        for comparison in comparisons:
            score_mat, sign_mat = score_mats_d[comparison], sign_mats_d[comparison]
            matched, unmatched = get_match_idx_pair(score_mat, sign_mat)
            # Add '-unforced' in the match name since this is not forcing one-to-one match
            suf = '' if 'unforced' in comparison[1] else '-unforced'
            match_name = comparison[1] + suf
            # Component indices for matched R, L , RL are in matched["idx"][1].
            # Signs are in matched["sign"][1]
            wb_summary["matched%s_idx" % match_name] = matched["idx"][1]
            wb_summary["matched%s_sign" % match_name] = matched["sign"][1]

            matched_scores = score_mat[matched["idx"][0], matched["idx"][1]]
            wb_summary["match%s_score" % match_name] = matched_scores
            num_unmatched = unmatched["idx"].shape[1] if unmatched["idx"] is not None else 0
            wb_summary["n_unmatched%s" % comparison[1]] = num_unmatched

            # Save wb_summary
            wb_summary.to_csv(op.join(out_dir, "wb_summary.csv"))

    return (wb_summary, R_summary, L_summary)
