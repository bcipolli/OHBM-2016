# *- encoding: utf-8 -*-
# Author: Ami Tsuchida
# License: BSD
"""
How symmetric are the whole-brain ICA components? How are they similar to
half-brain ICA components?

Calculate the HPI (Hemisphere Participation Index) and the SAS (spatial
asymmetry score: dissimilarity score between R and L) for each whole-brain ICA
component images to show the relationship between the two.

Then for each component, find the best-matching half-brain R&L components,
compare the SAS between them to see how much it increases relative to the
whole-brain SAS. Also compare terms associated with whole-brain and matching
half-brain components.

Do that with a hard loop on the # of components, then
plotting the mean SAS change.
"""

import os
import os.path as op

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from textwrap import wrap

from main import do_main_analysis, get_dataset
from nibabel_ext import NiftiImageWithTerms
from nilearn_ext.masking import HemisphereMasker
from nilearn_ext.plotting import save_and_close, rescale
from nilearn_ext.utils import get_match_idx_pair
from nilearn_ext.decomposition import compare_components
from sklearn.externals.joblib import Memory


def getHemiSparsity(img, hemisphere, threshold=0.000005,
                    memory=Memory(cachedir='nilearn_cache')):
    """
    Calculate sparsity of the image for the given hemisphere. Sparsity here is
    defined as the # of voxels above a given threshold, and will be calculated
    for positive and negative side of the activation, as well as the both side.
    It assumes the values of the img is normalized.
    Returns (pos_arr, neg_arr, abs_arr), with each array containing integer
    values and with the length of the img.
    """
    # Transform img to vector for the specified hemisphere
    hemi_masker = HemisphereMasker(hemisphere=hemisphere, memory=memory).fit()
    hemi_vector = hemi_masker.transform(img)

    pos_arr = (hemi_vector > threshold).sum(axis=1)
    neg_arr = (hemi_vector < -threshold).sum(axis=1)
    abs_arr = (np.abs(hemi_vector) > threshold).sum(axis=1)

    return (pos_arr, neg_arr, abs_arr)


def load_or_generate_summary(images, term_scores, n_components, scoring, dataset,
                             force=False, sparsityThreshold=0.000005,
                             memory=Memory(cachedir='nilearn_cache')):
    """
    For a given n_components, load summary csvs if they already exist, or
    run main.py to get and save necessary summary data required for plotting.

    Returns (wb_summary, R_sparsity, L_sparsity), each of which are DataFrame.
    """
    # Directory to find or save the summary csvs
    out_dir = op.join('ica_imgs', dataset, 'analyses', str(n_components))
    summary_csvs = ["wb_summary.csv", "R_sparsity.csv", "L_sparsity.csv"]

    # If summary data are already saved as csv files, simply load them
    if not force and all([op.exists(op.join(out_dir, csv)) for csv in summary_csvs]):
        print("Loading summary data from %s" % out_dir)
        (wb_summary, R_sparsity, L_sparsity) = (pd.read_csv(op.join(out_dir, csv))
                                                for csv in summary_csvs)

    # Otherwise run main.py and save them as csv files
    else:
        # Initialize summary DFs
        (wb_summary, R_sparsity, L_sparsity) = (pd.DataFrame(
            {"n_comp": [n_components] * n_components}) for i in range(3))
        if not op.exists(out_dir):
            os.makedirs(out_dir)

        # Use wb matching in main analysis to get component images and
        # matching scores
        match_method = 'wb'
        img_d, score_mats_d, sign_mats_d = do_main_analysis(
            dataset=dataset, images=images, term_scores=term_scores,
            key=match_method, force=force, plot=False,
            n_components=n_components, scoring=scoring)

        # 1) Get sparsity for each hemisphere for "wb", "R" and "L" imgs
        hemis = ("R", "L")
        sparsitySigns = ("pos", "neg", "abs")
        # Dict of DF and labels used to get and store Sparsity results
        label_dict = {"wb": (wb_summary, hemis),
                      "R": (R_sparsity, ["R"]),
                      "L": (L_sparsity, ["L"])}
        for key in label_dict:
            (df, labels) = label_dict[key]
            sparsityResults = {label: getHemiSparsity(img_d[key], label,
                               threshold=sparsityThreshold, memory=memory)
                               for label in labels}  # {label: (pos_arr, neg_arr, abs_arr)}

            for i, sign in enumerate(sparsitySigns):
                for label in labels:
                    df["%s_%s" % (sign, label)] = sparsityResults[label][i]
                # For wb only, also compute Total sparsity and HPI
                if key == "wb":
                    df["%sTotal" % sign] = df["%s_R" % sign] + df["%s_L" % sign]
                    df["%sHPI" % sign] = ((df["%s_R" % sign] - df["%s_L" % sign]) /
                                          df["%sTotal" % sign].astype(float))

        # Save R/L_sparsity DFs
        R_sparsity.to_csv(op.join(out_dir, "R_sparsity.csv"))
        L_sparsity.to_csv(op.join(out_dir, "L_sparsity.csv"))

        # 2) Get SAS of wb component images as well as matched RL images by passing
        # 2 x wb or RL images and hemi labels to the compare_components (make sure
        # not to flip when comparing R and L)
        name_img_pairs = [("wb_SAS", img_d["wb"]),
                          ("matchedRL_SAS", img_d["RL"])]
        for (name, img) in name_img_pairs:
            sas_imgs = [img] * 2
            score_mat, sign_mat = compare_components(sas_imgs, hemis, scoring,
                                                     flip=False)
            # we only care about the diagonal in score_mat
            wb_summary[name] = score_mat.diagonal()

        # 3) Finally store indices of matched R, L, and RL components, and the
        # respective match scores against wb
        comparisons = [('wb', 'R'), ('wb', 'L'), ('wb', 'RL')]
        for comparison in comparisons:
            score_mat, sign_mat = score_mats_d[comparison], sign_mats_d[comparison]
            matched, unmatched = get_match_idx_pair(score_mat, sign_mat)
            # Component indices for matched R, L , RL are in matched[1].
            # Multiply it by matched[2], which stores sign flipping info.
            matched_indices = matched[1] * matched[2]
            wb_summary["matched%s" % comparison[1]] = matched_indices

            matched_scores = score_mat[matched[0], matched[1]]
            wb_summary["match%s_score" % comparison[1]] = matched_scores

            # Save wb_summary
            wb_summary.to_csv(op.join(out_dir, "wb_summary.csv"))

    return (wb_summary, R_sparsity, L_sparsity)


def loop_main_and_plot(components, scoring, dataset, query_server=True,
                       force=False, sparsityThreshold=0.000005,
                       memory=Memory(cachedir='nilearn_cache'), **kwargs):
    """
    Loop main.py to plot summaries of WB vs hemi ICA components
    """
    out_dir = op.join('ica_imgs', dataset, 'analyses')

    # Get the data once.
    # images, term_scores = get_dataset(dataset, max_images=200,
    #                                   query_server=query_server)
    images = None  # for testing
    term_scores = None
    # Initialize master DFs
    (wb_master, R_master, L_master) = (pd.DataFrame() for i in range(3))

    for c in components:
        print("Running analysis with %d components" % c)
        (wb_summary, R_sparsity, L_sparsity) = load_or_generate_summary(
            images=images, term_scores=term_scores, n_components=c,
            scoring=scoring, dataset=dataset, force=force,
            sparsityThreshold=sparsityThreshold, memory=memory)
        # Append them to master DFs
        wb_master = wb_master.append(wb_summary)
        R_master = R_master.append(R_sparsity)
        L_master = L_master.append(L_sparsity)

        ### Generate component-specific plots ###
        # Save component-specific images in the component dir
        comp_outdir = op.join(out_dir, str(c))

        # 1) Relationship between positive and negative HPI in wb components
        out_path = op.join(comp_outdir, "1_PosNegHPI_%dcomponents.png" % c)

        hpi_signs = ['pos', 'neg', 'abs']
        # set color to be proportional to the symmetry in the sparsity (Pos-Neg/Abs),
        # and set size to be proportional to the total sparsity (Abs)
        color = (wb_summary['posTotal'] - wb_summary['negTotal']) / wb_summary['absTotal']
        size = rescale(wb_summary['absTotal'])
        ax = wb_summary.plot.scatter(x='posHPI', y='negHPI', c=color, s=size,
                                     xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), edgecolors="grey",
                                     colormap='Reds', colorbar=True, figsize=(7, 6))
        title = ax.set_title("\n".join(wrap("The relationship between HPI on "
                                            "positive and negative side: "
                                            "n_components = %d" % c, 60)))
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.spines['left'].set_position(('data', 0))
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('data', 0))
        ticks = [-1.1, -1.0, -0.5, 0, 0.5, 1.0, 1.1]
        labels = ['L', '-1.0', '-0.5', '0', '0.5', '1.0', 'R']
        plt.setp(ax, xticks=ticks, xticklabels=labels, yticks=ticks, yticklabels=labels)
        f = plt.gcf()
        title.set_y(1.05)
        f.subplots_adjust(top=0.8)
        cax = f.get_axes()[1]
        cax.set_ylabel('Balance between pos/neg(anti-correlated network)',
                       rotation=270, labelpad=20)

        save_and_close(out_path)

        # 2) Relationship between HPI and SAS in wb components
        out_path = op.join(comp_outdir, "2_HPIvsSAS_%dcomponents.png" % c)

        fh, axes = plt.subplots(1, 3, sharey=True, figsize=(18, 6))
        fh.suptitle("The relationship between HPI values and SAS: "
                    "n_components = %d" % c, fontsize=16)
        hpi_sign_colors = {'pos': 'r', 'neg': 'b', 'abs': 'g'}
        for ax, sign in zip(axes, hpi_signs):
            size = rescale(wb_summary['%sTotal' % sign]) * 2
            ax.scatter(wb_summary['%sHPI' % sign], wb_summary['wb_SAS'],
                       c=hpi_sign_colors[sign], s=size, edgecolors="grey")
            ax.set_xlabel("%s HPI" % sign)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(0, 1)
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.yaxis.set_ticks_position('left')
            ax.spines['left'].set_position(('data', 0))
            ax.xaxis.set_ticks_position('bottom')
            ax.spines['bottom'].set_position(('data', 0))
            plt.setp(ax, xticks=ticks, xticklabels=labels)
        fh.text(0.04, 0.5, "Spatial Asymmetry Score", va='center', rotation='vertical')

        save_and_close(out_path)

    ### Generate plots over a range of specified n_components ###
    # 1) HPI-for pos, neg, and abs in wb components
    out_path = op.join(out_dir, '1_wb_HPI.png')

    fh, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(18, 6))
    fh.suptitle("Hemispheric Participation Index for each component", fontsize=16)
    hpi_styles = {'pos': ['r', 'lightpink', 'above %d' % sparsityThreshold],
                  'neg': ['b', 'lightblue', 'below -%d' % sparsityThreshold],
                  'abs': ['g', 'lightgreen', 'with abs value above %d' % sparsityThreshold]}
    by_comp = wb_master.groupby("n_comp")
    for ax, sign in zip(axes, hpi_signs):
        mean, sd = by_comp.mean()["%sHPI" % sign], by_comp.std()["%sHPI" % sign]
        ax.fill_between(components, mean + sd, mean - sd, linewidth=0,
                        facecolor=hpi_styles[sign][1], alpha=0.5)
        size = rescale(wb_master['%sTotal' % (sign)])
        ax.scatter(wb_master.n_comp, wb_master["%sHPI" % sign], label=sign,
                   c=hpi_styles[sign][0], s=size, edgecolors="grey")
        ax.plot(components, mean, c=hpi_styles[sign][0])
        ax.set_xlim((0, components[-1] + 5))
        ax.set_ylim((-1, 1))
        ax.set_xticks(components)
        ax.set_ylabel("HPI((R-L)/(R+L) for # of voxels %s" % (hpi_styles[sign][2]))
    fh.text(0.5, 0.04, "# of components", ha="center")

    save_and_close(out_path, fh=fh)

    # 2) SAS for wb components
    fh, ax = plt.subplots(1, 1, figsize=(18, 6))
    fh.suptitle("Spatial Asymmetry Score for each component", fontsize=16)
    sas_mean, sas_sd = by_comp.mean()["wb_SAS"], by_comp.std()["wb_SAS"]
    ax.fill_between(components, sas_mean + sas_sd, sas_mean - sas_sd,
                    linewidth=0, facecolor='lightgrey', alpha=0.5)
    size = rescale(wb_master["absTotal"])
    ax.scatter(wb_master.n_comp, wb_master["wb_SAS"], c='grey', s=size, edgecolors="black")
    ax.plot(components, sas_mean, c='grey')
    ax.set_xlim((0, components[-1] + 5))
    ax.set_ylim((-1, 1))
    ax.set_xticks(components)
    ax.set_ylabel("SAS (higher values indicate asymmetry)")

    out_path = op.join(out_dir, '2_wb_SAS.png')
    save_and_close(out_path, fh=fh)


if __name__ == '__main__':
    import warnings
    from argparse import ArgumentParser

    # Look for image computation errors
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('error', RuntimeWarning)  # Detect bad NV images

    # Arg parsing
    hemi_choices = ['R', 'L', 'wb']
    parser = ArgumentParser(description="Really?")
    parser.add_argument('--force', action='store_true', default=False)
    parser.add_argument('--offline', action='store_true', default=False)
    parser.add_argument('--components', nargs='?',
                        default="5,10,15,20,25,30,35,40,45,50")
    parser.add_argument('--dataset', nargs='?', default='neurovault',
                        choices=['neurovault', 'abide', 'nyu'])
    parser.add_argument('--seed', nargs='?', type=int, default=42,
                        dest='random_state')
    parser.add_argument('--scoring', nargs='?', default='correlation',
                        choices=['l1norm', 'l2norm', 'correlation'])
    args = vars(parser.parse_args())

    # Alias args
    query_server = not args.pop('offline')
    # keys = args.pop('key1'), args.pop('key2')
    components = [int(c) for c in args.pop('components').split(',')]
    dataset = args.pop('dataset')

    loop_main_and_plot(components=components, dataset=dataset,
                       query_server=query_server, **args)
