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

import os.path as op

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from textwrap import wrap

from main import do_main_analysis, get_dataset
from nibabel_ext import NiftiImageWithTerms
from nilearn_ext.masking import HemisphereMasker
from nilearn_ext.plotting import save_and_close
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


def analyzeWBimages(n_component, dataset, scoring,
                    memory=Memory(cachedir='nilearn_cache'), **kwargs):
    """
    For each wb component images, get the # of voxels above an arbitrary threshold
    (currently hard-coded as 5e-6) to calculate HPI and SAS and return summary DF.

    HPI(hemisphere participation index): (R-L)/(R+L) for # of voxels above the threshold
    SAS(spatial asymmetry score): dissimilarity score using the choice of scoring (l1,l2, corr)

    Assumes the component image is already computed and saved.
    """
    out_dir = op.join('analysis', dataset)
    THRESH = 0.000005

    # Get ICA images from ica_nii dir--assumes it exists already
    print("Simply loading component images for n_component = %s" % c)
    nii_dir = op.join('ica_nii', dataset, str(n_component))
    img_path = op.join(nii_dir, 'wb_ica_components.nii.gz')
    wb_img = NiftiImageWithTerms.from_filename(img_path)

    # Store HPI and SAS vals in a DF
    summary = pd.DataFrame({"n_comp":[n_component]*n_component})

    # 1) Get # of voxels above the given participation threshold for
    # pos and neg side, and both (abs), for each hemisphere. Use them
    # to calculate HPI: (R-L)/(R+L) for # of voxels.
    hemis = ("R", "L")
    sparsitySigns = ("pos", "neg", "abs")
    sparsityResults = {hemi: getHemiSparsity(wb_img, hemi, memory=memory)
                       for hemi in hemis} # {hemi: (pos_arr, neg_arr, abs_arr)}
    for i, sign in enumerate(sparsitySigns):
        for hemi in hemis:
            summary["%s_%s" % (sign, hemi)] = sparsityResults[hemi][i]
        summary["%sTotal" % sign] = summary["%s_R" % sign]+summary["%s_L" % sign]
        summary["%sHPI" % sign] = ((summary["%s_R" % sign]-summary["%s_L" % sign])
                                  /summary["%sTotal" % sign].astype(float))

    # 2) Get SAS by passing two wb images with hemi labels to compare_aomponents
    imgs = [wb_img] * 2
    score_mat, sign_mat = compare_components(imgs, hemis, scoring)
    # we only care about the diagonal in score_mat (diagonal in sign_mat should all be 1)
    summary["SAS"] = score_mat.diagonal()

    return summary

def plotWBanalysis(n_components, dataset, scoring="test", memory=Memory(cachedir='nilearn_cache'),
                  **kwargs):
    """
    Use analyzeWBimages to get HPI and SAS values for each component in a given set of
    n_components and make a series of plots per n_component, and also make summary plots
    for HPI and SAS acorss the range of n_components given.
    """
    # Save plots in the analysis folder
    out_dir = op.join('ica_imgs', dataset, 'analyses')

    # Initialie master DF to store summary DF returned from analyzeWBimages per
    # specific n_component
    master_df = pd.DataFrame()

    # Loop over n_components
    for c in n_components:
        # Save component-specific images in a subfolder
        comp_outdir = op.join(out_dir, 'component-specific')
        # Get data from analyzeWBimages and append to master df
        summary = analyzeWBimages(c, dataset=dataset, scoring=scoring, memory=memory)
        master_df = master_df.append(summary)

        # Now plot:
        hpi_signs = ['pos', 'neg', 'abs']
        # 1) Relationship between positive and negative HPI
        # set color to be proportional to the symmetry in the sparsity (Pos-Neg/Abs),
        # and set size to be proportional to the total sparsity (Abs)
        color = (summary['posTotal'] - summary['negTotal'])/summary['absTotal']
        size = summary['absTotal']/20.0
        ax = summary.plot.scatter(x='posHPI', y='negHPI', c=color, s=size,
                                  xlim=(-1.1,1.1), ylim=(-1.1,1.1),
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

        out_path = op.join(comp_outdir, "PosNegHPI_%dcomponents.png" % c)
        save_and_close(out_path)

        # 2) Relationship between HPI and SAS
        fh, axes = plt.subplots(1, 3, sharey=True, figsize=(18, 6))
        fh.suptitle("The relationship between HPI values and SAS: "
                    "n_components = %d" % c, fontsize=16)
        hpi_sign_colors = {'pos': 'r', 'neg': 'b', 'abs': 'g'}
        for ax, sign in zip(axes, hpi_signs):
            ax.scatter(summary['%sHPI' % sign], summary['SAS'],
                       c=hpi_sign_colors[sign], s=totals[sign]/20)
            ax.set_xlabel("%s HPI" % sign)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(0,1)
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.yaxis.set_ticks_position('left')
            ax.spines['left'].set_position(('data', 0))
            ax.xaxis.set_ticks_position('bottom')
            ax.spines['bottom'].set_position(('data', 0))
            plt.setp(ax, xticks=ticks, xticklabels=labels)
        fh.text(0.04, 0.5, "Spatial Asymmetry Score", va='center', rotation='vertical')
        out_path = op.join(comp_outdir, "HPIvsSAS_%dcomponents.png" % c)
        save_and_close(out_path)

    # Now plot the HPI and SAS over a range of n_components
    # 1) HPI-for pos, neg, and abs_005
    fh, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(18, 6))
    fh.suptitle("Hemispheric Participation Index for each component", fontsize=16)
    hpi_styles = {'pos': ['r', 'lightpink', 'above %d' % THRESH],
                  'neg': ['b', 'lightblue', 'below -%d' % THRESH],
                  'abs': ['g', 'lightgreen', 'with abs value above %d' % THRESH]}
    by_comp = master_df.groupby("n_comp")
    for ax, sign in zip(axes, hpi_signs):
        mean, sd = by_comp.mean()["%sHPI" % sign], by_comp.std()["%sHPI" % sign]
        ax.fill_between(n_components, mean + sd, mean - sd, linewidth=0,
                        facecolor=hpi_styles[sign][1], alpha=0.5)
        size = master_df['%sTotal' % (sign)]/20.0
        ax.scatter(master_df.n_comp, master_df["%sHPI" % sign], label=sign,
                   c=hpi_styles[sign][0], s=size / 20)
        ax.plot(n_components, mean, c=hpi_styles[sign][0])
        ax.set_xlim((0, n_components[-1] + 5))
        ax.set_ylim((-1, 1))
        ax.set_xticks(n_components)
        ax.set_ylabel("HPI((R-L)/(R+L) for # of voxels %s" % (hpi_styles[sign][2]))
    fh.text(0.5, 0.04, "# of components", ha="center")

    out_path = op.join(out_dir, 'wb_HPI.png')
    save_and_close(out_path, fh=fh)

    # 2) SAS
    fh, ax = plt.subplots(1, 1, figsize=(18, 6))
    fh.suptitle("Spatial Asymmetry Score for each component", fontsize=16)
    sas_mean, sas_sd = by_comp.mean()["SAS"], by_comp.std()["SAS"]
    ax.fill_between(n_components, sas_mean + sas_sd, sas_mean - sas_sd,
                    linewidth=0, facecolor='lightgrey', alpha=0.5)
    size = master_df["absTotal"]/20.0
    ax.scatter(master_df.n_comp, master_df["SAS"], c='grey', s=size)
    ax.plot(n_components, sas_mean, c='grey')
    ax.set_xlim((0, n_components[-1] + 5))
    ax.set_ylim((-1, 1))
    ax.set_xticks(n_components)
    ax.set_ylabel("SAS (higher values indicate asymmetry)")

    out_path = op.join(out_dir, 'wb_SAS.png')
    save_and_close(out_path, fh=fh)


def main_ic_loop(components, scoring,
                 dataset, query_server=True, force=False,
                 memory=Memory(cachedir='nilearn_cache'), **kwargs):
    """
    Loop main.py to plot summaries of WB vs hemi ICA components
    """
    raise NotImplementedError
    ## Below are old codes from ps.py
    match_methods = ['wb', 'rl', 'lr']
    out_dir = op.join('ica_imgs', dataset)
    mean_scores, unmatched = [], []

    # Get the data once.
    images, term_scores = get_dataset(
        dataset, query_server=query_server)

    for match in match_methods:
        print("Plotting results for %s matching method" % match)
        mean_score_d, num_unmatched_d = {}, {}
        for force_match in [False, True]:
            for c in components:
                print("Running analysis with %d components" % c)
                img_d, score_mats_d, sign_mats_d = do_main_analysis(
                    dataset=dataset, images=images, term_scores=term_scores,
                    key=match, force=force, force_match=force_match,
                    n_components=c, scoring=scoring, **kwargs)

                # Get mean dissimilarity scores and number of unmatched for each comparisons
                # in score_mats_d
                for comp in score_mats_d:
                    score_mat, sign_mat = score_mats_d[comp], sign_mats_d[comp]
                    mia, uma = get_match_idx_pair(score_mat, sign_mat)
                    mean_score = score_mat[[mia[0], mia[1]]].mean()
                    n_unmatched = uma.shape[1] if uma is not None else 0
                    # Store values in respective dict
                    score_label = "%s%s" % (" vs ".join(comp), "-forced" if force_match else "")
                    um_label = "unmatched %s%s" % (comp[1], "-forced" if force_match else "")
                    if c == components[0]:
                        mean_score_d[score_label] = [mean_score]
                        num_unmatched_d[um_label] = [n_unmatched]
                    else:
                        mean_score_d[score_label].append(mean_score)
                        num_unmatched_d[um_label].append(n_unmatched)

        # Store vals as df
        ms_df = pd.DataFrame(mean_score_d, index=components)
        um_df = pd.DataFrame(num_unmatched_d, index=components)
        mean_scores.append(ms_df)
        unmatched.append(um_df)
        # Save combined df
        combined = pd.concat([ms_df, um_df], axis=1)
        out = op.join(out_dir, '%s-matching_simscores.csv' % match)
        combined.to_csv(out)

    # We have all the scores for the matching method; now plot.
    fh, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(18, 6))
    fh.suptitle("Average dissimilarity scores for the best-match pairs", fontsize=16)
    labels = ["wb vs R", "wb vs L", "R vs L", "L vs R", "wb vs RL",
              "wb vs R-forced", "wb vs L-forced", "R vs L-forced", "L vs R-forced", "wb vs RL-forced",
              "unmatched R", "unmatched L", "unmatched RL"]
    styles = ["r-", "b-", "m-", "m-", "g-",
              "r:", "b:", "m:", "m:", "g:",
              "r--", "b--", "m--"]
    for i, ax in enumerate(axes):
        ax2 = ax.twinx()
        ms_df, um_df = mean_scores[i], unmatched[i]
        for label, style in zip(labels, styles):
            if label in ms_df.columns:
                ms_df[label].plot(ax=ax, style=style)
            elif label in um_df.columns:
                um_df[label].plot(ax=ax2, style=style)
        ax.legend()
        ax.set_title("%s-matching" % (match_methods[i]))
        ax2.set_ylim(ymax=(um_df.values.max() + 9) // 10 * 10)
        ax2.legend(loc=4)
        ax2.set_ylabel("# of unmatched R- or L- components")
    fh.text(0.5, 0.04, "# of components", ha="center")
    fh.text(0.05, 0.5, "mean %s scores" % scoring, va='center', rotation='vertical')
    fh.text(0.95, 0.5, "# of unmatched R- or L- components", va='center', rotation=-90)

    out_path = op.join(out_dir, '%s_simscores.png' % scoring)
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
    # parser.add_argument('key1', nargs='?', default='R', choices=hemi_choices)
    # parser.add_argument('key2', nargs='?', default='L', choices=hemi_choices)
    parser.add_argument('--noSimScore', action='store_true', default=False)
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
    dataset=args.pop('dataset')

    # If noSimScore, run only image_analyses
    if args.pop('noSimScore'):
        plotWBanalysis(n_components=components, dataset=dataset, **args)
    # Otherwise run loops, followed by the image_analyses
    else:
        main_ic_loop(query_server=query_server,
                 components=components, **args)
        image_analyses(query_server=query_server,
                   components=components, **args)
    plt.show()
