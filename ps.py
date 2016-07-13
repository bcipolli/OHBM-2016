# *- encoding: utf-8 -*-
# Author: Ben Cipollini
# License: BSD
"""
How many ics? Find the sweet spot where the
average dissimilarity between unilateral ('RL') and
bilateral ('wb') components is minimized.

Do that with a hard loop on the # of components, then
plotting the mean dissimilarity (derived from the
score matrix).
"""

import os.path as op

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

from main import do_main_analysis, get_dataset
from nibabel_ext import NiftiImageWithTerms
from nilearn_ext.masking import HemisphereMasker
from nilearn_ext.plotting import save_and_close
from nilearn_ext.utils import get_match_idx_pair
from sklearn.externals.joblib import Memory


def image_analyses(components, dataset, memory=Memory(cachedir='nilearn_cache'),
                   **kwargs):
    """
    1) Plot sparsity of ICA images for wb, R, and L.
    2) Plot Hemispheric Participation Index (HPI) for wb ICA images
    """
    out_dir = op.join('ica_imgs', dataset)
    images_key = ["R", "L", "wb"]
    sparsity_levels = ['pos_005', 'neg_005', 'abs_005']

    # For calculating hemispheric participation index (HPI) from wb components,
    # prepare hemisphere maskers
    hemi_maskers = [HemisphereMasker(hemisphere=hemi, memory=memory).fit()
                    for hemi in ['R', 'L']]

    # Store sparsity (and hpi for wb) vals in a DF
    columns = ["n_comp"] + sparsity_levels
    wb_columns = columns + ["pos_hpi", "neg_hpi"]
    hemi_dfs = {hemi: pd.DataFrame(columns=wb_columns if hemi == "wb" else columns)
                for hemi in images_key}

    # Loop over components
    for c in components:
        print("Simply loading component images for n_component = %s" % c)
        nii_dir = op.join('ica_nii', dataset, str(c))
        for hemi in images_key:
            img_path = op.join(nii_dir, '%s_ica_components.nii.gz' % (hemi))
            img = NiftiImageWithTerms.from_filename(img_path)
            data = pd.DataFrame({"n_comp": [c] * c}, columns=columns)
            # get mean sparsity for the ica iamge and store in sparsity dict
            for s in sparsity_levels:
                thresh = float('0.%s' % (re.findall('\d+', s)[0]))
                # sparsity is # of voxels above the given sparsity level for each component
                if 'pos' in s:
                    data[s] = (img.get_data() > thresh).sum(axis=0).sum(axis=0).sum(axis=0)
                elif 'neg' in s:
                    data[s] = (img.get_data() < -thresh).sum(axis=0).sum(axis=0).sum(axis=0)
                elif 'abs' in s:
                    data[s] = (abs(img.get_data()) > thresh).sum(axis=0).sum(axis=0).sum(axis=0)

            # get hpi values for wb components
            if hemi == "wb":
                hemi_vectors = [masker.transform(img) for masker in hemi_maskers]
                # transform back so that values for each component can be calculated
                hemi_imgs = [masker.inverse_transform(vec) for masker, vec in
                             zip(hemi_maskers, hemi_vectors)]
                # pos/neg_vals[0] = # voxels in R, pos/neg_vals[1] = # voxels in L
                pos_vals = [(hemi_img.get_data() > 0.005).sum(axis=0).sum(axis=0).sum(axis=0)
                            for hemi_img in hemi_imgs]
                neg_vals = [(hemi_img.get_data() < -0.005).sum(axis=0).sum(axis=0).sum(axis=0)
                            for hemi_img in hemi_imgs]

                for sign, val in zip(['pos', 'neg'], [pos_vals, neg_vals]):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        # pos/neg HPI vals, calculated as (R-L)/(R+L) for num. of voxels above
                        # the given threshold
                        hpi = (val[0].astype(float) - val[1]) / (val[0] + val[1])
                    data["%s_hpi" % (sign)] = hpi

            hemi_dfs[hemi] = hemi_dfs[hemi].append(data)

    # Now plot:
    # 1) Sparsity for wb, R and L ICA images
    fh, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(18, 6))
    sparsity_styles = {'pos_005': ['b', 'lightblue'],
                       'neg_005': ['r', 'lightpink'],
                       'abs_005': ['g', 'lightgreen']}
    for ax, hemi in zip(axes, images_key):
        df = hemi_dfs[hemi]
        by_comp = df.groupby('n_comp')
        for s in sparsity_levels:
            mean, sd = by_comp.mean()[s], by_comp.std()[s]
            ax.fill_between(components, mean + sd, mean - sd, linewidth=0,
                            facecolor=sparsity_styles[s][1], alpha=0.5)
            ax.plot(components, mean, color=sparsity_styles[s][0], label=s)
        # Overlay individual points for absolute threshold
        ax.scatter(df.n_comp, df.abs_005, c=sparsity_styles['abs_005'][0])
        ax.set_title("Sparsity of the %s components" % (hemi))
        ax.set_xlim(xmin=components[0] - 1, xmax=components[-1] + 1)
        ax.set_xticks(components)
    plt.legend()
    fh.text(0.5, 0.04, "# of components", ha="center")
    fh.text(0.04, 0.5, "# of voxels above the threshold", va='center', rotation='vertical')

    out_path = op.join(out_dir, 'sparsity.png')
    save_and_close(out_path, fh=fh)

    # 2) HPI plot for wb components
    fh, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))
    fh.suptitle("Hemispheric Participation Index for each component", fontsize=16)
    hpi_styles = {'pos': ['b', 'lightblue', 'above 0.005'],
                  'neg': ['r', 'lightpink', 'below -0.005']}
    df = hemi_dfs["wb"]
    by_comp = df.groupby("n_comp")
    for ax, sign in zip(axes, ['pos', 'neg']):
        mean, sd = by_comp.mean()["%s_hpi" % sign], by_comp.std()["%s_hpi" % sign]
        ax.fill_between(components, mean + sd, mean - sd, linewidth=0,
                        facecolor=hpi_styles[sign][1], alpha=0.5)
        size = df['%s_005' % (sign)]
        ax.scatter(df.n_comp, df["%s_hpi" % sign], label=sign, c=hpi_styles[sign][0], s=size / 20)
        ax.plot(components, mean, c=hpi_styles[sign][0])
        ax.set_title("%s" % (sign))
        ax.set_xlim((0, components[-1] + 5))
        ax.set_ylim((-1, 1))
        ax.set_xticks(components)
        ax.set_ylabel("HPI((R-L)/(R+L) for # of voxels %s" % (hpi_styles[sign][2]))

    fh.text(0.5, 0.04, "# of components", ha="center")

    out_path = op.join(out_dir, 'wb_HPI.png')
    save_and_close(out_path, fh=fh)

    # Save sparsity and HPI vals for all the components
    for hemi in images_key:
        hemi_dfs[hemi].index.name = "idx"
        hemi_dfs[hemi].to_csv(op.join(out_dir, "%s_summary.csv" % (hemi)))


def main_ic_loop(components, scoring,
                 dataset, query_server=True, force=False,
                 memory=Memory(cachedir='nilearn_cache'), **kwargs):
    # Test with just 'wb' and 'rl' matching until 'lr' matching is fixed
    # match_methods = ['wb', 'rl', 'lr']
    match_methods = ['wb', 'rl']
    out_dir = op.join('ica_imgs', dataset)
    mean_scores, unmatched = [], []

    # Get the data once.
    images, term_scores = get_dataset(
        dataset, query_server=query_server)

    for match_method in match_methods:
        print("Plotting results for %s matching method" % match_method)
        mean_score_d, num_unmatched_d = {}, {}
        for c in components:
            print("Running analysis with %d components" % c)
            # main analysis is run for each component and match method:
            # plotting for component comparisons are done only if force=True
            img_d, score_mats_d, sign_mats_d = do_main_analysis(
                    dataset=dataset, images=images, term_scores=term_scores,
                    key=match_method, force=force, plot=force,
                    n_components=c, scoring=scoring, **kwargs)

            # Get mean dissimilarity scores and number of unmatched for each comparisons
            # in score_mats_d
            for comp in score_mats_d:
                score_mat, sign_mat = score_mats_d[comp], sign_mats_d[comp]
                # For ("wb", "RL-forced") and ("wb", "RL-unforced")
                if "forced" in comp[1]:
                    if "-forced" in comp[1]:
                        match, unmatch = get_match_idx_pair(score_mat, sign_mat, force=True)
                    elif "-unforced" in comp[1]:
                        match, unmatch = get_match_idx_pair(score_mat, sign_mat, force=False)
                        n_unmatched = unmatch["idx"].shape[1] if unmatch["idx"] is not None else 0
                        um_label = "unmatched RL"
                    mean_score = score_mat[[match["idx"][0], match["idx"][1]]].mean()
                    score_label = "%s" % (" vs ".join(comp))
                    # Store values in respective dict
                    if c == components[0]:
                        mean_score_d[score_label] = [mean_score]
                        if "-unforced" in comp[1]:
                            num_unmatched_d[um_label] = [n_unmatched]
                    else:
                        mean_score_d[score_label].append(mean_score)
                        if "-unforced" in comp[1]:
                            num_unmatched_d[um_label].append(n_unmatched)
                
                # For ("wb", "R"), ("wb", "L") --wb matching or ("R", "L") --rl matching 
                else:
                    for force_match in [True, False]:
                        match, unmatch = get_match_idx_pair(score_mat, sign_mat, force=force_match)
                        mean_score = score_mat[[match["idx"][0], match["idx"][1]]].mean()
                        if force_match:
                            score_label = "%s%s" % (" vs ".join(comp), "-forced")
                            n_unmatched = None
                        else:
                            score_label = "%s%s" % (" vs ".join(comp), "-unforced")
                            n_unmatched = unmatch["idx"].shape[1] if unmatch["idx"] is not None else 0
                            um_label = "unmatched %s" % comp[1]
                        # Store values in respective dict
                        if c == components[0]:
                            mean_score_d[score_label] = [mean_score]
                            if not force_match:
                                num_unmatched_d[um_label] = [n_unmatched]
                        else:
                            mean_score_d[score_label].append(mean_score)
                            if not force_match:
                                num_unmatched_d[um_label].append(n_unmatched)

        # Store vals as df
        ms_df = pd.DataFrame(mean_score_d, index=components)
        um_df = pd.DataFrame(num_unmatched_d, index=components)
        mean_scores.append(ms_df)
        unmatched.append(um_df)
        # Save combined df
        combined = pd.concat([ms_df, um_df], axis=1)
        out = op.join(out_dir, '%s-matching_simscores.csv' % match_method)
        combined.to_csv(out)

    # We have all the scores for the matching method; now plot.
    fh, axes = plt.subplots(1, len(match_methods), sharex=True, sharey=True, figsize=(18, 6))
    fh.suptitle("Average dissimilarity scores for the best-match pairs", fontsize=16)
    labels = ["wb vs R-unforced", "wb vs L-unforced", "R vs L-unforced", "wb vs RL-unforced",
              "wb vs R-forced", "wb vs L-forced", "R vs L-forced", "wb vs RL-forced",
              "unmatched R", "unmatched L", "unmatched RL"]
    styles = ["r-", "b-", "m-", "g-",
              "r:", "b:", "m:", "g:",
              "r--", "b--", "m--"]

    for i, ax in enumerate(axes):
        ax2 = ax.twinx()
        ms_df, um_df = mean_scores[i], unmatched[i]
        for label, style in zip(labels, styles):
            if label in ms_df.columns:
                ms_df[label].plot(ax=ax, style=style)
            elif label in um_df.columns:
                um_df[label].plot(ax=ax2, style=style)
        ax.set_title("%s-matching" % (match_methods[i]))
        # Shrink current axis by 30%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        ax2.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        # Put the legends to the right of the current axis
        ax.legend(loc='lower left', bbox_to_anchor=(1.3, 0.5))
        ax2.legend(loc='upper left', bbox_to_anchor=(1.3, 0.5))
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
    parser.add_argument('--scoring', nargs='?', default='l1norm',
                        choices=['l1norm', 'l2norm', 'correlation'])
    args = vars(parser.parse_args())

    # Alias args
    query_server = not args.pop('offline')
    # keys = args.pop('key1'), args.pop('key2')
    components = [int(c) for c in args.pop('components').split(',')]

    # If noSimScore, run only image_analyses
    if args.pop('noSimScore'):
        image_analyses(query_server=query_server,
                       components=components, **args)
    # Otherwise run loops, followed by the image_analyses
    main_ic_loop(query_server=query_server,
                 components=components, **args)
    image_analyses(query_server=query_server,
                   components=components, **args)
    plt.show()
