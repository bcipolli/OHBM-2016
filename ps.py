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

from main import main
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
    # For each component type, store sparsity values for each sparsity level
    sparsity = {hemi: {s: [] for s in sparsity_levels} for hemi in images_key}

    # For calculating hemispheric participation index (HPI) from wb components,
    # prepare hemisphere maskers and hpi_vals dict to store hpi values
    hemi_maskers = [HemisphereMasker(hemisphere=hemi, memory=memory).fit() for hemi in ['R', 'L']]
    hpi_vals = {sign: {val: [] for val in ['vals', 'mean', 'sd']} for sign in ['pos', 'neg']}

    # Loop over components
    for c in components:
        print("Simply loading component images for n_component = %s" % c)
        nii_dir = op.join('ica_nii', dataset, str(c))
        for hemi in images_key:
            img_path = op.join(nii_dir, '%s_ica_components.nii.gz' % (hemi))
            img = NiftiImageWithTerms.from_filename(img_path)

            # get mean sparsity for the ica iamge and store in sparsity dict
            for s in sparsity_levels:
                thresh = float('0.%s' % (re.findall('\d+', s)[0]))
                # sparsity_vals is a list containing # of voxels above the given sparsity level
                # for each component
                if 'pos' in s:
                    sparsity_vals = (img.get_data() > thresh).sum(axis=0).sum(axis=0).sum(axis=0)
                elif 'neg' in s:
                    sparsity_vals = (img.get_data() < -thresh).sum(axis=0).sum(axis=0).sum(axis=0)
                elif 'abs' in s:
                    sparsity_vals = (abs(img.get_data()) > thresh).sum(axis=0).sum(axis=0).sum(axis=0)
                sparsity[hemi][s].append(sparsity_vals)

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
                        hpi_mean = np.mean(hpi[np.isfinite(hpi)])
                        hpi_sd = np.std(hpi[np.isfinite(hpi)])
                        # set non finite values to 0
                        hpi[~np.isfinite(hpi)] = 0

                    hpi_vals[sign]['vals'].append(hpi)
                    hpi_vals[sign]['mean'].append(hpi_mean)
                    hpi_vals[sign]['sd'].append(hpi_sd)

    # Now plot:
    x = [[c] * c for c in components]
    # 1) Sparsity for wb, R and L ICA images
    fh, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(18, 6))
    sparsity_styles = {'pos_005': ['b', 'lightblue'],
                       'neg_005': ['r', 'lightpink'],
                       'abs_005': ['g', 'lightgreen']}
    for ax, hemi in zip(axes, images_key):
        for s in sparsity_levels:
            mean = np.asarray([np.mean(arr) for arr in sparsity[hemi][s]])
            sd = np.asarray([np.std(arr) for arr in sparsity[hemi][s]])
            ax.fill_between(components, mean + sd, mean - sd, linewidth=0,
                            facecolor=sparsity_styles[s][1], alpha=0.5)
            ax.plot(components, mean, color=sparsity_styles[s][0], label=s)
        # Overlay individual points for absolute threshold
        ax.scatter(np.hstack(x), np.hstack(sparsity[hemi]['abs_005']),
                   c=sparsity_styles['abs_005'][0])
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
    for ax, sign in zip(axes, ['pos', 'neg']):
        mean, sd = np.asarray(hpi_vals[sign]['mean']), np.asarray(hpi_vals[sign]['sd'])
        ax.fill_between(components, mean + sd, mean - sd, linewidth=0,
                        facecolor=hpi_styles[sign][1], alpha=0.5)
        size = sparsity['wb']['%s_005' % (sign)]
        ax.scatter(np.hstack(x), np.hstack(hpi_vals[sign]['vals']), label=sign,
                   c=hpi_styles[sign][0], s=np.hstack(size) / 20)
        ax.plot(components, mean, c=hpi_styles[sign][0])
        ax.set_title("%s" % (sign))
        ax.set_xlim((0, components[-1] + 5))
        ax.set_ylim((-1, 1))
        ax.set_xticks(components)
        ax.set_ylabel("HPI((R-L)/(R+L) for # of voxels %s" % (hpi_styles[sign][2]))

    fh.text(0.5, 0.04, "# of components", ha="center")

    out_path = op.join(out_dir, 'wb_HPI.png')
    save_and_close(out_path, fh=fh)


def main_ic_loop(components, scoring, dataset,
                 memory=Memory(cachedir='nilearn_cache'), **kwargs):
    match_methods = ['wb', 'rl', 'lr']
    out_dir = op.join('ica_imgs', dataset)
    mean_scores, unmatched = [], []
    for match in match_methods:
        print("Plotting results for %s matching method" % match)
        mean_score_d, num_unmatched_d = {}, {}
        for force_match in [False, True]:
            for c in components:
                print("Running analysis with %d components" % c)
                img_d, score_mats_d, sign_mats_d = main(key=match, force_match=force_match,
                                                        n_components=c, dataset=dataset,
                                                        scoring=scoring, **kwargs)

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
