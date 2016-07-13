# *- encoding: utf-8 -*-
# Author: Ben Cipollini
# License: BSD

import os
import os.path as op

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn.image import iter_img, index_img, math_img
from nilearn.plotting import plot_stat_map
from scipy import stats

from nilearn_ext.utils import reorder_mat, get_ic_terms, get_n_terms, get_match_idx_pair
from nilearn_ext.radar import radar_factory

import math


def nice_number(value, round_=False):
    """
    Convert a number to a print-ready value.

    nice_number(value, round_=False) -> float
    """
    exponent = math.floor(math.log(value, 10))
    fraction = value / 10 ** exponent

    if round_:
        if fraction < 1.5:
            nice_fraction = 1.
        elif fraction < 3.:
            nice_fraction = 2.
        elif fraction < 7.:
            nice_fraction = 5.
        else:
            nice_fraction = 10.
    else:
        if fraction <= 1:
            nice_fraction = 1.
        elif fraction <= 2:
            nice_fraction = 2.
        elif fraction <= 5:
            nice_fraction = 5.
        else:
            nice_fraction = 10.

    return nice_fraction * 10 ** exponent


def nice_bounds(axis_start, axis_end, num_ticks=8):
    """
    Returns tuple as (nice_axis_start, nice_axis_end, nice_tick_width)
    """
    axis_width = axis_end - axis_start
    if axis_width == 0:
        nice_tick_w = 0
    else:
        nice_range = nice_number(axis_width)
        nice_tick_w = nice_number(nice_range / (num_ticks - 1), round_=True)
        axis_start = math.floor(axis_start / nice_tick_w) * nice_tick_w
        axis_end = math.ceil(axis_end / nice_tick_w) * nice_tick_w

    nice_tick = np.arange(axis_start, axis_end, nice_tick_w)[1:]
    return axis_start, axis_end, nice_tick


def save_and_close(out_path, fh=None):
    fh = fh or plt.gcf()
    if not op.exists(op.dirname(out_path)):
        os.makedirs(op.dirname(out_path))
    fh.savefig(out_path)
    plt.close(fh)


def _title_from_terms(terms, ic_idx, label=None, n_terms=4, sign=1):

    if terms is None:
        return '%s[%d]' % (label, ic_idx)

    # Use the n terms weighted most as a positive title, n terms
    # weighted least as a negative title and return both

    pos_terms = get_n_terms(terms, ic_idx, n_terms=n_terms, sign=sign)
    neg_terms = get_n_terms(terms, ic_idx, n_terms=n_terms, top_bottom="bottom",
                            sign=sign)

    title = '%s[%d]: POS(%s) \n NEG(%s)' % (
        label, ic_idx, ', '.join(pos_terms), ', '.join(neg_terms))

    return title


def plot_components(ica_image, hemi='', out_dir=None,
                    bg_img=datasets.load_mni152_template()):
    print("Plotting %s components..." % hemi)

    # Determine threshoold and vmax for all the plots
    # get nonzero part of the image for proper thresholding of
    # r- or l- only component
    nonzero_img = ica_image.get_data()[np.nonzero(ica_image.get_data())]
    thr = stats.scoreatpercentile(np.abs(nonzero_img), 90)
    vmax = stats.scoreatpercentile(np.abs(nonzero_img), 99.99)
    for ci, ic_img in enumerate(iter_img(ica_image)):

        title = _title_from_terms(terms=ica_image.terms, ic_idx=ci, label=hemi)
        fh = plt.figure(figsize=(14, 6))
        plot_stat_map(ic_img, axes=fh.gca(), threshold=thr, vmax=vmax,
                      colorbar=True, title=title, black_bg=True, bg_img=bg_img)

        # Save images instead of displaying
        if out_dir is not None:
            save_and_close(out_path=op.join(
                out_dir, '%s_component_%i.png' % (hemi, ci)))


def plot_components_summary(ica_image, hemi='', out_dir=None,
                            bg_img=datasets.load_mni152_template()):
    print("Plotting %s components summary..." % hemi)

    n_components = ica_image.get_data().shape[3]

    # Determine threshoold and vmax for all the plots
    # get nonzero part of the image for proper thresholding of
    # r- or l- only component
    nonzero_img = ica_image.get_data()[np.nonzero(ica_image.get_data())]
    thr = stats.scoreatpercentile(np.abs(nonzero_img), 90)
    vmax = stats.scoreatpercentile(np.abs(nonzero_img), 99.99)
    for ii, ic_img in enumerate(iter_img(ica_image)):

        ri = ii % 5  # row i
        ci = (ii / 5) % 5  # column i
        pi = ii % 25 + 1  # plot i
        fi = ii / 25  # figure i

        if ri == 0 and ci == 0:
            fh = plt.figure(figsize=(30, 20))
            print('Plot %03d of %d' % (fi + 1, np.ceil(n_components / 25.)))
        ax = fh.add_subplot(5, 5, pi)

        title = _title_from_terms(terms=ica_image.terms, ic_idx=ii, label=hemi)

        colorbar = True if ci == 4 else False

        plot_stat_map(
            ic_img, axes=ax, threshold=thr, vmax=vmax, colorbar=colorbar,
            title=title, black_bg=True, bg_img=bg_img)

        if (ri == 4 and ci == 4) or ii == n_components - 1:
            out_path = op.join(
                out_dir, '%s_components_summary%02d.png' % (hemi, fi + 1))
            save_and_close(out_path)


def plot_component_comparisons(images, labels, score_mat, sign_mat,
                               force=False, out_dir=None):
    """
    Uses the score_mat to match up two images. If force, one-to-one matching 
    is forced.
    Sign_mat is used to flip signs when comparing two images.
    """
    # Be careful
    assert len(images) == 2
    assert len(labels) == 2
    assert images[0].shape == images[1].shape
    n_components = images[0].shape[3]  # values @ 0 and 1 are the same
    assert score_mat.shape == sign_mat.shape
    assert len(score_mat[0]) == n_components
    
    # Get indices for matching components
    match, unmatch = get_match_idx_pair(score_mat, sign_mat, force=force)
    idx_pair = match["idx"]
    sign_pair = match["sign"]

    if not force and unmatch["idx"] is not None:
        idx_pair = np.hstack((idx_pair, unmatch["idx"]))
        sign_pair = np.hstack((sign_pair, unmatch["sign"]))
    
    n_comp = len(idx_pair[0])   # number of comparisons

    # Calculate a vmax optimal across all the plots
    # get nonzero part of the image for proper thresholding of
    # r- or l- only component
    nonzero_imgs = [img.get_data()[np.nonzero(img.get_data())]
                    for img in images]
    dat = np.append(nonzero_imgs[0], nonzero_imgs[1])
    vmax = stats.scoreatpercentile(np.abs(dat), 99.99)

    print("Plotting results.")
    for i in range(n_comp):
        c1i, c2i = idx_pair[0][i], idx_pair[1][i]
        cis = [c1i, c2i]
        
        prefix = "unmatched-" if i >= n_components else ""
        num = i-n_components if i >= n_components else i
        png_name = '%s%s_%s_%s.png' % (prefix, labels[0], labels[1], num)
        print "plotting %s" % png_name 

        comp_imgs = [index_img(img, ci) for img, ci in zip(images, cis)]

        # flip the sign if sign_mat for the corresponding comparison is -1
        signs = [sign_pair[0][i], sign_pair[1][i]]
        comp_imgs = [math_img("%d*img" % (sign), img=img)
                     for sign, img in zip(signs, comp_imgs)]

        if ('R' in labels and 'L' in labels):
            # Combine left and right image, show just one.
            # terms are not combined here
            comp = math_img("img1+img2", img1=comp_imgs[0], img2=comp_imgs[1])
            titles = [_title_from_terms(
                terms=comp_imgs[labels.index(hemi)].terms,
                ic_idx=cis[labels.index(hemi)], label=hemi,
                sign=signs[labels.index(hemi)]) for hemi in labels]
            fh = plt.figure(figsize=(14, 8))
            plot_stat_map(
                comp, axes=fh.gca(), title="\n".join(titles), black_bg=True,
                symmetric_cbar=True, vmax=vmax)

        else:
            # Show two images, one above the other.
            fh = plt.figure(figsize=(14, 12))

            for ii in [0, 1]:  # Subplot per image
                ax = fh.add_subplot(2, 1, ii + 1)
                comp = comp_imgs[ii]

                title = _title_from_terms(
                    terms=images[ii].terms, ic_idx=cis[ii],
                    label=labels[ii], sign=signs[ii])

                if ii == 0:
                    display = plot_stat_map(comp, axes=ax, title=title,    # noqa
                                            black_bg=True, symmetric_cbar=True,
                                            vmax=vmax)
                else:
                    # use same cut coords
                    cut_coords = display.cut_coords  # noqa
                    display = plot_stat_map(comp, axes=ax, title=title,
                                            black_bg=True, symmetric_cbar=True,
                                            vmax=vmax, display_mode='ortho',
                                            cut_coords=cut_coords)

        # Save images instead of displaying
        if out_dir is not None:
            save_and_close(out_path=op.join(out_dir, png_name), fh=fh)


def plot_comparison_matrix(score_mat, labels, scoring, normalize=True,
                           out_dir=None, vmax=None, colorbar=True, prefix=""):

    # Settings
    score_mat, x_idx, y_idx = reorder_mat(score_mat, normalize=normalize)
    idx = np.arange(score_mat.shape[0])
    vmax = vmax  # or min(scores.max(), 10 if normalize else np.inf)
    vmin = 0  # 1 if normalize else 0

    # Plotting
    fh = plt.figure(figsize=(10, 10))
    ax = fh.gca()
    cax = ax.matshow(score_mat, vmin=vmin, vmax=vmax)
    ax.set_xlabel("%s components" % (labels[1]))
    ax.set_ylabel("%s components" % (labels[0]))
    ax.set_xticks(idx), ax.set_xticklabels(x_idx)
    ax.set_yticks(idx), ax.set_yticklabels(y_idx)
    if colorbar:
        fh.colorbar(cax)

    # Saving
    if out_dir is not None:
        save_and_close(out_path=op.join(out_dir, '%s%s_%s_simmat%s.png' % (
            prefix, labels[0], labels[1], '-normalized' if normalize else '')))


def plot_term_comparisons(terms, labels, ic_idx_list, sign_list, color_list=('g', 'r', 'b'),
                          top_n=4, bottom_n=4, standardize=True, out_dir=None):
    """
    Take the list of ica image terms and the indices of components to be compared, and
    plots the top n and bottom n term values for each component as a radar graph.

    The sign_list should indicate whether term values should be flipped (-1) or not (1).
    """
    assert len(terms) == len(labels)
    assert len(terms) == len(ic_idx_list)
    assert len(terms) == len(sign_list)
    assert len(terms) == len(color_list)
    n_comp = len(ic_idx_list[0])   # length of each ic_idx_list and sign_list
    for i in range(len(terms)):
        assert len(ic_idx_list[i]) == n_comp
        assert len(sign_list[i]) == n_comp

    # iterate over the ic_idx_list and sign_list for each term and plot
    # store top n and bottom n terms for each label
    term_arr = np.empty((len(labels), n_comp, top_n + bottom_n), dtype="S30")
    for n in range(n_comp):

        terms_of_interest = []
        term_vals = []
        name = ''

        for i, (term, label) in enumerate(zip(terms, labels)):
            idx = ic_idx_list[i][n]
            sign = sign_list[i][n]
            # Get list of top n and bottom n terms for each term list
            top_terms = get_n_terms(
                term, idx, n_terms=top_n, top_bottom='top', sign=sign)
            bottom_terms = get_n_terms(
                term, idx, n_terms=bottom_n, top_bottom='bottom', sign=sign)
            combined = np.append(top_terms, bottom_terms)
            terms_of_interest.append(combined)
            term_arr[i][n] = combined

            # Also store term vals (z-score if standardize) for each list
            t, vals = get_ic_terms(term, idx, sign=sign, standardize=standardize)
            s = pd.Series(vals, index=t, name=label)
            term_vals.append(s)

            # Construct name for the comparison
            name += label + '[%d] ' % (idx)

        # Data for all the terms
        termscore_df = pd.concat(term_vals, axis=1)

        # Get unique terms from terms_of_interest list
        toi_unique = np.unique(terms_of_interest)

        # Get values for unique terms_of_interest
        data = termscore_df.loc[toi_unique]
        data = data.sort_values(list(labels), ascending=False)

        # Now plot radar!
        N = len(toi_unique)
        theta = radar_factory(N)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1, projection='radar')
        title = "Term comparisons for %scomponents" % (name)
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')

        y_min, y_max, y_tick = nice_bounds(data.values.min(), data.values.max())
        ax.set_ylim(y_min, y_max)
        ax.set_yticks([0], minor=True)
        ax.set_yticks(y_tick)
        ax.yaxis.grid(which='major', linestyle=':')
        ax.yaxis.grid(which='minor', linestyle='-')

        for label, color in zip(labels, color_list):
            ax.plot(theta, data[label], color=color)
            ax.fill(theta, data[label], facecolor=color, alpha=0.25)
        ax.set_varlabels(data.index.values)

        legend = plt.legend(labels, loc=(1.1, 0.9), labelspacing=0.1)
        plt.setp(legend.get_texts(), fontsize='small')
#        plt.show()

        # Saving
        if out_dir is not None:
            save_and_close(
                out_path=op.join(out_dir, '%sterm_comparisons.png' % (
                    name.replace(" ", "_"))))

    # Save top n and bottom n terms for each label
    term_dfs = []
    col_names = ["top%d" % (n + 1) for n in range(top_n)] + ["bottom%d" % (n + 1) for n in range(bottom_n)]
    for i, label in enumerate(labels):
        term_df = pd.DataFrame(term_arr[i], columns=["%s_%s" % (label, col) for col in col_names])
        term_df.insert(0, "%s_idx" % (label), ic_idx_list[i])
        term_dfs.append(term_df)
    all_term_df = pd.concat(term_dfs, axis=1)
    out_file = op.join(out_dir, 'term_comparison_summary.csv')
    all_term_df.to_csv(out_file)
