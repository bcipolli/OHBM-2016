# *- encoding: utf-8 -*-
# Author: Ben Cipollini, Ami Tsuchida
# License: BSD

import os.path as op

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.image import index_img, math_img

from nibabel_ext import NiftiImageWithTerms
from nilearn_ext.datasets import fetch_neurovault
from nilearn_ext.decomposition import compare_components, generate_components
from nilearn_ext.plotting import (plot_component_comparisons, plot_components,
                                  plot_components_summary, plot_comparison_matrix,
                                  plot_term_comparisons)
from nilearn_ext.utils import get_ic_terms, get_match_idx_pair


def load_or_generate_components(hemi, out_dir='.', plot_dir=None, force=False,
                                *args, **kwargs):
    """Load an image and return if it exists, otherwise compute via ICA"""
    # Only re-run if image doesn't exist.
    img_path = op.join(out_dir, '%s_ica_components.nii.gz' % hemi)
    if not force and op.exists(img_path):
        img = NiftiImageWithTerms.from_filename(img_path)

    else:
        img = generate_components(hemi=hemi, out_dir=out_dir, *args, **kwargs)
        png_dir = op.join(out_dir, 'png')
        plot_components(img, hemi=hemi, out_dir=png_dir)
        plot_components_summary(img, hemi=hemi, out_dir=png_dir)
    return img


def concat_RL(R_img, L_img, rl_idx_pair, rl_sign_pair=None):
    """
    Given R and L ICA images and their component index pairs, concatenate images to
    create bilateral image using the index pairs. Sign flipping can be specified in rl_sign_pair.

    """
    # Make sure images have same number of components and indices are less than the n_components
    assert R_img.shape == L_img.shape
    n_components = R_img.shape[3]
    assert np.max(rl_idx_pair) < n_components
    n_rl_imgs = len(rl_idx_pair[0])
    assert n_rl_imgs == len(rl_idx_pair[1])
    if rl_sign_pair:
        assert n_rl_imgs == len(rl_sign_pair[0])
        assert n_rl_imgs == len(rl_sign_pair[1])

    # Match indice pairs and combine
    terms = R_img.terms.keys()
    rl_imgs = []
    rl_term_vals = []

    for i in range(n_rl_imgs):
        rci, lci = rl_idx_pair[0][i], rl_idx_pair[1][i]
        R_comp_img = index_img(R_img, rci)
        L_comp_img = index_img(L_img, lci)

        # sign flipping
        r_sign = rl_sign_pair[0][i] if rl_sign_pair else 1
        l_sign = rl_sign_pair[1][i] if rl_sign_pair else 1

        R_comp_img = math_img("%d*img" % (r_sign), img=R_comp_img)
        L_comp_img = math_img("%d*img" % (l_sign), img=L_comp_img)

        # combine images
        rl_imgs.append(math_img("r+l", r=R_comp_img, l=L_comp_img))

        # combine terms
        if terms:
            r_ic_terms, r_ic_term_vals = get_ic_terms(R_img.terms, rci, sign=r_sign)
            l_ic_terms, l_ic_term_vals = get_ic_terms(L_img.terms, lci, sign=l_sign)
            rl_term_vals.append((r_ic_term_vals + l_ic_term_vals) / 2)

    # Squash into single image
    concat_img = nib.concat_images(rl_imgs)
    if terms:
        concat_img.terms = dict(zip(terms, np.asarray(rl_term_vals).T))
    return concat_img


def compare_components_and_plot(images, labels, scoring, force_match=False, out_dir=None):
    """
    For any given pair of ica component images, compute score matrix and plot each matching
    and (if force_match = False) non-matching pair of component images.

    Returns score matrix and sign matrix.
    """
    # Compare components
    # The sign_mat contains signs that gave the best score for the comparison
    score_mat, sign_mat = compare_components(images, labels, scoring)

    n_components = score_mat.shape[0]

    # Plot comparison matrix
    for normalize in [False, True]:
        plot_comparison_matrix(
            score_mat, labels, scoring, normalize=normalize, out_dir=out_dir)

    # Show component comparisons
    matched_idx_arr, unmatched_idx_arr = get_match_idx_pair(
        score_mat, sign_mat, force=force_match)

    # ...for matched pairs
    matched_idx_pair = [matched_idx_arr[0], matched_idx_arr[1]]
    matched_sign_pair = [np.ones(n_components), matched_idx_arr[2]]

    plot_component_comparisons(
        images, labels, idx_pair=matched_idx_pair,
        sign_pair=matched_sign_pair, out_dir=out_dir)

    # ...for unmatched pairs
    if unmatched_idx_arr is not None and len(unmatched_idx_arr) > 0:
        unmatched_idx_pair = [unmatched_idx_arr[0], unmatched_idx_arr[1]]
        unmatched_sign_pair = [np.ones(unmatched_idx_arr[2].shape), unmatched_idx_arr[2]]

        plot_component_comparisons(
            images, labels, idx_pair=unmatched_idx_pair,
            sign_pair=unmatched_sign_pair, out_dir=out_dir,
            prefix="unmatched")

    return score_mat, sign_mat


def get_dataset(dataset, max_images=np.inf, **kwargs):
    """Retrieve & normalize dataset from nilearn"""
    # Download
    if dataset == 'neurovault':
        images, term_scores = fetch_neurovault(max_images=max_images, **kwargs)

    elif dataset == 'abide':
        dataset = datasets.fetch_abide_pcp(
            n_subjects=min(94, max_images), **kwargs)
        images = [{'local_path': p} for p in dataset['func_preproc']]
        term_scores = None

    elif dataset == 'nyu':
        dataset = datasets.fetch_nyu_rest(
            n_subjects=min(25, max_images), **kwargs)
        images = [{'local_path': p} for p in dataset['func']]
        term_scores = None

    else:
        raise ValueError("Unknown dataset: %s" % dataset)
    return images, term_scores


def do_main_analysis(dataset, images, term_scores,
                     key="wb", force_match=False, n_components=20,
                     max_images=np.inf, scoring='l1norm', query_server=True,
                     force=False, nii_dir=None, plot_dir=None, random_state=42,
                     hemis=('wb', 'R', 'L')):

    # Output directories
    nii_dir = nii_dir or op.join('ica_nii', dataset, str(n_components))
    plot_dir = plot_dir or op.join('ica_imgs', dataset,
                                   '%s-%dics' % (scoring, n_components))
    plot_sub_dir = op.join(plot_dir, '%s-matching%s' % (key, '_forced' if force_match else ''))

    # 1) Components are generated for R-, L-only, and whole brain images.

    imgs = {}

    # Load or generate components
    kwargs = dict(images=[im['local_path'] for im in images],
                  n_components=n_components, term_scores=term_scores,
                  out_dir=nii_dir, plot_dir=plot_dir)
    for hemi in hemis:
        print("Running analyses on %s" % hemi)
        imgs[hemi] = (load_or_generate_components(hemi=hemi, force=force,
                                                  random_state=random_state, **kwargs))

    # 2) Compare components in order to get concatenated RL image
    #    "wb": R- and L- is compared to wb-components, then matched
    #    "rl": direct R- and L- comparison, using R as a ref
    #    "lr": direct R- and L- comparison, using L as a ref
    if key == "wb":
        comparisons = [('wb', 'R'), ('wb', 'L')]
    elif key == "rl":
        comparisons = [('R', 'L')]
    elif key == "lr":
        comparisons = [('L', 'R')]

    score_mats, sign_mats = {}, {}

    for comp in comparisons:

        img_pair = [imgs[comp[0]], imgs[comp[1]]]

        # Compare components and plot
        # The sign_mat contains signs that gave the best score for the comparison
        score_mat, sign_mat = compare_components_and_plot(images=img_pair, labels=comp,
                                                          scoring=scoring, force_match=force_match,
                                                          out_dir=plot_sub_dir)

        # Store score_mat and sign_mat
        score_mats[comp] = score_mat
        sign_mats[comp] = sign_mat

        # Get indices for matching up R and L components
        matched_idx_arr, unmatched_idx_arr = get_match_idx_pair(score_mat, sign_mat,
                                                                force=force_match)
        if comp[0] == 'R':
            r_idx_arr = matched_idx_arr[0]
            r_sign_arr = np.ones(n_components)
        elif comp[0] == 'L':
            l_idx_arr = matched_idx_arr[0]
            l_sign_arr = np.ones(n_components)

        if comp[1] == 'R':
            r_idx_arr = matched_idx_arr[1]
            r_sign_arr = matched_idx_arr[2]
        elif comp[1] == 'L':
            l_idx_arr = matched_idx_arr[1]
            l_sign_arr = matched_idx_arr[2]

    # 3) Now match up R and L
    imgs['RL'] = concat_RL(R_img=imgs['R'], L_img=imgs['L'],
                           rl_idx_pair=(r_idx_arr, l_idx_arr),
                           rl_sign_pair=(r_sign_arr, l_sign_arr))

    # 4) Now compare the concatenated image to bilateral components
    # Note that for wb-matching, diagnal components will be matched by definition
    comp = ('wb', 'RL')
    img_pair = [imgs['wb'], imgs['RL']]
    score_mat, sign_mat = compare_components_and_plot(images=img_pair, labels=comp,
                                                      scoring=scoring, force_match=force_match,
                                                      out_dir=plot_sub_dir)

    # Store score_mat and sign_mat
    score_mats[comp] = score_mat
    sign_mats[comp] = sign_mat

    # Show term comparisons between the matched wb, R and L components
    terms = [imgs[hemi].terms for hemi in hemis]
    matched_idx_arr, unmatched_idx_arr = get_match_idx_pair(score_mat, sign_mat,
                                                            force=force_match)
    # component index list for wb, R and L
    wb_idx_arr = matched_idx_arr[0]
    r_idx_arr = r_idx_arr[matched_idx_arr[1]]
    l_idx_arr = l_idx_arr[matched_idx_arr[1]]
    ic_idx_list = [wb_idx_arr, r_idx_arr, l_idx_arr]

    # sign flipping list for wb, R and L
    wb_sign_arr = np.ones(n_components)
    r_sign_arr = matched_idx_arr[2] * r_sign_arr[matched_idx_arr[1]]
    l_sign_arr = matched_idx_arr[2] * l_sign_arr[matched_idx_arr[1]]
    sign_list = [wb_sign_arr, r_sign_arr, l_sign_arr]

    plot_term_comparisons(terms, labels=hemis, ic_idx_list=ic_idx_list,
                          sign_list=sign_list, color_list=['g', 'r', 'b'],
                          top_n=5, bottom_n=5, standardize=True, out_dir=plot_sub_dir)

    return imgs, score_mats, sign_mats


def main(dataset, key="wb", force_match=False, n_components=20,
         max_images=np.inf, scoring='l1norm', query_server=True,
         force=False, nii_dir=None, plot_dir=None, random_state=42):
    """
    Compute components, then run requested comparisons.

    "wb": R- and L- components are first matched to wb components, and concatenated
    based on their match with wb components. Concatenated RL components are then
    compared to wb components.

    "rl": R- and L- components are compared and matched directly, using R as a ref.
    If one-to-one matching is forced with force_match=True, this is identical as lr.

    "lr": R- and L- components are compared and matched directly, using L as a ref.
    If one-to-one matching is forced with force_match=True, this is identical as rl.

    """
    images, term_scores = get_dataset(dataset, max_images=max_images,
                                      query_server=query_server)

    return do_main_analysis(
        dataset=dataset, images=images, term_scores=term_scores,
        key=key, force_match=force_match,
        n_components=n_components, scoring=scoring,
        force=force, nii_dir=nii_dir, plot_dir=plot_dir,
        random_state=random_state)


if __name__ == '__main__':
    import warnings
    from argparse import ArgumentParser

    # Look for image computation errors
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('error', RuntimeWarning)  # Detect bad NV images

    # Arg parsing
    match_methods = ('wb', 'rl', 'lr')
    parser = ArgumentParser(description="Run ICA on individual hemispheres, "
                                        "and whole brain, then compare.\n\n"
                                        "wb = R- and L- components are first matched "
                                        "with wb,and concatenated through the wb-match.\n"
                                        "rl = R- and L- components are directly compared, "
                                        "using R as a ref, then combined based on their "
                                        "spatial similarity.\n"
                                        "lr = same as rl, but using L as a ref")
    parser.add_argument('key', nargs='?', default='wb', choices=match_methods)
    parser.add_argument('--force_match', action='store_true', default=False)
    parser.add_argument('--force', action='store_true', default=False)
    parser.add_argument('--offline', action='store_true', default=False)
    parser.add_argument('--qc', action='store_true', default=False)
    parser.add_argument('--components', nargs='?', type=int, default=20,
                        dest='n_components')
    parser.add_argument('--dataset', nargs='?', default='neurovault',
                        choices=['neurovault', 'abide', 'nyu'])
    parser.add_argument('--seed', nargs='?', type=int, default=42,
                        dest='random_state')
    parser.add_argument('--scoring', nargs='?', default='l1norm',
                        choices=['l1norm', 'l2norm', 'correlation'])
    args = vars(parser.parse_args())

    # Run qc
    query_server = not args.pop('offline')
    if args.pop('qc'):
        from qc import qc_image_data
        qc_image_data(args['dataset'], query_server=query_server)

    # Run main
    main(query_server=query_server, **args)

    plt.show()
