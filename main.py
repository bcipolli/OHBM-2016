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
    For any given pair of ica component images, compute score matrix and plot the matrix.
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


def do_main_analysis(dataset, images, term_scores, key="wb", n_components=20,
                     plot=True, max_images=np.inf, scoring='l1norm', 
                     query_server=True, force=False, nii_dir=None, 
                     plot_dir=None, random_state=42, hemis=('wb', 'R', 'L')):

    # Output directories
    nii_dir = nii_dir or op.join('ica_nii', dataset, str(n_components))
    plot_dir = plot_dir or op.join('ica_imgs', dataset,
                                   '%s-%dics' % (scoring, n_components),
                                   '%s-matching' % key)

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
    RL_arr = {}

    for comp in comparisons:

        img_pair = [imgs[comp[0]], imgs[comp[1]]]

        # Compare components and plot similarity matrix
        # The sign_mat contains signs that gave the best score for the comparison
        score_mat, sign_mat = compare_components_and_plot(images=img_pair, labels=comp,
                                                          scoring=scoring, out_dir=plot_dir)

        # Store score_mat and sign_mat
        score_mats[comp] = score_mat
        sign_mats[comp] = sign_mat
        
        # Get indices for matching up components for both forced and unforced one-to-one matching
        for force_match in [True, False]:
            force_status = 'forced' if force_match else 'unforced'
            plot_sub_dir = op.join(plot_dir, '%s-match' % force_status)
            match, unmatch = get_match_idx_pair(score_mat, sign_mat, force=force_match)
            
            # Store R and L indices/signs to match up R and L 
            for i, hem in enumerate(comp):
                if hem in ['R', 'L']:
                    RL_arr[(force_status, hem, "idx")] = match["idx"][i]
                    RL_arr[(force_status, hem, "sign")] = match["sign"][i]
            
            # If plot=True, plot matched (and unmatched, if unforced matching) components
            if plot:
                plot_component_comparisons(images=img_pair, labels=comp, 
                                           score_mat=score_mat, sign_mat=sign_mat, 
                                           force=force_match, out_dir=plot_sub_dir)
                           
    #3) Now match up R and L (forced vs unforced match)
    for force_match in [True, False]:
        force_status = 'forced' if force_match else 'unforced'
        plot_sub_dir = op.join(plot_dir, '%s-match' % force_status)
    
        rl_idx_pair = (RL_arr[(force_status, "R", "idx")], RL_arr[(force_status, "L", "idx")])
        rl_sign_pair = (RL_arr[(force_status, "R", "sign")], RL_arr[(force_status, "L", "sign")])
        imgs['RL-%s' % force_status] = concat_RL(R_img=imgs['R'], L_img=imgs['L'],
                                                 rl_idx_pair=rl_idx_pair,
                                                 rl_sign_pair=rl_sign_pair)

        # 4) Compare the concatenated image to bilateral components (ie wb vs RL)
        # Note that for wb-matching, diagnal components will be matched by definition
        comp = ('wb', 'RL-%s'% force_status)
        img_pair = [imgs[comp[0]], imgs[comp[1]]]
        score_mat, sign_mat = compare_components_and_plot(images=img_pair, labels=comp,
                                                          scoring=scoring, out_dir=plot_sub_dir)

        # Store score_mat and sign_mat
        score_mats[comp] = score_mat
        sign_mats[comp] = sign_mat

        # If plot=True, plot matched (and unmatched, if unforced matching) components
        if plot:
            plot_component_comparisons(images=img_pair, labels=comp,
                                       score_mat=score_mat, sign_mat=sign_mat,
                                       force=force_match, out_dir=plot_sub_dir)
        
            # Show term comparisons between the matched wb, R and L components
            match, unmatch = get_match_idx_pair(score_mat, sign_mat, force=force_match)
            terms = [imgs[hemi].terms for hemi in hemis]
        
            # component index list for wb, R and L
            wb_idx_arr = match["idx"][0]
            r_idx_arr, l_idx_arr = [arr[match["idx"][1]] for arr in rl_idx_pair]
            ic_idx_list = [wb_idx_arr, r_idx_arr, l_idx_arr]

            # sign flipping list for wb, R and L
            wb_sign_arr = match["sign"][0]
            r_sign_arr, l_sign_arr = [match["sign"][1] * arr[match["idx"][1]] for arr in rl_sign_pair]
            sign_list = [wb_sign_arr, r_sign_arr, l_sign_arr]
    
            plot_term_comparisons(terms, labels=hemis, ic_idx_list=ic_idx_list,
                                  sign_list=sign_list, color_list=['g', 'r', 'b'],
                                  top_n=5, bottom_n=5, standardize=True, out_dir=plot_sub_dir)

    return imgs, score_mats, sign_mats


def main(dataset, key="wb", n_components=20, plot=True,
         max_images=np.inf, scoring='l1norm', query_server=True,
         force=False, nii_dir=None, plot_dir=None, random_state=42):
    """
    Compute components, then run requested comparisons.

    "wb": R- and L- components are first matched to wb components, and concatenated
    based on their match with wb components. Concatenated RL components are calculated
    with and without forced one-to-one matching, then compared to wb components.

    "rl": R- and L- components are compared and matched directly, using R as a ref.
    For forced one-to-one matching, this is identical as lr.

    "lr": R- and L- components are compared and matched directly, using L as a ref.
    For forced one-to-one matching,, this is identical as rl.

    """
    images, term_scores = get_dataset(dataset, max_images=max_images,
                                      query_server=query_server)

    return do_main_analysis(
        dataset=dataset, images=images, term_scores=term_scores,
        key=key, n_components=n_components, plot=plot, scoring=scoring,
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
    parser.add_argument('--skip_plot', action='store_true', default=False)
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
    plot = not args.pop('skip_plot') 
    main(query_server=query_server, plot=plot, **args)

    plt.show()
