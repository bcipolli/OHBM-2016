# *- encoding: utf-8 -*-
# Author: Ben Cipollini, Ami Tsuchida
# License: BSD

import os
import os.path as op

import numpy as np
from nilearn import datasets
from nilearn.image import iter_img

from six import string_types
from sklearn.decomposition import FastICA
from sklearn.externals.joblib import Memory
from scipy import stats

from nibabel_ext import NiftiImageWithTerms
from .image import cast_img, clean_img
from .masking import HemisphereMasker, flip_img_lr, GreyMatterNiftiMasker


def generate_components(images, hemi, term_scores=None,
                        n_components=20, random_state=42,
                        out_dir=None, memory=Memory(cachedir='nilearn_cache')):
    """
        images: list
            Can be nibabel images, can be file paths.
    """
    # Create grey matter mask from mni template
    target_img = datasets.load_mni152_template()

    # Reshape & mask images
    print("%s: Reshaping and masking images; may take time." % hemi)
    if hemi == 'wb':
        masker = GreyMatterNiftiMasker(target_affine=target_img.affine,
                                       target_shape=target_img.shape,
                                       memory=memory)

    else:  # R and L maskers
        masker = HemisphereMasker(target_affine=target_img.affine,
                                  target_shape=target_img.shape,
                                  memory=memory,
                                  hemisphere=hemi)
    masker = masker.fit()

    # Images may fail to be transformed, and are of different shapes,
    # so we need to trasnform one-by-one and keep track of failures.
    X = []  # noqa
    xformable_idx = np.ones((len(images),), dtype=bool)
    for ii, im in enumerate(images):
        img = cast_img(im, dtype=np.float32)
        img = clean_img(img)
        try:
            X.append(masker.transform(img))
        except Exception as e:
            print("Failed to mask/reshape image %d/%s: %s" % (
                im.get('collection_id', 0),
                op.basename(im),
                e))
            xformable_idx[ii] = False

    # Now reshape list into 2D matrix
    X = np.vstack(X)  # noqa

    # Run ICA and map components to terms
    print("%s: Running ICA; may take time..." % hemi)
    fast_ica = FastICA(n_components=n_components, random_state=random_state)
    fast_ica = memory.cache(fast_ica.fit)(X.T)
    ica_maps = memory.cache(fast_ica.transform)(X.T).T

    if term_scores is not None:
        terms = term_scores.keys()
        term_matrix = np.asarray(term_scores.values())
        term_matrix[term_matrix < 0] = 0
        term_matrix = term_matrix[:, xformable_idx]  # terms x images
        # Don't use the transform method as it centers the data
        ica_terms = np.dot(term_matrix, fast_ica.components_.T).T

    # 2015/12/26 - sign matters for comparison, so don't do this!
    # 2016/02/01 - sign flipping is ok for R-L comparison, but RL concat
    #              may break this.
    # Pretty up the results
    for idx, ic in enumerate(ica_maps):
        if -ic.min() > ic.max():
            # Flip the map's sign for prettiness
            ica_maps[idx] = -ic
            if term_scores:
                ica_terms[idx] = -ica_terms[idx]

    # Create image from maps, save terms to the image directly
    ica_image = NiftiImageWithTerms.from_image(
        masker.inverse_transform(ica_maps))
    if term_scores:
        ica_image.terms = dict(zip(terms, ica_terms.T))

    # Write to disk
    if out_dir is not None:
        out_path = op.join(out_dir, '%s_ica_components.nii.gz' % hemi)
        if not op.exists(op.dirname(out_path)):
            os.makedirs(op.dirname(out_path))
        ica_image.to_filename(out_path)
    return ica_image


def compare_components(images, labels, scoring='l1norm',
                       memory=Memory(cachedir='nilearn_cache')):
    assert len(images) == 2
    assert len(labels) == 2
    assert images[0].shape == images[1].shape
    n_components = images[0].shape[3]  # values @ 0 and 1 are the same
    labels = [l.upper() for l in labels] # make input labels case insensitive
    print("Loading images.")
    for img in images:
        img.get_data()  # Just loaded to get them in memory..

    print("Scoring closest components (by %s)" % str(scoring))
    score_mat = np.zeros((n_components, n_components))
    sign_mat = np.zeros((n_components, n_components), dtype=np.int)
    c1_data = [None] * n_components
    c2_data = [None] * n_components

    c1_images = list(iter_img(images[0]))
    c2_images = list(iter_img(images[1]))

    lh_masker = HemisphereMasker(hemisphere='L', memory=memory).fit()
    rh_masker = HemisphereMasker(hemisphere='R', memory=memory).fit()

    for c1i, comp1 in enumerate(c1_images):
        for c2i, comp2 in enumerate(c2_images):
            # Make sure the two images align (i.e. not R and L opposite),
            #   and that only good voxels are compared (i.e. not full vs half)
            if 'R' in labels and 'L' in labels:
                if c1_data[c1i] is None or c2_data[c2i] is None:
                    R_img = comp1 if labels.index('R') == 0 else comp2  # noqa
                    L_img = comp1 if labels.index('L') == 0 else comp2  # noqa
                    masker = lh_masker  # use same masker; ensures same size
                if c1_data[c1i] is None:
                    c1_data[c1i] = masker.transform(flip_img_lr(R_img)).ravel()
                if c2_data[c2i] is None:
                    c2_data[c2i] = masker.transform(L_img).ravel()
                
            elif 'R' in labels or 'L' in labels:
                masker = rh_masker if 'R' in labels else lh_masker
                if c1_data[c1i] is None:
                    c1_data[c1i] = masker.transform(comp1).ravel()
                if c2_data[c2i] is None:
                    c2_data[c2i] = masker.transform(comp2).ravel()
                
            else:
                if c1_data[c1i] is None:
                    c1_data[c1i] = comp1.get_data().ravel()
                if c2_data[c2i] is None:
                    c2_data[c2i] = comp2.get_data().ravel()

            # Choose a scoring system.
            # Score should indicate DISSIMILARITY
            # Component sign is meaningless, so try both, but keep track of 
            # comparisons that had better score when flipping the sign
            score = np.inf
            for sign in [1, -1]:
                c1d, c2d = c1_data[c1i], sign * c2_data[c2i]
                if not isinstance(scoring, string_types):  # function
                    sc = scoring(c1d, c2d)
                elif scoring == 'l1norm':
                    sc = np.linalg.norm(c1d - c2d, ord=1)
                elif scoring == 'l2norm':
                    sc = np.linalg.norm(c1d - c2d, ord=2)
                elif scoring == 'correlation':
                    sc = 1 - stats.stats.pearsonr(c1d, c2d)[0]
                else:
                    raise NotImplementedError(scoring)
                if sc < score:
                    sign_mat[c1i, c2i] = sign
                score = min(score, sc)
            score_mat[c1i, c2i] = score

    return score_mat, sign_mat
