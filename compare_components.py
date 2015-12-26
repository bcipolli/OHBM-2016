# *- encoding: utf-8 -*-
# Author: Ben Cipollini
# License: BSD

import os
import os.path as op

import numpy as np
from matplotlib import pyplot as plt
from nilearn.image import iter_img, index_img
# from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map

from hemisphere_masker import (MniHemisphereMasker)


def compare_components(images, labels):
    assert len(images) == 2
    assert len(labels) == 2
    assert images[0].shape == images[1].shape
    n_components = images[0].shape[3]  # values @ 0 and 1 are the same

    print("Loading images.")
    for img in images:
        img.get_data()  # Just loaded to get them in memory..

    print("Scoring closest components (by L1 norm)")
    score_mat = np.zeros((n_components, n_components))
    for c1i, comp1 in enumerate(iter_img(images[0])):
        for c2i, comp2 in enumerate(iter_img(images[1])):
            if 'R' in labels or 'L' in labels:
                hemi_idx = labels.index('R') or labels.index('L')
                masker = MniHemisphereMasker(hemisphere=labels[hemi_idx]).fit()
                c1_data = masker.transform(comp1)
                c2_data = masker.transform(comp2)
            else:
                c1_data = comp1.get_data()
                c2_data = comp2.get_data()
            l1norm = np.abs(c1_data - c2_data).sum()
            score_mat[c1i, c2i] = l1norm

    return score_mat


def plot_comparisons(images, labels, score_mat, out_dir):
    assert len(images) == 2
    assert len(labels) == 2
    assert images[0].shape == images[1].shape
    n_components = images[0].shape[3]  # values @ 0 and 1 are the same

    # Find cross-image mapping
    most_similar_idx = score_mat.argmin(axis=1)

    print("Plotting results.")
    for c1i in range(n_components):
        cis = [c1i, most_similar_idx[c1i]]
        fh = plt.figure(figsize=(14, 8))

        # Determine scale bars
        dat = np.asarray([index_img(img, ci).get_data()
                          for img, ci in zip(images, cis)])
        vmax = np.abs(dat).max()
        del dat

        # Subplot per image
        for ii in [0, 1]:  # image index
            ax = fh.add_subplot(2, 1, ii + 1)
            comp = index_img(images[ii], cis[ii])

            # Use the 4 terms weighted most as a title
            terms = np.asarray(images[ii].terms.keys())
            ica_terms = np.asarray(images[ii].terms.values()).T
            ic_terms = ica_terms[cis[ii]]
            important_terms = terms[np.argsort(ic_terms)[-4:]]
            title = '%s[%d]: %s' % (
                labels[ii], cis[ii], ', '.join(important_terms[::-1]))

            if 'R' in labels or 'L' in labels and ii == 1:
                # use same cut coords
                plot_stat_map(comp, axes=ax, title=title,
                              symmetric_cbar=True, vmax=vmax,
                              display_mode='ortho',
                              cut_coords=display.cut_coords)  # noqa
            else:
                display = plot_stat_map(comp, axes=ax, title=title,  # noqa
                                        symmetric_cbar=True, vmax=vmax)

        # Save images instead of displaying
        if out_dir is not None:
            out_path = op.join(out_dir, '%s_%s_%s.png' % (
                labels[0], labels[1], c1i))
            if not op.exists(op.dirname(out_path)):
                os.makedirs(op.dirname(out_path))
            plt.savefig(out_path)
            plt.close()
