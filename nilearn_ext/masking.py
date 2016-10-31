# *- encoding: utf-8 -*-
# Author: Ben Cipollini
# License: BSD

import numpy as np

import nibabel as nib
from nilearn import datasets
from nilearn.image import iter_img, reorder_img, new_img_like, index_img
from nilearn.input_data import NiftiMasker
from sklearn.externals.joblib import Memory

from nilearn_ext.datasets import fetch_grey_matter_mask


class GreyMatterNiftiMasker(NiftiMasker):
    def __init__(self, sessions=None, smoothing_fwhm=None,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='background',
                 mask_args=None, sample_mask=None,
                 memory_level=1, memory=Memory(cachedir=None),
                 verbose=0):

        # Use grey matter mask computed for Neurovault analysis
        # ('https://github.com/NeuroVault/neurovault_analysis/')
        target_img = nib.load(fetch_grey_matter_mask())
        grey_voxels = (target_img.get_data() > 0).astype(int)
        mask_img = new_img_like(target_img, grey_voxels, copy_header=True)

        super(GreyMatterNiftiMasker, self).__init__(
            mask_img=mask_img,
            target_affine=mask_img.affine,
            target_shape=mask_img.shape,
            sessions=sessions,
            smoothing_fwhm=smoothing_fwhm,
            standardize=standardize,
            detrend=detrend,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=t_r,
            mask_strategy=mask_strategy,
            mask_args=mask_args,
            sample_mask=sample_mask,
            memory_level=memory_level,
            memory=memory,
            verbose=verbose)


class MniNiftiMasker(NiftiMasker):
    def __init__(self, sessions=None, smoothing_fwhm=None,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='background',
                 mask_args=None, sample_mask=None,
                 memory_level=1, memory=Memory(cachedir=None),
                 verbose=0):

        # Create grey matter mask from mni template
        target_img = datasets.load_mni152_template()
        grey_voxels = (target_img.get_data() > 0).astype(int)
        mask_img = new_img_like(target_img, grey_voxels, copy_header=True)

        super(MniNiftiMasker, self).__init__(
            mask_img=mask_img,
            target_affine=mask_img.affine,
            target_shape=mask_img.shape,
            sessions=sessions,
            smoothing_fwhm=smoothing_fwhm,
            standardize=standardize,
            detrend=detrend,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=t_r,
            mask_strategy=mask_strategy,
            mask_args=mask_args,
            sample_mask=sample_mask,
            memory_level=memory_level,
            memory=memory,
            verbose=verbose)


def flip_img_lr(img):
    """ Convenience function to flip image on X axis"""
    # This won't work for all image formats! But
    # does work for those that we're working with...
    assert isinstance(img, nib.nifti1.Nifti1Image)
    img = new_img_like(img, data=img.get_data()[::-1], copy_header=True)
    return img


def split_bilateral_rois(maps_img):
    """Convenience function for splitting bilateral ROIs
    into two unilateral ROIs"""

    new_rois = []

    for map_img in iter_img(maps_img):
        for hemi in ['L', 'R']:
            hemi_mask = HemisphereMasker(hemisphere=hemi)
            hemi_mask.fit(map_img)
            if hemi_mask.mask_img_.get_data().sum() > 0:
                hemi_vectors = hemi_mask.transform(map_img)
                hemi_img = hemi_mask.inverse_transform(hemi_vectors)
                new_rois.append(hemi_img.get_data())

    new_maps_data = np.concatenate(new_rois, axis=3)
    new_maps_img = new_img_like(maps_img, data=new_maps_data, copy_header=True)
    # print("Changed from %d ROIs to %d ROIs" % (maps_img.shape[-1], # This isn't right...
    #                                            new_maps_img.shape[-1]))
    return new_maps_img


def join_bilateral_rois(R_img, L_img):  # noqa
    """Convenience function for joining two unilateral ROIs
    into a bilateral ROIs"""

    joined_data = R_img.get_data() + L_img.get_data()
    return new_img_like(R_img, data=joined_data)


def get_hemi_gm_mask(hemi="L"):
    """Convenience function for getting R or L gm mask"""
    gm_img = nib.load(fetch_grey_matter_mask())
    gm_imgs = split_bilateral_rois(gm_img)
    gm_imgs_d = {hemi: index_img(gm_imgs, i)
                 for hemi, i in zip(("L", "R"), (0, 1))}

    return gm_imgs_d[hemi]



class HemisphereMasker(NiftiMasker):
    """
    Masker to segregate by hemisphere.

    Parameters
    ==========
    hemisphere: L or R

    """
    def __init__(self, mask_img=None, sessions=None, smoothing_fwhm=None,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='background',
                 mask_args=None, sample_mask=None,
                 memory_level=1, memory=Memory(cachedir=None),
                 verbose=0, hemisphere='L'):
        if hemisphere.lower() in ['l', 'left']:
            self.hemi = 'l'
        elif hemisphere.lower() in ['r', 'right']:
            self.hemi = 'r'
        else:
            raise ValueError('Hemisphere must be left or right; '
                             'got value %s' % self.hemi)

        mask_img = mask_img or nib.load(fetch_grey_matter_mask())
        target_affine = mask_img.affine
        target_shape = mask_img.shape
        super(HemisphereMasker, self).__init__(mask_img=mask_img,
                                               sessions=sessions,
                                               smoothing_fwhm=smoothing_fwhm,
                                               standardize=standardize,
                                               detrend=detrend,
                                               low_pass=low_pass,
                                               high_pass=high_pass,
                                               t_r=t_r,
                                               target_affine=target_affine,
                                               target_shape=target_shape,
                                               mask_strategy=mask_strategy,
                                               mask_args=mask_args,
                                               sample_mask=sample_mask,
                                               memory_level=memory_level,
                                               memory=memory,
                                               verbose=verbose)

    def fit(self, X=None, y=None):  # noqa
        super(HemisphereMasker, self).fit(X, y)

        # x, y, z
        hemi_mask_data = reorder_img(self.mask_img_).get_data().astype(np.bool)

        xvals = hemi_mask_data.shape[0]
        midpt = np.ceil(xvals / 2.)
        if self.hemi == 'r':
            other_hemi_slice = slice(midpt, xvals)
        else:
            other_hemi_slice = slice(0, midpt)

        hemi_mask_data[other_hemi_slice] = False
        mask_data = self.mask_img_.get_data() * hemi_mask_data
        self.mask_img_ = new_img_like(self.mask_img_, data=mask_data)

        return self


class MniHemisphereMasker(HemisphereMasker):
    """ Alias for HemisphereMasker with mask_img==Mni template"""
    def __init__(self, sessions=None, smoothing_fwhm=None,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='background',
                 mask_args=None, sample_mask=None,
                 memory_level=1, memory=Memory(cachedir=None),
                 verbose=0, hemisphere='L'):
        target_img = datasets.load_mni152_template()
        grey_voxels = (target_img.get_data() > 0).astype(int)
        mask_img = new_img_like(target_img, grey_voxels)

        super(MniHemisphereMasker, self).__init__(
            mask_img=mask_img,
            target_affine=mask_img.affine,
            target_shape=mask_img.shape,
            sessions=sessions,
            smoothing_fwhm=smoothing_fwhm,
            standardize=standardize,
            detrend=detrend,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=t_r,
            mask_strategy=mask_strategy,
            mask_args=mask_args,
            sample_mask=sample_mask,
            memory_level=memory_level,
            memory=memory,
            verbose=verbose,
            hemisphere=hemisphere)

    def mask_as_img(self, img):
        """ Convenience function to mask image, return as image."""
        X = self.fit_transform(img)  # noqa
        new_img = self.inverse_transform(X)
        return new_img_like(img, data=new_img.get_data(), copy_header=True)
