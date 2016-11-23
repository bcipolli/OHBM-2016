# *- encoding: utf-8 -*-
# Author: Ben Cipollini, Ami Tsucihda
# License: BSD

from collections import OrderedDict

import nibabel as nib
import numpy as np
from nilearn import datasets


def _dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o: (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}
    same = set(o for o in intersect_keys if d1[o] == d2[o])
    return added, removed, modified, same


def _is_dict_same(d1, d2):
    added, removed, modified, same = _dict_compare(d1, d2)
    return len(added) == 0 and len(removed) == 0 and len(modified) == 0 and len(same) == len(d1)


def _neurovault_dedupe(images, term_scores, strict=False, verbose=False):
    print "Deduping %d images..." % len(images)

    # map type is from the user, so not dependable.
    keys_to_compare = [
        'file_size', 'brain_coverage',
        'not_mni', 'perc_bad_voxels',
        'perc_voxels_outside']

    reduced_dicts = OrderedDict()  # store as dicts for easy manual verification
    non_dup_images = []
    binary_idx = []

    for image in images:
        reduced_image = dict([(key, image.get(key))
                              for key in keys_to_compare])
        dup_images = [_is_dict_same(reduced_image, im)
                      for key, im in reduced_dicts.items()]
        is_dup = not np.any(dup_images)
        binary_idx.append(is_dup)

        if is_dup:
            # print "Added image %s" % image['id']
            reduced_dicts[image['id']] = reduced_image
            non_dup_images.append(image)

        elif verbose:
            dup_idx = np.where(dup_images)[0][0]
            dup_key = reduced_dicts.keys()[dup_idx]
            dup_image = non_dup_images[np.find(dup_images)[0]]

            if not strict:
                hard_reject = True
            else:
                # Duplicate; verify via image data.
                d1 = nib.load(dup_image['absolute_path']).get_data()
                d1 = d1[np.logical_not(np.isnan(d1))]
                d2 = nib.load(image['absolute_path']).get_data()
                d2 = d2[np.logical_not(np.isnan(d2))]

                hard_reject = np.all(np.abs(d1 - d2) < 1E-10)

            if hard_reject:
                print "Duplicate: %s duplicates %s" % (
                    image['id'], dup_key)
            else:
                print "Duplicate metadata, but not duplicate image (%s and %s). Discarding anyway." % (
                    image['id'], dup_key)

    # Now filter each term
    binary_idx = np.asarray(binary_idx)
    for term in term_scores:
        term_scores[term] = term_scores[term][binary_idx]

    print "Kept %d of %d images" % (len(non_dup_images), len(images))
    return non_dup_images, term_scores


def _neurovault_remove_bad_images(images, term_scores, verbose=False):
    """Bad images are:
    * Images that only have positive or negative values.
    * Also remove those with <2000 unique vals (for parcellated stat maps)
    """
    print "Searching for bad data across %d images..." % len(images)
    good_images = []
    binary_idx = []

    for image in images:
        dat = nib.load(image['absolute_path']).get_data()
        dat = dat[dat != 0]
        is_rejected = np.all(dat > 0) or np.all(dat < 0) or (np.unique(dat).size < 2000)

        image['rejected'] = is_rejected
        binary_idx.append(not is_rejected)

        if not is_rejected:
            good_images.append(image)
        elif verbose:
            print "REJECT: %s %s " % (image['map_type'], image['absolute_path'])

    # Now filter each term
    binary_idx = np.asarray(binary_idx)
    for term in term_scores:
        term_scores[term] = term_scores[term][binary_idx]

    print "Kept %d of %d images" % (len(good_images), len(images))
    return good_images, term_scores


def fetch_neurovault(max_images=np.inf, query_server=True, fetch_terms=True,
                     map_types=['F map', 'T map', 'Z map'], collection_ids=tuple(),
                     image_filters=tuple(), sort_images=True, **kwargs):
    """Give meaningful defaults, extra computations."""
    # Set image filters: The filt_dict contains metadata field for the key
    # and the desired entry for each field as the value.
    # Since neurovault metadata are not always filled, it also includes any
    # images with missing values for the any given field.
    datasets.neurovault.set_logging_level(datasets.neurovault.logging.DEBUG)
    filt_dict = {'modality': 'fMRI-BOLD',
                 'is_thresholded': False, 'not_mni': False}

    def make_fun(key, val):
        return lambda img: img.get(key) == val

    image_filters = list(image_filters) + [
        lambda img: (img.get('map_type') or '') in map_types,
        lambda img: (img.get('analysis_level') or '') in ('group')
    ]
    image_filters = (list(image_filters) +
                     [make_fun(key, val) for key, val in filt_dict.items()])

    # Also remove bad collections
    bad_collects = [367,   # Single image w/ large uniform area value > 0
                    1003,  # next three collections contain stat maps on
                    1011,  # parcellated brains. Likely causes odd-looking
                    1013,  # ICA component
                    1071,  # Added Oct2016-strange-looking images
                    1889]  # Added Oct2016-extreme vals on edge
    col_ids = [-bid for bid in bad_collects]
    collection_ids = list(collection_ids) + col_ids

    # kwargs['image_filters'] = (list(kwargs.get('image_filters', [])) +
    #                            image_filters)
    # kwargs['collection_ids'] = (kwargs.get('collection_ids', []) +
    #                             collection_ids)

    # Download matching images
    def image_filter(img_metadata):
        if img_metadata.get('collection_id') in collection_ids:
            return False
        for filt in image_filters:
            if not filt(img_metadata):
                return False
        return True

    ss_all = datasets.fetch_neurovault(
        mode='download_new' if query_server else 'offline',
        max_images=max_images, image_filter=image_filter,
        fetch_neurosynth_words=fetch_terms)
    images = ss_all['images_meta']
    term_scores = dict(zip(
        ss_all.get('vocabulary', list()),
        ss_all.get('word_frequencies', np.array((0,))).T))

    # Post-fetcher filtering: remove duplicates, bad images from raw data.
    images, term_scores = _neurovault_dedupe(images, term_scores)
    images, term_scores = _neurovault_remove_bad_images(images, term_scores)

    # Stamp some collection properties onto images.
    colls = dict([(c['id'], c) for c in ss_all['collections_meta']])
    for image in images:
        image['DOI'] = colls.get(image['collection_id'], {}).get('DOI')

    if fetch_terms:
        # Clean & report term scores
        terms = np.array(term_scores.keys())
        term_matrix = np.asarray(term_scores.values())
        term_matrix[term_matrix < 0] = 0
        total_scores = np.mean(term_matrix, axis=1)

        print("Top 10 neurosynth terms from downloaded images:")
        for term_idx in np.argsort(total_scores)[-10:][::-1]:
            print('\t%-25s: %.2f' % (terms[term_idx], total_scores[term_idx]))

    if sort_images:
        idx = sorted(
            range(len(images)),
            lambda k1, k2: images[k1]['id'] - images[k2]['id'])
        images = [images[ii] for ii in idx]
        for key in term_scores:
            term_scores[key] = [term_scores[key][ti] for ti in idx]
    return images, term_scores


def fetch_grey_matter_mask():
    url = 'https://github.com/NeuroVault/neurovault_analysis/raw/master/gm_mask.nii.gz'
    mask = datasets.utils._fetch_files(
        datasets.utils._get_dataset_dir('neurovault'),
        (('gm_mask.nii.gz', url, {}),))[0]
    return mask
