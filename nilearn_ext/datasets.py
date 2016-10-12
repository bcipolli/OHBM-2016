# *- encoding: utf-8 -*-
# Author: Ben Cipollini, Ami Tsucihda
# License: BSD

import numpy as np
from nilearn import datasets


def fetch_neurovault(max_images=np.inf, query_server=True, fetch_terms=True,
                     map_types=['F map', 'T map', 'Z map'], collection_ids=tuple(),
                     image_filters=tuple(), sort_images=True):
    """Give meaningful defaults, extra computations."""
    # Set image filters: The filt_dict contains metadata field for the key
    # and the desired entry for each field as the value.
    # Since neurovault metadata are not always filled, it also includes any
    # images with missing values for the any given field.
    filt_dict = {'modality': 'fMRI-BOLD', 'analysis_level': 'group',
                 'is_thresholded': False, 'not_mni': False}

    def make_fun(key, val):
        return lambda img: (img.get(key) or '') in ('', val)
    image_filters = (list(image_filters) +
                     [make_fun(key, val) for key, val in filt_dict.items()])

    # Also remove bad collections
    bad_collects = [367,   # Single image w/ large uniform area value > 0
                    1003,  # next three collections contain stat maps on
                    1011,  # parcellated brains. Likely causes odd-looking
                    1013]  # ICA component
    collection_ids = list(collection_ids) + bad_collects

    # Download matching images
    def image_filter(img_metadata):
        if img_metadata.get('collection_id') in collection_ids:
            return False
        for filt in image_filters:
            if not filt(img_metadata):
                return False
        return True
        #    query_server=query_server, map_types=map_types,, **kwargs)

    ss_all = datasets.fetch_neurovault(
        mode='download_new' if query_server else 'offline',
        max_images=max_images, image_filter=image_filter,
        fetch_neurosynth_words=fetch_terms)
    images = ss_all['images_meta']

    # Stamp some collection properties onto images.
    colls = dict([(c['id'], c) for c in ss_all['collections_meta']])
    for image in images:
        image['DOI'] = colls.get(image['collection_id'], {}).get('DOI')

    if not fetch_terms:
        term_scores = None
    else:
        term_scores = ss_all['terms']

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
        if term_scores:
            term_scores = [term_scores[ti] for ti in idx]
    return images, term_scores


def fetch_grey_matter_mask():
    url = 'https://github.com/NeuroVault/neurovault_analysis/raw/master/gm_mask.nii.gz'
    mask = datasets.utils._fetch_files(
        datasets.utils._get_dataset_dir('neurovault'),
        (('gm_mask.nii.gz', url, {}),))[0]
    return mask
