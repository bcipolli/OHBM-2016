"""
This script helps visualize brains, to validate
what looks good and what looks like crap.
"""

import os.path as op
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn.plotting import plot_stat_map
from sklearn.externals.joblib import Memory

from match import get_dataset
from nilearn_ext.image import clean_img, cast_img
from nilearn_ext.masking import GreyMatterNiftiMasker
from nilearn_ext.plotting import save_and_close


def qc_image_metadata(images):
    for key in sorted(images[0].keys()):
        unique_vals = np.unique([im.get(key, 'blue') for im in images])
        print("%s (%d): " % (key, len(unique_vals)),
              unique_vals[:5])
        print("Sample image with missing value",
              [im['url'] for im in images if key not in im][-1:])
        print("")


def qc_image_data(dataset, images, plot_dir='qc'):
    # Get ready
    masker = GreyMatterNiftiMasker(memory=Memory(cachedir='nilearn_cache')).fit()
    if op.exists(plot_dir):  # Delete old plots.
        shutil.rmtree(plot_dir)

    # Dataframe to contain summary metadata for neurovault images
    if dataset == 'neurovault':
        fetch_summary = pd.DataFrame(
            columns=('Figure #', 'col_id', 'image_id', 'name',
                     'modality', 'map_type', 'analysis_level',
                     'is_thresholded', 'not_mni', 'brain_coverage',
                     'perc_bad_voxels', 'perc_voxels_outside'))

    for ii, image in enumerate(images):
        im_path = image['absolute_path']
        if im_path is None:
            continue

        ri = ii % 4  # row i
        ci = (ii / 4) % 4  # column i
        pi = ii % 16 + 1  # plot i
        fi = ii / 16  # figure i

        if ri == 0 and ci == 0:
            fh = plt.figure(figsize=(16, 10))
            print('Plot %03d of %d' % (fi + 1, np.ceil(len(images) / 16.)))
        ax = fh.add_subplot(4, 4, pi)
        title = "%s%s" % (
            '(X) ' if image['rejected'] else '', op.basename(im_path))

        if dataset == 'neurovault':
            fetch_summary.loc[ii] = [
                'fig%03d' % (fi + 1), image.get('collection_id'),
                image.get('id'), title, image.get('modality'),
                image.get('map_type'), image.get('analysis_level'),
                image.get('is_thresholded'), image.get('not_mni'),
                image.get('brain_coverage'), image.get('perc_bad_voxels'),
                image.get('perc_voxels_outside')]

        # Images may fail to be transformed, and are of different shapes,
        # so we need to trasnform one-by-one and keep track of failures.
        img = cast_img(im_path, dtype=np.float32)
        img = clean_img(img)
        try:
            img = masker.inverse_transform(masker.transform(img))
        except Exception as e:
            print("Failed to mask/reshape image %s: %s" % (title, e))

        plot_stat_map(img, axes=ax, black_bg=True, title=title, colorbar=False)

        if (ri == 3 and ci == 3) or ii == len(images) - 1:
            out_path = op.join(plot_dir, 'fig%03d.png' % (fi + 1))
            save_and_close(out_path)

    # Save fetch_summary
    if dataset == 'neurovault':
        fetch_summary.to_csv(op.join(plot_dir, 'fetch_summary.csv'))


def _dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o: (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}
    same = set(o for o in intersect_keys if d1[o] == d2[o])
    return added, removed, modified, same


if __name__ == '__main__':
    import warnings
    from argparse import ArgumentParser

    # Look for image computation errors
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('error', RuntimeWarning)  # Detect bad NV images

    # Arg parsing
    parser = ArgumentParser(description="Really?")
    parser.add_argument('check', nargs='?', default='data',
                        choices=('data', 'metadata'))
    parser.add_argument('--offline', action='store_true', default=False)
    parser.add_argument('--dataset', nargs='?', default='neurovault',
                        choices=['neurovault', 'abide', 'nyu'])
    parser.add_argument('--max-images', nargs='?', default=np.inf, type=int)
    args = vars(parser.parse_args())

    # Alias args
    check = args.pop('check')
    query_server = not args.pop('offline')
    dataset = args.pop('dataset')
    if dataset == 'neurovault':
        args['fetch_terms'] = False
    images = get_dataset(query_server=query_server, dataset=dataset, **args)[0]

    if check == 'data':
        qc_image_data(images=images, dataset=dataset)
    else:
        qc_image_metadata(images=images)

    plt.show()
