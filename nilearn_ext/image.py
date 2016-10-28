# *- encoding: utf-8 -*-
# Author: Ben Cipollini
# License: BSD

import numpy as np

from nilearn._utils import check_niimg
from nilearn.image import new_img_like


def clean_img(img):
    """Remove nan/inf entries."""
    img = check_niimg(img)
    img_data = img.get_data()
    img_data[np.isnan(img_data)] = 0
    img_data[np.isinf(img_data)] = 0
    return new_img_like(img, img_data, copy_header=True)


def cast_img(img, dtype=np.float32):
    """Cast image to the specified dtype"""
    img = check_niimg(img)
    img_data = img.get_data().astype(dtype)
    return new_img_like(img, img_data, copy_header=True)
