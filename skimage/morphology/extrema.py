"""extrema.py - local minima and maxima

This module provides functions to find local maxima and minima of an image.
Here, local maxima (minima) are defined as connected sets of pixels with equal
grey level which is strictly greater (smaller) than the grey level of all
pixels in direct neighborhood of the connected set. In addition, the module
provides the closely related functions h-maxima and h-minima.

Soille, P. (2003). Morphological Image Analysis: Principles and Applications
(2nd ed.). Springer-Verlag New York, Inc.

Original author: Thomas Walter
"""

import greyreconstruct
import numpy as np


def _add_constant_clip(img, const_value):
    """Adds a constant to the image and handles overflow issues
    for integer typed images.
    """
    array_info = np.iinfo(img.dtype)
    max_dtype = array_info.max
    min_dtype = array_info.min

    if const_value > (max_dtype - min_dtype):
        raise ValueError("The added constant is not compatible"
                         "with the image data type.")

    result = img + const_value
    result[img > max_dtype-const_value] = max_dtype
    return(result)


def _subtract_constant_clip(img, const_value):
    """Subtracts a constant from the image and handles overflow issues
    for integer typed images.
    """
    array_info = np.iinfo(img.dtype)
    max_dtype = array_info.max
    min_dtype = array_info.min
    if const_value > (max_dtype-min_dtype):
        raise ValueError("The subtracted constant is not compatible"
                         "with the image data type.")

    result = img - const_value
    result[img < (const_value + min_dtype)] = min_dtype
    return(result)


def h_maxima(img, h, selem=None):
    """Determines all maxima of the image with height >= h.

    The local maxima are defined as connected sets of pixels with equal
    grey level strictly greater than the grey level of all pixels in direct
    neighborhood of the set.

    A local maximum M of height h is a local maximum for which
    there is at least one path joining M with a higher maximum on which the
    minimal value is f(M) - h (i.e. the values along the path are not
    decreasing by more than h with respect to the maximum's value) and no
    path for which the minimal value is greater.

    Parameters
    ----------
    img : ndarray
        The input image for which the maxima are to be calculated.
    h : unsigned integer
        The minimal height of all extracted maxima.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
        Defaults to a 3x3 square (8-connectivity).

    Returns
    -------
    h-maxima : ndarray
       The maxima of height >= h. The result image is a binary image, where
       pixels belonging to the selected maxima take value 1, the others
       take value 0.

    See also
    --------
    skimage.morphology.extrema.h_minima
    skimage.morphology.extrema.local_maxima
    skimage.morphology.extrema.local_minima

    References
    ----------
    .. [1] Soille, P., "Morphological Image Analysis: Principles and
           Applications", 2nd edition (2003), ISBN 3540429883.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import extrema

    We create an image (quadratic function with a maximum in the center and
    4 additional constant maxima.
    The heights of the maxima are: 1, 21, 41, 61, 81, 101

    >>> w = 10
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 20 - 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:4,2:4] = 40; f[2:4,7:9] = 60; f[7:9,2:4] = 80; f[7:9,7:9] = 100
    >>> f = f.astype(np.int)

    We can calculate all maxima with a height of at least 40:

    >>> maxima = extrema.h_maxima(f, 40)

    The resulting image will contain 4 local maxima.
    """
    if np.issubdtype(img.dtype, 'half'):
        shifted_img = img - h
        resolution = np.finfo(img.dtype).resolution
    else:
        shifted_img = _subtract_constant_clip(img, h)
        resolution = 0

    rec_img = greyreconstruct.reconstruction(shifted_img, img,
                                             method='dilation', selem=selem)
    residue_img = img - rec_img
    result_img = np.zeros(img.shape)
    result_img[residue_img >= h-10*resolution] = 1
    return result_img.astype(np.uint8)


def h_minima(img, h, selem=None):
    """Determines all minima of the image with depth >= h.

    The local minima are defined as connected sets of pixels with equal
    grey level strictly smaller than the grey levels of all pixels in direct
    neighborhood of the set.

    A local minimum M of depth h is a local minimum for which
    there is at least one path joining M with a deeper minimum on which the
    maximal value is f(M) + h (i.e. the values along the path are not
    increasing by more than h with respect to the minimum's value) and no
    path for which the maximal value is smaller.

    Parameters
    ----------
    img : ndarray
        The input image for which the minima are to be calculated.
    h : unsigned integer
        The minimal depth of all extracted minima.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
        Defaults to a 3x3 square (8-connectivity).

    Returns
    -------
    h-minima : ndarray
       The minima of depth >= h. The result image is a binary image, where
       pixels belonging to the selected minima take value 1, the other pixels
       take value 0.

    See also
    --------
    skimage.morphology.extrema.h_maxima
    skimage.morphology.extrema.local_maxima
    skimage.morphology.extrema.local_minima

    References
    ----------
    .. [1] Soille, P., "Morphological Image Analysis: Principles and
           Applications", 2nd edition (2003), ISBN 3540429883.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import extrema

    We create an image (quadratic function with a minimum in the center and
    4 additional constant maxima.
    The depth of the minima are: 1, 21, 41, 61, 81, 101

    >>> w = 10
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 180 + 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:4,2:4] = 160; f[2:4,7:9] = 140; f[7:9,2:4] = 120; f[7:9,7:9] = 100
    >>> f = f.astype(np.int)

    We can calculate all minima with a depth of at least 40:

    >>> minima = extrema.h_minima(f, 40)

    The resulting image will contain 4 local minima.

    """
    if np.issubdtype(img.dtype, 'half'):
        shifted_img = img + h
        resolution = np.finfo(img.dtype).resolution
    else:
        shifted_img = _add_constant_clip(img, h)
        resolution = 0

    rec_img = greyreconstruct.reconstruction(shifted_img, img,
                                             method='erosion', selem=selem)
    residue_img = rec_img - img
    result_img = np.zeros(img.shape)
    result_img[residue_img >= h - 10*resolution] = 1
    return result_img.astype(np.uint8)


def find_min_diff(img):
    """
    Finds the minimal difference of grey levels inside the image.
    """
    img_vec = np.unique(img.flatten())
    img_vec.sort()
    min_diff = np.min(img_vec[1:] - img_vec[:-1])
    return min_diff


def local_maxima(img, selem=None):
    """Determines all local maxima of the image.

    The local maxima are defined as connected sets of pixels with equal
    grey level strictly greater than the grey levels of all pixels in direct
    neighborhood of the set.

    For integer typed images, this corresponds to the h-maxima with h=1.
    For float typed images, h is determined as the smallest difference
    between grey levels.

    Parameters
    ----------
    img : ndarray
        The input image for which the maxima are to be calculated.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
        Defaults to a 3x3 square (8-connectivity).

    Returns
    -------
    local_maxima : ndarray
       All local maxima of the image. The result image is a binary image,
       where pixels belonging to local maxima take value 1, the other pixels
       take value 0.

    See also
    --------
    skimage.morphology.extrema.h_minima
    skimage.morphology.extrema.h_maxima
    skimage.morphology.extrema.local_minima

    References
    ----------
    .. [1] Soille, P., "Morphological Image Analysis: Principles and
           Applications", 2nd edition (2003), ISBN 3540429883.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import extrema

    We create an image (quadratic function with a maximum in the center and
    4 additional constant maxima.
    The heights of the maxima are: 1, 21, 41, 61, 81, 101

    >>> w = 10
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 20 - 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:4,2:4] = 40; f[2:4,7:9] = 60; f[7:9,2:4] = 80; f[7:9,7:9] = 100
    >>> f = f.astype(np.int)

    We can calculate all local maxima:

    >>> maxima = extrema.local_maxima(f)

    The resulting image will contain all 6 local maxima.

    """
    if np.issubdtype(img.dtype, 'half'):
        # find the minimal grey level difference
        h = find_min_diff(img)
    else:
        h = 1
    local_maxima = h_maxima(img, h, selem=selem)
    return local_maxima


def local_minima(img, selem=None):
    """Determines all local minima of the image.

    The local minima are defined as connected sets of pixels with equal
    grey level strictly smaller than the grey levels of all pixels in direct
    neighborhood of the set.

    For integer typed images, this corresponds to the h-minima with h=1.
    For float typed images, h is determined as the smallest difference
    between grey levels.

    Parameters
    ----------
    img : ndarray
        The input image for which the minima are to be calculated.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
        Defaults to a 3x3 square (8-connectivity).

    Returns
    -------
    local_minima : ndarray
       All local minima of the image. The result image is a binary image,
       where pixels belonging to local minima take value 1, the other pixels
       take value 0.

    See also
    --------
    skimage.morphology.extrema.h_minima
    skimage.morphology.extrema.h_maxima
    skimage.morphology.extrema.local_maxima

    References
    ----------
    .. [1] Soille, P., "Morphological Image Analysis: Principles and
           Applications", 2nd edition (2003), ISBN 3540429883.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import extrema

    We create an image (quadratic function with a minimum in the center and
    4 additional constant maxima.
    The depth of the minima are: 1, 21, 41, 61, 81, 101

    >>> w = 10
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 180 + 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:4,2:4] = 160; f[2:4,7:9] = 140; f[7:9,2:4] = 120; f[7:9,7:9] = 100
    >>> f = f.astype(np.int)

    We can calculate all local minima:

    >>> minima = extrema.local_minima(f)

    The resulting image will contain all 6 local minima.
    """
    if np.issubdtype(img.dtype, 'half'):
        # find the minimal grey level difference
        h = find_min_diff(img)
    else:
        h = 1
    local_minima = h_minima(img, h, selem=selem)
    return local_minima
