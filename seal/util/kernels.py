# -*- coding: utf-8 -*-
"""
Functions to create kernels for firing rate estimation.

@author: David Samu
"""

import warnings

import numpy as np
import pandas as pd

from quantities import ms
from elephant.kernels import GaussianKernel, RectangularKernel


# Constants.
kstep = 10 * ms


# %% Functions to create kernels.

def rect_width_from_sigma(sigma):
    """Return rectangular kernel width from sigma."""

    width = 2 * np.sqrt(3) * sigma.rescale(ms)
    return width


def sigma_from_rect_width(width):
    """Return sigma from rectangular kernel width."""

    sigma = width.rescale(ms) / 2 / np.sqrt(3)
    return sigma


def rect_kernel(width):
    """Create rectangular kernel with given width."""

    sigma = sigma_from_rect_width(width)
    rk = RectangularKernel(sigma=sigma)
    return rk


def gaus_kernel(sigma):
    """Create Gaussian kernel with given sigma."""

    gk = GaussianKernel(sigma=sigma)
    return gk


def create_kernel(kerneltype, width):
    """Create kernel of given type with given width."""

    if kerneltype in ('G', 'Gaussian'):
        fkernel = gaus_kernel

    elif kerneltype in ('R', 'Rectangular'):
        fkernel = rect_kernel

    else:
        warnings.warn('Unrecognised kernel type %s'.format(kerneltype) +
                      ', returning Rectangular kernel.')
        fkernel = rect_kernel

    krnl = fkernel(width)

    return krnl


def kernel(kname):
    """
    Return kernel created with parameters encoded in name (type and width).
    """

    kerneltype = kname[0]
    width = int(kname[1:]) * ms
    kern = create_kernel(kerneltype, width)

    return kern


def kernel_set(kpars):
    """Return set of kernels specified in list of knames."""

    kset = pd.DataFrame([(kernel(kname), kstep) for kname, kstep in kpars],
                        index=[kp[0] for kp in kpars],
                        columns=['kernel', 'step'])
    return kset


# %% Predefined example kernel sets.

R100_kernel = kernel_set([('R100', 10*ms)])
G20_kernel = kernel_set([('G20', 10*ms)])
RG_kernels = kernel_set([('G20', 10*ms), ('R100', 10*ms)])
R2G2_kernels = kernel_set([('G20', 10*ms), ('G40', 10*ms),
                           ('R100', 10*ms), ('R200', 10*ms)])
shrtR_kernels = kernel_set([('R50', 10*ms), ('R75', 10*ms), ('R100', 10*ms)])
lngR_kernels = kernel_set([('R100', 10*ms), ('R200', 10*ms), ('R500', 10*ms)])
