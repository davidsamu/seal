#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions related to calculating direction selectivity.

@author: David Samu
"""


import numpy as np
import pandas as pd

from quantities import deg, rad

from seal.analysis import tuning
from seal.object import constants
from seal.util import util

# Constants.
res_names = ['PD', 'cPD', 'AD', 'cAD']


# %% Different functions to calculate DS and PD.

def max_DS(dirs, resp):
    """DS based on maximum rate only (legacy method)."""

    # Preferred and anti-preferred direction.
    PD = dirs[np.argmax(resp)]   # direction with maximal response
    AD = anti_dir(PD)

    # Coarse PD and AD.
    cPD, cAD = PD, AD   # same as PD and AD

    # Direction selectivity.
    PR, AR = [resp[np.where(dirs == d)[0]] for d in (PD, AD)]
    DSI = float(util.modulation_index(PR, AR)) if AR.size else np.nan

    PDres = pd.Series([PD, cPD, AD, cAD], res_names, name='max')

    return PDres, DSI


def weighted_DS(dirs, resp):
    """DS based on weighted vector average method."""

    # DS and PD: length and direction of weighted average.
    DSI, PD = polar_wmean(dirs, resp)
    cPD = coarse_dir(PD, constants.all_dirs)

    # Anti-preferred.
    AD, cAD = [anti_dir(d) for d in (PD, cPD)]

    PDres = pd.Series([PD, cPD, AD, cAD], res_names, name='weighted')

    return PDres, DSI


def tuned_DS(dirs, resp, dir0=0*deg, **kwargs):
    """DS based on Gaussian tuning curve fit."""

    # Center stimulus - response.
    dirs, _ = center_to_dir(dirs, dir0)

    # Fit Gaussian tuning curve to stimulus - response.
    fit_params, fit_res = tuning.fit_gaus_curve(dirs, resp, **kwargs)

    # DS based on Gaussian tuning curve fit.
    PD = deg_mod(dir0 + fit_params.loc['fit', 'x0'] * deg)
    cPD = coarse_dir(PD, constants.all_dirs)

    # Anti-preferred.
    AD, cAD = [anti_dir(d) for d in (PD, cPD)]

    PDres = pd.Series([PD, cPD, AD, cAD], res_names, name='tuned')

    return PDres, fit_params, fit_res


# %% Utility functions to analyze directions.

def deg2rad(v_deg):
    """Convert degrees to radians."""

    v_rad = np.pi * v_deg / 180
    return v_rad


def rad2deg(v_rad):
    """Convert radians to degrees."""

    v_deg = 180 * v_rad / np.pi
    return v_deg


def cart2pol(x, y):
    """Perform convertion from Cartesian to polar coordinates."""

    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    """Perform convertion from polar to Cartesian coordinates."""

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def deg_mod(d, max_d=360*deg):
    """Converts cirulcar value (degree) into modulo interval."""

    d = d.rescale(deg)
    d_mod = np.mod(d, max_d)
    return d_mod


def anti_dir(d):
    """Return anti-preferred (opposite) direction."""

    anti_d = deg_mod(d+180*deg)
    return anti_d


def deg_diff(d1, d2):
    """Return difference between two angles in degrees."""

    d1 = deg_mod(d1)
    d2 = deg_mod(d2)
    d = abs(d1-d2)
    d = d if d < 180*deg else 360*deg - d
    return d


def coarse_dir(origd, dirs):
    """Return direction from list that is closest to provided one."""

    deg_diffs = np.array([deg_diff(d, origd) for d in dirs])
    cd = dirs[np.argmin(deg_diffs)]

    return cd


def polar_wmean(dirs, weights=None):
    """Return weighted mean of unit length polar coordinate vectors."""

    if weights is None:
        weights = np.ones(len(dirs))

    # Remove values correspinding to NaN weights.
    idxs = np.logical_not(np.isnan(weights))
    dirs, weights = dirs[idxs], weights[idxs]

    # Uniform zero weights (i.e. no response).
    if dirs.size == 0 or np.all(weights == 0):
        return 0, np.nan*deg

    # Convert directions to Cartesian unit vectors.
    dirs_xy = np.array([pol2cart(1, d.rescale(rad)) for d in dirs])

    # Calculate mean along x and y dimensions.
    x_mean = np.average(dirs_xy[:, 0], weights=weights)
    y_mean = np.average(dirs_xy[:, 1], weights=weights)

    # Re-convert into angle in degrees.
    rho, phi = cart2pol(x_mean, y_mean)
    phi_deg = deg_mod(phi*rad)

    return rho, phi_deg


def center_to_dir(dirs, dir0=0*deg):
    """Center direction - response values to given direction by shifting."""

    if not len(dirs):
        return dirs

    if not isinstance(dirs, pd.Series):
        dirs = pd.Series(dirs)

    # Init.
    dirs_idx = dirs.index
    dirs = np.array(dirs)*deg
    idx = np.arange(len(dirs))

    # Center preferred direction.
    dirs_offset = dirs
    if dir0 is not None and not np.isnan(float(dir0)):
        dirs_offset = dirs - dir0

    # Reorganise direction and response arrays to even number of values
    # to left and right (e.g. 4 - 4 for 8 directions).
    to_right = dirs_offset < -180*deg   # indices to move to the right
    to_left = dirs_offset > 180*deg     # indices to move to the left
    center = np.invert(np.logical_or(to_right, to_left))  # indices to keep
    idx = np.hstack((idx[to_left], idx[center], idx[to_right]))

    # Shift and modulo directions.
    dirs_ctrd = deg_mod(dirs_offset[idx])
    idx_to_flip = dirs_ctrd > 180*deg
    dirs_ctrd[idx_to_flip] = dirs_ctrd[idx_to_flip] - 360*deg

    dirs_ctrd = pd.Series(dirs_ctrd, index=dirs_idx)

    return dirs_ctrd, idx
