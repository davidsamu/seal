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


# %% Functions to calculate direction selectivity over time.

def calc_DSI(u, fDSI=None, prd_pars=None, nrate=None):
    """Calculate DSI over trial periods."""

    # Init periods to analyse and DSI function to use.
    if prd_pars is None:
        prd_pars = u.init_analysis_prd()
    if fDSI is None:
        fDSI = weighted_DS

    # For each period.
    DSI = []
    for prd, (stim, ref, lbl_shift, dur) in prd_pars.iterrows():

        # Get timing and trial params.
        nrate = u.init_nrate(nrate)
        t1s, t2s = u.pr_times(prd, concat=False)
        ref_ts = u.ev_times(ref)
        trs_by_tgt = u.to_report_trials()
        trs_by_dir = u.trials_by_features(stim, 'Dir')

        # For each target feature.
        DSIprd = []
        for tgt, trs in trs_by_tgt.iteritems():

            # Subselect trials by target.
            trg_dir_trs = trs_by_dir.apply(np.intersect1d, args=(trs,))

            # Get mean rates per direction over time.
            tr_ref_ts = ref_ts - lbl_shift
            rates = [u._Rates[nrate].get_rates(trs, t1s, t2s, tr_ref_ts).mean()
                     for trs in trg_dir_trs]
            rates = pd.DataFrame(rates, index=trg_dir_trs.index)

            # Calculate direction selectivity over time.
            dirs = np.array(rates.index) * deg
            dsi = pd.Series([fDSI(dirs, rates[t])[1] for t in rates],
                            index=rates.columns)

            # Collect results.
            DSIprd.append(dsi)

        # Aggregate data of period.
        DSIprd = pd.concat(DSIprd, axis=1).T
        DSIprd.index = trs_by_tgt.index

        # Collect results.
        DSI.append(DSIprd)

    # Aggregate data across periods.
    DSI = pd.concat(DSI, axis=1)

    return DSI


# %% Functions to calculate DS and PD.

def max_DS(dirs, resp):
    """DS based on maximum rate only (legacy method)."""

    # Init.
    resp = np.array(resp)

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

    # Init.
    resp = np.array(resp)

    # DS and PD: length and direction of weighted average.
    DSI, PD = polar_wmean(dirs, resp)
    cPD = coarse_dir(PD, constants.all_dirs)

    # Anti-preferred.
    AD, cAD = [anti_dir(d) for d in (PD, cPD)]

    PDres = pd.Series([PD, cPD, AD, cAD], res_names, name='weighted')

    return PDres, DSI


def tuned_DS(dirs, resp, dir0=0*deg, **kwargs):
    """DS based on Gaussian tuning curve fit."""

    # Init.
    resp = np.array(resp)

    # Center stimulus - response.
    dirs_ctrd = center_to_dir(dirs, dir0)

    # Fit Gaussian tuning curve to stimulus - response.
    fit_params, fit_res = tuning.fit_gaus_curve(dirs_ctrd, resp, **kwargs)

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
    dirs_arr = np.array(dirs)*deg

    # Center directions around dir0 + 180*deg.
    dirs_ctrd = dirs_arr
    if dir0 is not None and not np.isnan(float(dir0)):
        dirs_ctrd = deg_mod(dirs_arr - dir0 + 180*deg) - 180*deg

    dirs_ctrd = pd.Series(dirs_ctrd, index=dirs.index)

    return dirs_ctrd
