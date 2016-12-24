#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 09:44:23 2016

Functions related to calculating direction selectivity.

@author: David Samu
"""


import numpy as np
import pandas as pd

from quantities import deg

from seal.analysis import tuning
from seal.object import constants
from seal.util import util

# Constants.
res_names = ['PD', 'cPD', 'AD', 'cAD']


# %% Utility functions.

def anti_dir(d):
    """Return anti-preferred (opposite) direction."""

    anti_d = util.deg_mod(d+180*deg)
    return anti_d


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
    DSI, PD = util.polar_wmean(dirs, resp)
    cPD = util.coarse_dir(PD, constants.all_dirs)

    # Anti-preferred.
    AD, cAD = [anti_dir(d) for d in (PD, cPD)]

    PDres = pd.Series([PD, cPD, AD, cAD], res_names, name='weighted')

    return PDres, DSI


def tuned_DS(dirs, resp, dir0=0*deg, resp_sem=None):
    """DS based on Gaussian tuning curve fit."""

    # Center stimulus - response.
    tun_res = tuning.center_pref_dir(dirs, dir0, resp, resp_sem)
    dirs_cntr, resp_cntr, resp_sem_cntr = tun_res

    # Fit Gaussian tuning curve to stimulus - response.
    fit_params, fit_res = tuning.fit_gaus_curve(dirs_cntr, resp_cntr,
                                                resp_sem_cntr)

    # DS based on Gaussian tuning curve fit.
    PD = dir0 + fit_params.loc['fit', 'x0']
    cPD = util.coarse_dir(PD, constants.all_dirs)

    # Anti-preferred.
    AD, cAD = [anti_dir(d) for d in (PD, cPD)]

    PDres = pd.Series([PD, cPD, AD, cAD], res_names, name='tuned')
    tun_res['fit_pars'] = fit_params
    tun_res['fit_res'] = fit_res

    return PDres, tun_res
