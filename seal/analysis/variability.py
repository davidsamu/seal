#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions related to analyzing unit-wise, pair-wise and higher-order
variability.(Fano factor, noise correlation, etc).

@author: David Samu
"""

import numpy as np
import pandas as pd


# %% Functions to calculate Fano factor.

def fano_factor(v):
    """Calculate Fano factor of vector of spike counts."""

    varv, meanv = np.var(v),  np.mean(v)

    if meanv == 0:
        return np.nan

    fanofac = varv / meanv
    return fanofac


def fano_factor_prd(rate):
    """Calculate Fano factor at each time point of rate DataFrame."""

    ff_prd = pd.Series([fano_factor(rate[t]) for t in rate.columns],
                       index=rate.columns)

    return ff_prd


def fano_factor_trial_sets(u, trs_ser, nrate, t1s, t2s, ref_ts):
    """Calculate Fano factor for each set of trials."""

    dff = {}
    for name, trs in trs_ser.items():
        rate = u._Rates[nrate].get_rates(trs, t1s, t2s, ref_ts)
        dff[name] = fano_factor_prd(rate)

    ff = pd.concat(dff, axis=1).T

    return ff
