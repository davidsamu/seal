#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions related to analyzing unit-wise, pair-wise and higher-order
variability.(Fano factor, noise correlation, etc).

@author: David Samu
"""

from itertools import combinations

import numpy as np
import pandas as pd

from quantities import ms

from seal.analysis import stats
from seal.util import util, constants


# %% Functions to calculate Fano factor.

def fano_factor(v):
    """
    Calculate Fano factor of vector of spike counts.
    IMPORTANT: v should be vector of spike counts, not rates!
    """

    varv, meanv = np.var(v),  np.mean(v)

    if meanv == 0:
        return np.nan

    fanofac = varv / meanv
    return fanofac


def fano_factor_prd(spk_cnt_df):
    """Calculate Fano factor at each time point of rate DataFrame."""

    ff_prd = pd.Series([fano_factor(spk_cnt_df[t])
                        for t in spk_cnt_df.columns],
                       index=spk_cnt_df.columns)

    return ff_prd


def fano_factor_trial_sets(u, trs_ser, win_width, t1s, t2s, ref_ts):
    """Calculate Fano factor for each set of trials."""

    # TODO: calculate this in pool!
    nrate = 'R' + str(int(win_width))
    dff = {}
    for name, trs in trs_ser.items():
        rate = u._Rates[nrate].get_rates(trs, t1s, t2s, ref_ts)
        spk_cnt = rate * (float(win_width.rescale(ms)) / 1000)
        dff[name] = fano_factor_prd(spk_cnt)

    ff = pd.concat(dff, axis=1).T

    return ff


# %% Functions to calculate correlations.

def spike_count_corr_prd(spk_cnt1, spk_cnt2):
    """Calculate mean spike count correlation."""

    idxs = spk_cnt1.index.intersection(spk_cnt2.index)

    sp_corrs = [spk_cnt1.loc[idx].corr(spk_cnt2.loc[idx]) for idx in idxs]
    msc = np.array(sp_corrs).mean()

    return msc


def spike_count_corr_trial_sets(u1, u2, trs_ser, win_width, t1s, t2s, ref_ts):
    """Calculate pairwise correlation for each set of trials."""

    nrate = 'R' + str(int(win_width))
    dsc = {}
    for name, trs in trs_ser.items():
        r1, r2 = [u._Rates[nrate].get_rates(trs, t1s, t2s, ref_ts)
                  for u in (u1, u2)]
        spk_cnt1, spk_cnt2  = [r * (float(win_width.rescale(ms)) / 1000)
                               for r in (r1, r2)]
        dsc[name] = spike_count_corr_prd(spk_cnt1, spk_cnt2)

    sc = pd.Series(dsc)

    return sc


def spike_count_corr_unit_pair(u1, u2, win_width, prds):
    """Calculate spike count correlation for unit pair."""

    sc = {}
    for prd in prds.index:

        ref_ev = prds.loc[prd, 'ref_ev']

        # Split trials by stimulus direction.
        stim = ref_ev[:2]
        dir_trs = u1.trials_by_param((stim, 'Dir'))

        # Set up params of period timing.
        t1s, t2s = u1.pr_times(prd, concat=False)
        ref_ts = u1.ev_times(ref_ev)

        # Calculate pairwise correlation
        # for each direction during period.
        prd_sc = spike_count_corr_trial_sets(u1, u2, dir_trs, win_width,
                                             t1s, t2s, ref_ts)
        sc[prd] = prd_sc

    sc = pd.concat(sc, axis=1)

    return sc


def spike_count_corr_unit_array(UA, tasks, win_width, prds=None):
    """Calculate spike count correlation for each recording and task in UA."""

    # Init.
    if win_width is None:
        win_width = 200 * ms
    if prds is None:
        prds = constants.classic_tr_prds
    ich = constants.utid_names.index('ch')

    sc_res = {}
    for task in tasks:
        for rec in UA.recordings():
            utids = UA.utids([task], [rec], as_series=True)

            # Get all unit pairs.
            # Skip units on the same channel to minimize
            # inflation of correlation values.
            upairs = [(UA.get_unit_by_utid(utid1), UA.get_unit_by_utid(utid2),
                       win_width, prds)
                      for utid1, utid2 in combinations(utids, 2)
                      if utid1[ich] != utid2[ich]]

            # Run in pool.
            sc_res[rec, task] = util.run_in_pool(spike_count_corr_unit_pair,
                                                 upairs)

    pd.concat(sc_res)
    # FINISH THIS!!!
