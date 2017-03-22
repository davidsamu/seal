# -*- coding: utf-8 -*-
"""
Functions to analyze ROC results.

@author: David Samu
"""

import numpy as np
import pandas as pd

from seal.util import util


def get_v_ths(vth_hi, vth_lo, n_perm):
    """
    Returns default high and low value thresholds, depending on whether
    permutation test has been done.
    """

    vth_hi = vth_hi if (n_perm == 0) or (n_perm is None) else 0.5
    vth_lo = vth_lo if (n_perm == 0) or (n_perm is None) else 0.5

    return vth_hi, vth_lo


def first_period(vauc, pvec, min_len, pth=None, vth_hi=0.5, vth_lo=0.5):
    """
    Return effect direction and times of earliest period with given length
    above or below value threshold and below p-value threshold (both optional).
    """

    # Indices with significant p-values.
    sign_idx = pd.Series(True, index=vauc.index, name=vauc.name)
    if ((pth is not None) and (pvec is not None)
        and (not pvec.isnull().all().all())):
        sign_idx = pvec < pth

    # Indices above and below value thresholds and with significant p values.
    sig_hi = (vauc >= vth_hi) & sign_idx
    sig_lo = (vauc <= vth_lo) & sign_idx

    # Corresponding periods with minimum length.
    hi_prds = util.long_periods(sig_hi, min_len)
    lo_prds = util.long_periods(sig_lo, min_len)

    # Earliest periods of each.
    first_hi_run = pd.Series(hi_prds.apply(min)).min()
    first_lo_run = pd.Series(lo_prds.apply(min)).min()

    # Do different runs exist at all?
    hi_run = not np.isnan(first_hi_run)
    lo_run = not np.isnan(first_lo_run)

    # Select first significant period and effect direction, if any exists.
    if hi_run and lo_run:
        t = np.nanmin([first_hi_run, first_lo_run])
        effect_dir = 'high' if first_hi_run < first_lo_run else 'low'
    elif hi_run and not lo_run:
        t, effect_dir = first_hi_run, 'high'
    elif lo_run and not hi_run:
        t, effect_dir = first_lo_run, 'low'
    else:
        t, effect_dir = np.nan, 'none'

    return t, effect_dir


# Analyse ROC results.
def sort_by_time(aroc, pval, tmin, tmax, merge_hi_lo=False, fout=None,
                 **kwargs):
    """Return restuls table with AROC effect sizes and timings."""

    # Get interval to sort by.
    aroc_w, pval_w = [df.ix[:, float(tmin):float(tmax)] for df in (aroc, pval)]

    # Get timing of first significant run of each unit.
    eff_time = [first_period(aroc_w.loc[uid], pval_w.loc[uid], **kwargs)
                for uid in aroc.index]
    eff_res = pd.DataFrame(eff_time, columns=['time', 'effect_dir'],
                           index=aroc.index)

    # Sort by effect timing.
    effs_list = ([['high', 'low'], ['none']] if merge_hi_lo else
                 [['high'], ['none'], ['low']])
    sorted_dfs = [eff_res.loc[eff_res.effect_dir.isin(effs)].sort_values('time')
                  for effs in effs_list]
    sorted_res = pd.concat(sorted_dfs)

    # Export table as Excel table
    util.save_sheets([sorted_res], 'effect results', fout)

    return sorted_res
