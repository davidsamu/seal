# -*- coding: utf-8 -*-
"""
Functions to analyze and plot ROC results.

@author: David Samu
"""

import numpy as np
import pandas as pd

from quantities import ms

from seal.util import util, constants
from seal.plot import pauc
from seal.roc import rocutil


# %% Analyse ROC results.

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
    if (pth is not None) and (pvec is not None) and (not pvec.isnull().all()):
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


def sort_by_time(aroc, pval, tmin, tmax, merge_hi_lo=False, fout=None,
                 **kwargs):
    """Return restuls table with AROC effect sizes and timings."""

    # Get interval to sort by.
    aroc_w, pval_w = [df.ix[:, float(tmin):float(tmax)] for df in (aroc, pval)]

    # Get timing of first significant run of each unit.
    efftime = [first_period(aroc_w.loc[uid], pval_w.loc[uid], **kwargs)
               for uid in aroc.index]
    effres = pd.DataFrame(efftime, columns=['time', 'effect_dir'],
                          index=aroc.index)

    # Sort by effect timing.
    effs_list = ([['high', 'low'], ['none']] if merge_hi_lo else
                 [['high'], ['none'], ['low']])
    sorted_dfs = [effres.loc[effres.effect_dir.isin(effs)].sort_values('time')
                  for effs in effs_list]
    sorted_res = pd.concat(sorted_dfs)

    # Export table as Excel table
    util.save_sheets([sorted_res], 'effect results', fout)

    return sorted_res


# %% Plot AROC results.

def plot_AROC_heatmap(aroc, prd_pars, title, cmap='jet', ffig=None):
    """Plot AROC result matrix (units by time points) on heatmap."""

    # Init plotting.
    xlab = 'time since S1 onset (ms)'
    ylab = 'unit index'

    # Get trial periods.
    events = pd.DataFrame(columns=['time', 'label'])
    for stim in prd_pars.index:
        ton, toff = prd_pars.loc[stim, ['stim_start', 'stim_stop']]
        ion, ioff = [int(np.where(np.array(aroc.columns) == t)[0])
                     for t in (ton, toff)]
        events.loc[stim+' on'] = (ion, stim+' on')
        events.loc[stim+' off'] = (ioff, stim+' off')

    # Plot on heatmap and save figure.
    pauc.plot_auc_heatmap(aroc, cmap=cmap, events=events,
                          xlbl_freq=500, ylbl_freq=50, xlab=xlab, ylab=ylab,
                          title=title, ffig=ffig)


def plot_ROC_heatmap(aroc, pval, task, nrate, tstep, n_perm, sort_prds,
                     prd_pars, offsets, res_dir, prefix, btw_str, pth=0.05,
                     min_len=30*ms, vth_hi=0.7, vth_lo=0.3, cmaps=['coolwarm'],
                     merge_hi_lo=False, flip_aroc_vals=False):
    """Plot heatmap sorted by timing of first significant period."""

    # For each period.
    for sort_prd in sort_prds:

        if sort_prd == 'unsorted':
            aroc_sorted = aroc
        else:
            tmin, tmax = constants.fixed_tr_prds.loc[sort_prd]

            # Sorted by effect size and save into table.
            ftable = rocutil.aroc_table_fname(res_dir, task, nrate, tstep,
                                              n_perm, offsets, sort_prd,
                                              min_len, pth, vth_hi, vth_lo)
            sres = sort_by_time(aroc, pval, tmin, tmax, min_len=min_len,
                                pth=pth, vth_hi=vth_hi, vth_lo=vth_lo,
                                merge_hi_lo=merge_hi_lo, fout=ftable)

            # Sort AROC matrix.
            aroc_sorted = aroc.loc[sres.index]

        if flip_aroc_vals:
            idx = aroc_sorted < 0.5
            aroc_sorted[idx] = 1 - aroc_sorted[idx]

        # Replot heatmap with sorted units.
        nunits = len(aroc)
        title = rocutil.aroc_fig_title(btw_str, task, nunits,
                                       offsets, sort_prd)
        for cmap in cmaps:
            ffig = rocutil.aroc_fig_fname(res_dir, prefix, offsets,
                                          cmap, sort_prd)
            plot_AROC_heatmap(aroc_sorted, prd_pars, title, cmap, ffig)
