#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 10:17:45 2016

Functions for performing and processing ROC analyses.

@author: David Samu
"""

import numpy as np
from quantities import ms
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model import LogisticRegression

from seal.util import plot, util

# Extra import to be removed/refactored.
from matplotlib import gridspec as gs


# %% Core ROC analysis functions.

def calc_auc(clf, x, y):
    """Calculate area under the curve of ROC analysis."""

    if len(x.shape) < 2:
        x = np.array(x, ndmin=2).T

    # Fit model to data.
    clf.fit(x, y)

    # Change y labels based on mean x values to keep < 0.5 AUC values.
    idx = int(np.mean(x[y == 0]) > np.mean(x[y == 1]))

    # Get prediction probability of classes.
    preds = clf.predict_proba(x)[:, idx]

    # Calculate area under the curve (AUC).
    auc = roc_auc_score(y, preds)

    return auc


def ROC(x, y, n_perm=None, clf=None):
    """Perform ROC analysis with optional permutation test."""

    # Also checked in calc_auc, but put here to speed things up.
    if len(x.shape) < 2:
        x = np.array(x, ndmin=2).T

    # Default classifier.
    if clf is None:
        clf = LogisticRegression()

    # Calculate AUC of true data.
    true_auc = calc_auc(clf, x, y)

    pval = None
    if n_perm is not None and n_perm is not 0:
        # Get null-hypothesis distribution by permutation test.
        rand_splits = ShuffleSplit(len(x), n_perm, len(np.where(y == 1)[0]))
        perm_auc = np.array([calc_auc(clf, x, y[np.concatenate(s)]) for s
                             in rand_splits])
        # p-value estimate of a two-tailed test
        true_dev = abs(true_auc - 0.5)  # deviation from 0.5 (baseline)
        perm_dev = np.abs(perm_auc - 0.5)
        n_extreme = np.sum(true_dev <= perm_dev)
        pval = n_extreme / n_perm

    return true_auc, pval


# %% Wrapper functions.

def run_unit_ROC(u, get_trials, get_trials_kwargs, nrate,
                 t1=None, t2=None, n_perm=None):
    """Run ROC analysis on single unit."""

    # Get rates for two sets of trials.
    all_trials = get_trials(u, **get_trials_kwargs)
    all_rates = np.array([[u.Rates[nrate].get_rates(tr.trials, t1, t2)
                          for tr in trials] for trials in all_trials])

    # Create rate matrix and target vector.
    # Z-score across time in corresponding trial-pairs (e.g., per direction).
    Rates = []
    ZRates = []
    Targets = []
    for itr in range(all_rates.shape[1]):

        # Extract and stack rates.
        rates1 = all_rates[0, itr]
        rates2 = all_rates[1, itr]

        # Stack rates and target values across all trials.
        rates = np.vstack((rates1, rates2))
        target = np.array(rates1.shape[0]*[0] + rates2.shape[0]*[1])

        # Z-score rates.
        zrates = util.zscore_timeseries(rates)

        Rates.append(rates)
        ZRates.append(zrates)
        Targets.append(target)

    # Stack them together across trial-pairs.
    Rates = np.vstack(Rates)
    ZRates = np.vstack(ZRates)
    Targets = np.hstack(Targets)

    # Run ROC on rates and target.
    roc_res = np.array([ROC(ZRates[:, t_idx], Targets, n_perm)
                        for t_idx in range(rates.shape[1])], dtype='float')
    return roc_res


def run_AROC(Units, nrate, t1, t2, offsets, n_perm,
             get_trials, get_trials_kwargs, base_dir, force_run):
    """Run AROC and save results."""

    # Results folder and file name.
    dir_res, file_res = res_dir_file_name(base_dir, nrate, n_perm, offsets)
    prev_results = util.get_latest_file(dir_res, ext='.data')

    if not force_run and prev_results:

        f_res = dir_res + prev_results
        print('Loading in saved results from: ' + f_res)

        # Load in save results.
        objs = ['aroc', 'pval', 'tvec', 't1', 't2',
                'n_perm', 'nrate', 'offsets']
        aroc_res = util.read_objects(f_res, objs)
        aroc, pval, tvec, t1, t2, n_perm, nrate, offsets = aroc_res

    else:

        # Set up parameters for parallel computing.
        params = [(u, get_trials, get_trials_kwargs, nrate, t1, t2, n_perm)
                  for u in Units]

        # Calculate AROC and p-value by permutation test.
        res = np.array(util.run_in_pool(run_unit_ROC, params))
        aroc = res[:, :, 0]
        pval = res[:, :, 1]
        tvec = util.quantity_linspace(t1, t2, ms, aroc.shape[1])

        # Save results.
        aroc_results = {'aroc': aroc, 'pval': pval, 'tvec': tvec,
                        't1': t1, 't2': t2, 'n_perm': n_perm,
                        'nrate': nrate, 'offsets': offsets}
        f_res = dir_res + file_res
        util.write_objects(aroc_results, f_res)

    return aroc, pval, tvec, t1, t2, n_perm, nrate, offsets, f_res


def res_dir_file_name(base_dir, nrate, n_perm, offsets):
    """Return folder and file name for results."""

    # Create parameterized folder and file name.
    offset_str = '_'.join([str(int(off)) for off in offsets])
    dir_res = '{}{}_nperm_{}_offsets_{}/pickle/'.format(base_dir, nrate,
                                                        n_perm, offset_str)
    file_res = util.timestamp() + '.data'

    return dir_res, file_res


# %% Post-AROC analysis functions.

def first_period(vec, time, prd_len, pvec=None, pth=None,
                 vth_hi=0.5, vth_lo=0.5):
    """
    Return effect direction and times of earliest period with given length
    above or below value threshold (optional) and
    below p-value threshold (optional).
    """

    # Indices with significant p-values.
    sign_idx = np.ones(len(vec), dtype=bool)
    if pth is not None and pvec is not None and not np.all(np.isnan(pvec)):
        sign_idx = pvec < pth

    # Indices above and below value thresholds and with significant p values.
    sign_hi_idxs = np.logical_and(vec > vth_hi, sign_idx)
    sign_lo_idxs = np.logical_and(vec < vth_lo, sign_idx)

    # Corresponding periods with minimum length.
    hi_prds = util.periods(sign_hi_idxs, time, prd_len)
    lo_prds = util.periods(sign_lo_idxs, time, prd_len)

    # Earliest periods of each.
    earliest_hi_run = min([sp[0] for sp in hi_prds]) if hi_prds else np.nan
    earliest_lo_run = min([sp[0] for sp in lo_prds]) if lo_prds else np.nan

    # Find the earlier one, if any.
    try:
        earliest_times = [earliest_hi_run, earliest_lo_run]
        iearlier = np.nanargmin(earliest_times)
        effect_dir = ['S > D', 'D > S'][iearlier]
        t = earliest_times[iearlier]
    except ValueError:
        # No sufficiently long period of either type.
        effect_dir = 'S = D'
        t = None

    return effect_dir, t


# Plot S and D trials on raster and rate plots and add AROC, per unit.
def plot_AROC_results(Units, aroc, tvec, nrate, offsets,
                      get_trials, get_trials_kwargs, fig_dir):
    """Plots AROC results for each unit."""

    for i, u in enumerate(Units):

        # Plot standard raster-rate plot
        trials = get_trials(u, **get_trials_kwargs)
        outer_gs = gs.GridSpec(3, 1, height_ratios=[1, 1, 1])
        fig = u.plot_raster_rate(nrate, trials, outer_gs=outer_gs)

        # Remove x axis and label from rate plot
        ax_rate = fig.axes[-1]
        ax_rate.get_xaxis().set_visible(False)

        # Add axes for AROC
        gs_aroc = gs.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_gs[2])
        ax = fig.add_subplot(gs_aroc[0, 0])

        # Add chance line and grid lines
        ax.axhline(0.5, color='k', alpha=0.33, lw=1)
        for y in [0.25, 0.75]:
            ax.axhline(y, color='k', linestyle=':', alpha=0.5, lw=0.5)

        # Plot AROC
        plot.lines(tvec, aroc[i, :], ylim=[0, 1], xlab='Time (ms)',
                   ylab='AROC', ax=ax, color='m')
        plot.plot_segments(u.ExpSegments, t_unit=ms, ax=ax)
        ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
        plot.show_spines(True, False, True, False, ax)

        # Save plot
        ffig = fig_dir + u.name_to_fname() + '.png'
        plot.save_fig(fig, ffig)


# Analyse ROC results.
def results_table(Units, aroc, pval, tvec, tmin, tmax, prd_len,
                  th_hi, th_lo, pth, excel_writer=None):
    """Return restuls table with AROC effect sizes and timings."""

    # Get timing of earliest significant run of each unit.

    # Get interval of interest from results.
    t_idxs = util.indices_in_window(tvec, tmin, tmax)
    tvec_w = tvec[t_idxs]
    aroc_w = aroc[:, t_idxs]
    pval_w = pval[:, t_idxs]

    eff_time = [first_period(aroc_w[i, :], tvec_w, prd_len, pval_w[i, :], pth,
                             th_hi, th_lo) for i in range(aroc_w.shape[0])]
    eff_time = np.array(eff_time)

    # Put results into data table.
    T = pd.DataFrame()
    T['index'] = range(1, aroc_w.shape[0]+1)
    T['name'] = [u.Name for u in Units]
    T['effect'] = eff_time[:, 0]
    T['time (ms)'] = np.array(eff_time[:, 1], dtype=float)
    T['AROC'] = [aroc_w[i, util.index(tvec_w, t)] if pd.notnull(t) else None
                 for i, t in enumerate(T['time (ms)'])]
    T['p-value'] = [pval_w[i, util.index(tvec_w, t)] if pd.notnull(t) else None
                    for i, t in enumerate(T['time (ms)'])]

    # Order by effect timing.
    isort = T.sort_values(['effect', 'time (ms)'],
                          ascending=[True, False]).index
    T['sorted index'] = np.argsort(isort)

    # Export table as Excel table
    util.save_table(T, excel_writer, sheet_name='AROC',
                    na_rep='N/A', index=False)

    return T
