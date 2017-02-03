#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for performing and processing ROC analyses.

@author: David Samu
"""

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score

from seal.util import util
from seal.plot import pauc


# Some analysis constants.
min_sample_size = 10
n_folds = 5
n_jobs = util.get_n_cores() - 1


# %% Core ROC analysis functions.

def calc_auc(clf, x, y):
    """
    Calculate area under the curve of ROC analysis.

    y values have to be 0 and 1!
    """

    # Format x into array of arrays.
    if len(x.shape) < 2:
        x = np.array(x, ndmin=2).T

    # Fit model to data.
    clf.fit(x, y)

    # Get prediction probability of classes.
    preds = clf.predict_proba(x)

    # Select class of higher mean to be taken as one to be predicted.
    # idx = pd.Series(x, index=y).groupby(y).mean().idxmax()  # much slower :(
    idx = int(np.mean(x[y == 0]) < np.mean(x[y == 1]))
    y_pred = preds[:, idx]

    # Calculate area under the curve (AUC) using true and predicted y values.
    auc = roc_auc_score(y, y_pred)

    return auc


def ROC(x, y, n_perm=None, clf=None):
    """
    Perform ROC analysis with optional permutation test.

    y values have to be 0 and 1 for calc_auc!
    """

    # Remove NaN values.
    idx = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x, y = np.array(x[idx]), np.array(y[idx])

    # Insufficient sample size.
    if min(len(x), len(y)) < min_sample_size:
        return np.nan, None

    # Format x into array of arrays.
    x = np.array(x, ndmin=2).T

    # Default classifier.
    if clf is None:
        clf = LogisticRegression()

    # Calculate AUC of true data.
    true_auc = calc_auc(clf, x, y)

    # Permutation test.
    pvalue = None
    if n_perm is not None and n_perm > 0:

        cv = StratifiedKFold(n_folds)

        # Test significance of classification with cross-validated permutation.
        res = permutation_test_score(clf, x, y, scoring='accuracy', cv=cv,
                                     n_permutations=n_perm, n_jobs=n_jobs)
        score, perm_scores, pvalue = res

    return true_auc, pvalue


# %% Wrapper functions.

def run_ROC_over_time(rates1, rates2, n_perm=None, clf=None):
    """Run ROC analysis between two rate frames (trials by time)."""

    # Merge rates and create and target vector.
    rates = pd.concat([rates1, rates2])
    target_vec = pd.Series(len(rates.index)*[1], index=rates.index)
    target_vec[rates2.index] = 0  # y values have to be 0 and 1 for above code!

    # Default classifier.
    if clf is None:
        clf = LogisticRegression()

    # Run ROC across time.
    roc_res = pd.DataFrame([ROC(rates[t], target_vec, n_perm, clf)
                            for t in rates],
                           index=rates.columns, columns=['auc', 'pval'])

    return roc_res


def run_group_ROC_over_time(ulist, trs_list, prd, ref, n_perm=None,
                            nrate=None, verbose=True):
    """Run ROC over list of units over time."""

    aroc_dict, pval_dict = {}, {}
    for i, u in enumerate(ulist):

        # Report progress.
        if verbose:
            print(u.Name)

        # Set up params: trials, time period and rates.
        t1s, t2s = u.pr_times(prd, concat=False)
        ref_ts = u.ev_times(ref)
        trs = trs_list[i]

        # Calculate AROC on rates.
        rates1, rates2 = [u._Rates[nrate].get_rates(tr, t1s, t2s, ref_ts)
                          for tr in trs]
        aroc_res = run_ROC_over_time(rates1, rates2, n_perm)
        aroc_dict[u.Name] = aroc_res.auc
        pval_dict[u.Name] = aroc_res.pval

    # Concat into DF.
    aroc_res = pd.concat(aroc_dict, axis=1).T
    pval_res = pd.concat(pval_dict, axis=1).T

    return aroc_res, pval_res


# Calculate AROC values for both stimuli.
def calc_plot_AROC(ulist, trs_list, stims, prd_limits, stim_timings, n_perm,
                   nrate, title, ffig, fres, verbose=True,
                   remove_all_nan_units=True, remove_any_nan_times=True):
    """Calculate and plot AROC over time between specified sets of trials."""

    aroc_list, pval_list = [], []
    for stim in stims:

        # Set up params.
        prd = stim + ' half'
        ref = stim + ' on'

        # Calculate AROC DF.
        aroc, pval = run_group_ROC_over_time(ulist, trs_list[stim], prd, ref,
                                             n_perm, nrate, verbose)

        if stim == 'S2':
            aroc.columns = aroc.columns + stim_timings.loc['S2', 'on']
            pval.columns = pval.columns + stim_timings.loc['S2', 'on']

        # Truncate to provided period limits.
        tstart, tstop = [prd_limits.loc[stim, t] for t in ['start', 'stop']]
        prdcols = aroc.columns[(aroc.columns >= tstart) &
                               (aroc.columns <= tstop)]
        aroc = aroc[prdcols]
        pval = pval[prdcols]

        aroc_list.append(aroc)
        pval_list.append(pval)

    # Concatenate them them.
    aroc = pd.concat(aroc_list, axis=1)
    pval = pd.concat(pval_list, axis=1)
    aroc.columns = aroc.columns.astype(int)
    pval.columns = pval.columns.astype(int)

    # Remove units (rows) with all NaN values (not enough trials to do ROC).
    if remove_all_nan_units:
        to_keep = ~aroc.isnull().all(1)
        aroc = aroc.loc[to_keep, :]
        pval = pval.loc[to_keep, :]

    # Remove time points (columns) with any NaN value (S2 happened earlier).
    if remove_any_nan_times:
        to_keep = ~aroc.isnull().any(0)
        aroc = aroc.loc[:, to_keep]
        pval = pval.loc[:, to_keep]

    # Remove duplicated time points (overlaps of periods).
    aroc = aroc.loc[:, ~aroc.columns.duplicated()]
    pval = pval.loc[:, ~pval.columns.duplicated()]

    # Plot results on heatmap.
    plot_AROC_heatmap(aroc, stims, stim_timings, title, ffig)

    # Save results.
    if fres is not None:
        aroc_results = {'aroc': aroc, 'pval': pval, 'n_perm': n_perm,
                        'nrate': nrate}
        util.write_objects(aroc_results, fres)


def plot_AROC_heatmap(aroc, stims, stim_timings, title, ffig=None):
    """Plot AROC result matrix (units by time points) on heatmap."""

    # Init plotting.
    xlab = 'time since S1 onset (ms)'
    ylab = 'unit index'

    # Get trial periods.
    events = pd.DataFrame(columns=['time', 'label'])
    for stim in stims:
        ton, toff = stim_timings.loc[stim]
        ion, ioff = [int(np.where(np.array(aroc.columns) == t)[0])
                     for t in (ton, toff)]
        events.loc[stim+' on'] = (ion, stim+' on')
        events.loc[stim+' off'] = (ioff, stim+' off')

    # Plot on heatmap and save figure.
    pauc.plot_auc_heatmap(aroc, cmap='viridis', events=events,
                          xlbl_freq=500, ylbl_freq=50, xlab=xlab, ylab=ylab,
                          title=title, ffig=ffig)


def aroc_res_fname(res_dir, nrate, n_perm, offsets):
    """Return full path to AROC result with given parameters."""

    offset_str = '_'.join([str(int(d)) for d in offsets])
    fres = '{}res/{}_nperm_{}_offs_{}.data'.format(res_dir, nrate,
                                                   n_perm, offset_str)
    return fres


def aroc_fig_fname(res_dir, prefix, offsets, sort_prd=None):
    """Return full path to AROC result with given parameters."""

    sort_prd_str = ('sorted_by_' + util.format_to_fname(sort_prd)
                    if sort_prd is not None else 'unsorted')
    offset_str = '_'.join([str(int(d)) for d in offsets])
    ffig = '{}AROC_{}_{}_{}.png'.format(res_dir, prefix, offset_str,
                                        sort_prd_str)
    return ffig


def aroc_fig_title(between_str, monkey, task, nrec, offsets, sort_prd=None):
    """Return full path to AROC result with given parameters."""

    prd_str = 'sorted by: ' + sort_prd if sort_prd is not None else 'unsorted'
    offset_str = ', '.join([str(int(d)) for d in offsets])
    title = ('AROC between {}, {}\n'.format(between_str, prd_str) +
             'monkey: {}, task: {}'.format(monkey, task) +
             ', # recordings: {}, offsets: {} deg'.format(nrec, offset_str))
    return title


def load_aroc_res(res_dir, nrate, n_perm, offsets):
    """Load AROC results."""

    fres = aroc_res_fname(res_dir, nrate, n_perm, offsets)
    aroc_res = util.read_objects(fres, ['aroc', 'pval', 'n_perm', 'nrate'])

    return aroc_res

# TODO: from this point, functions below need updating.


# %% Post-AROC analysis functions.

def first_period(aroc_vec, prd_len, pvec=None, pth=None,
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
    earliest_hi_run = min([prd[0] for prd in hi_prds]) if hi_prds else np.nan
    earliest_lo_run = min([prd[0] for prd in lo_prds]) if lo_prds else np.nan

    # Find the earliest one, if any.
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
    T['utidx'] = [u.get_rec_ch_un_task_index() for u in Units]
    T['effect'] = eff_time[:, 0]
    T['time'] = np.array(eff_time[:, 1], dtype=float)
    T['AROC'] = [aroc_w[i, util.index(tvec_w, t)] if pd.notnull(t) else None
                 for i, t in enumerate(T['time'])]
    T['p-value'] = [pval_w[i, util.index(tvec_w, t)] if pd.notnull(t) else None
                    for i, t in enumerate(T['time'])]

    # Order by effect timing.
    isort = T.sort_values(['effect', 'time'],
                          ascending=[True, False]).index
    T['sorted index'] = np.argsort(isort)

    # Export table as Excel table
    util.write_table(T, excel_writer, sheet_name='AROC',
                     na_rep='N/A', index=False)

    return T
