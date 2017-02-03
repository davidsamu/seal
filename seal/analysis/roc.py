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


# %% Higher level functions to run AROC on a unit and group of units over time.

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


# %% Meta-functions to run and plot AROC over a different trial periods.

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


# %% Utility functions for getting file names, and import / export data.

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
    ostr = 'offset_' + '_'.join([str(int(d)) for d in offsets])
    ffig = '{}heatmap/{}/AROC_{}_{}.png'.format(res_dir, ostr, prefix,
                                                sort_prd_str)
    return ffig


def aroc_fig_title(between_str, monkey, task, nrec, offsets, sort_prd=None):
    """Return full path to AROC result with given parameters."""

    prd_str = 'sorted by: ' + sort_prd if sort_prd is not None else 'unsorted'
    ostr = ', '.join([str(int(d)) for d in offsets])
    title = ('AROC between {}, {}\n'.format(between_str, prd_str) +
             'monkey: {}, task: {}'.format(monkey, task) +
             ', # recordings: {}, offsets: {} deg'.format(nrec, ostr))
    return title


def aroc_res_table_fname(res_dir, monkey, task, nrate, n_perm, offsets,
                         sort_prd, min_len, pth, vth_hi, vth_lo):
    """Return full path to AROC results table with given parameters."""

    ostr = '_'.join([str(int(d)) for d in offsets])
    ftable = ('{}_nperm_{}_offs_{}'.format(nrate, n_perm, ostr) +
              '_prd_{}_min_len_{}_pth_{}'.format(sort_prd, int(min_len), pth) +
              '_vth_hi_{}_vth_lo_{}'.format(vth_hi, vth_lo))
    ftable = res_dir + 'tables/' + util.format_to_fname(ftable) + '.xlsx'
    return ftable


def load_aroc_res(res_dir, nrate, n_perm, offsets):
    """Load AROC results."""

    fres = aroc_res_fname(res_dir, nrate, n_perm, offsets)
    aroc_res = util.read_objects(fres, ['aroc', 'pval', 'n_perm', 'nrate'])

    return aroc_res


# %% Post-AROC analysis functions.


def get_v_ths(vth_hi, vth_low, n_perm):
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
def results_table(aroc, pval, tmin, tmax, fout=None, **kwargs):
    """Return restuls table with AROC effect sizes and timings."""

    # Get interval to sort by.
    aroc_w, pval_w = [df.ix[:, float(tmin):float(tmax)] for df in (aroc, pval)]

    # Get timing of first significant run of each unit.
    eff_time = [first_period(aroc_w.loc[uid], pval_w.loc[uid], **kwargs)
                for uid in aroc.index]
    eff_res = pd.DataFrame(eff_time, columns=['time', 'effect_dir'],
                           index=aroc.index)

    # Sort by effect timing.
    sorted_dfs = [eff_res.loc[eff_res.effect_dir == eff].sort_values('time')
                  for eff in ('high', 'none', 'low')]
    sorted_eff_res = pd.concat(sorted_dfs)

    # Export table as Excel table
    util.save_sheets([sorted_eff_res], 'effect results', fout)

    return sorted_eff_res
