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
from seal.roc import rocutil, rocpost


# Some analysis constants.
min_sample_size = 10
n_folds = 5
n_jobs = 1  # or util.get_n_cores() - 1


# DataFrame containing analysis params per feature to analyze.
feat_pars = {'DS': ('pref_anti_dirs', 'pref and anti dirs', None, None),
             # 'CE': ('S_D_trials', 'S and D trials', 1900*ms, None)}
             'CE': ('S_D_trials', 'S and D trials', None, None)}
feat_pars = pd.DataFrame(feat_pars, index=('prefix', 'btw_str', 't1', 't2')).T


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

    y values have to be 0 and 1 for calc_auc()!
    """

    # Remove NaN values.
    idx = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x, y = np.array(x[idx]), np.array(y[idx])

    # Insufficient sample size or not exactly two values to classify.
    n_yvals = len(np.unique(y))
    if (min(len(x), len(y)) < min_sample_size) or (n_yvals != 2):
        if n_yvals > 2:
            print('More than two values to classify:' + str(np.unique(y)))
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
    target_vec[rates2.index] = 0  # all y values have to be 0/1 for ROC

    # Default classifier.
    if clf is None:
        clf = LogisticRegression()

    # Run ROC across time.
    roc_res = pd.DataFrame([ROC(rates[t], target_vec, n_perm, clf)
                            for t in rates],
                           index=rates.columns, columns=['auc', 'pval'])

    return roc_res


def run_unit_ROC_over_time(u, prd, ref_ev, trs_list, nrate, tstep, n_perm,
                           zscore_by, verbose):
    """Run ROC analysis of unit over time. Suitable for parallelization."""

    # Report progress.
    if verbose:
        print(u.Name)

    # Set up params: trials, time period and rates.
    t1s, t2s = u.pr_times(prd, concat=False)
    ref_ts = u.ev_times(ref_ev)

    # Prepare rates.
    rates1, rates2 = [u._Rates[nrate].get_rates(trs, t1s, t2s, ref_ts, tstep)
                      for trs in trs_list]

    # Z-score rates by some trial parameter.
    rates = pd.concat([rates1, rates2])
    if zscore_by is not None:
        ztrs = u.trials_by_param(zscore_by)
        for par, trs in ztrs.items():
            itrs = rates.index.intersection(trs)
            if len(itrs):
                irates = rates.loc[itrs]
                mean = irates.mean()
                std = irates.std(ddof=0)
                rates.loc[itrs, :] = (irates - mean)/std
    rates1, rates2 = [rates.loc[r.index] for r in (rates1, rates2)]

    # Calculate AROC on rates.
    aroc_res = run_ROC_over_time(rates1, rates2, n_perm)

    return aroc_res


def run_group_ROC_over_time(ulist, trs_list, prd, ref_ev, n_perm=None,
                            nrate=None, tstep=None, zscore_by=None,
                            verbose=True):
    """Run ROC over list of units over time."""

    # Run unit-wise AROC test in pool.
    params = [(u, prd, ref_ev, trs_list[i], nrate, tstep,
               n_perm, zscore_by, verbose) for i, u in enumerate(ulist)]
    aroc_res = util.run_in_pool(run_unit_ROC_over_time, params)

    # Separate AUC and p-values.
    aroc_dict = {u.Name: aroc.auc for u, aroc in zip(ulist, aroc_res)}
    pval_dict = {u.Name: aroc.pval for u, aroc in zip(ulist, aroc_res)}

    # Concat into DF.
    aroc_res = pd.concat(aroc_dict, axis=1).T
    pval_res = pd.concat(pval_dict, axis=1).T

    return aroc_res, pval_res


def calc_AROC(ulist, trs_list, prd_pars, n_perm, nrate, tstep, fres,
              verbose=True, rem_all_nan_units=True, rem_any_nan_times=True):
    """Calculate and plot AROC over time between specified sets of trials."""

    stims = prd_pars.index
    aroc_list, pval_list = [], []
    for stim in stims:
        print('    ' + stim)

        # Extract period params.
        pars = ['prd', 'ref_ev', 'feat', 'cond_by', 'zscore_by']
        prd, ref_ev, sfeat, sep_by, zscore_by = prd_pars.loc[stim, pars]

        # Calculate AROC DF.
        aroc, pval = run_group_ROC_over_time(ulist, trs_list[stim], prd,
                                             ref_ev, n_perm, nrate, tstep,
                                             zscore_by, verbose)
        aroc_list.append(aroc)
        pval_list.append(pval)

    # Concatenate stimulus-specific results.
    tshifts = list(prd_pars.stim_start)
    truncate_prds = [list(prd_pars.loc[stim, ['prd_start', 'prd_stop']])
                     for stim in stims]

    aroc = util.concat_stim_prd_res(aroc_list, tshifts, truncate_prds,
                                    rem_all_nan_units, rem_any_nan_times)
    pval = util.concat_stim_prd_res(pval_list, tshifts, truncate_prds,
                                    rem_all_nan_units, rem_any_nan_times)

    # Refill pvals with NaN's if it gets completely emptied above.
    if pval.empty:
        pval = pd.DataFrame(columns=pval.columns, index=aroc.index)

    # Save results.
    if fres is not None:
        aroc_res = {'aroc': aroc, 'pval': pval, 'nrate': nrate,
                    'tstep': tstep, 'n_perm': n_perm}
        util.write_objects(aroc_res, fres)

    return aroc, pval


# %% ROC misc functions.

def run_ROC(ulist, trs_list, nrate, tstep, n_perm, offsets, prd_pars,
            ares_dir):

    # Calculate and save AROC results.
    fres = rocutil.aroc_res_fname(ares_dir, nrate, tstep, n_perm, offsets)
    calc_AROC(ulist, trs_list, prd_pars, n_perm, nrate, tstep, fres)


def plot_ROC(nrate, task, sort_prds, prd_pars, tstep, n_perm, offsets, prefix,
             btw_str, pth, min_len, vth_hi, vth_lo, cmaps,  merge_hi_lo,
             flip_aroc_vals, ares_dir, t1=None, t2=None):

    # Plot AROC matrix sorted by different periods.
    res = rocutil.load_aroc_res(ares_dir, nrate, tstep, n_perm, offsets)
    aroc, pval = res['aroc'], res['pval']

    # Plot results on heatmap.
    rocpost.plot_ROC_heatmap(aroc, pval, task, nrate, tstep, n_perm,
                             sort_prds, prd_pars, offsets, ares_dir,
                             prefix, btw_str, pth, min_len, vth_hi, vth_lo,
                             cmaps, merge_hi_lo, flip_aroc_vals, t1, t2)


def run_plot_feat_ROC(task, feat, feat_params, sort_prds, prd_pars, stims,
                      tstep, nrate, n_perm, offsets, pth, min_len, nlist,
                      ulist, runroc, plotroc, vth_hi, vth_lo, cmaps,
                      merge_hi_lo, flip_aroc_vals, aroc_res_dir,
                      t1=None, t2=None):
    """Run and plot ROC run on given feature."""

    ares_dir = aroc_res_dir + '{}/{}/'.format(feat, nlist)
    prefix, btw_str, t1, t2 = feat_params.loc[feat]

    if runroc:

        # Collect trials to use.
        if feat == 'DS':
            trs_list = {stim: [u.dir_pref_anti_trials(stim, [stim], offsets)
                        for u in ulist] for stim in stims}
        elif feat == 'CE':
            trs_list = {stim: [u.S_D_trials('S2', offsets)
                        for u in ulist] for stim in stims}
        else:
            print('Unknown feature to run ROC on: ', feat)
            return

        # Run ROC.
        run_ROC(ulist, trs_list, nrate, tstep, n_perm, offsets, prd_pars,
                ares_dir)

    if plotroc:

        # Plot ROC results.
        plot_ROC(nrate, task, sort_prds, prd_pars, tstep, n_perm, offsets,
                 prefix, btw_str, pth, min_len, vth_hi, vth_lo, cmaps,
                 merge_hi_lo, flip_aroc_vals, ares_dir, t1, t2)
