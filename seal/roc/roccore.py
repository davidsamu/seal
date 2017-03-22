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
n_jobs = 1  # or util.get_n_cores() - 1


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

        # TODO: is this the best CV type?
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


def run_unit_ROC_over_time(u, prd, ref, trs_list, nrate, n_perm, verbose):
    """Run ROC analysis of unit over time. Suitable for parallelization."""

    # Report progress.
    if verbose:
        print(u.Name)

    # Set up params: trials, time period and rates.
    t1s, t2s = u.pr_times(prd, concat=False)
    ref_ts = u.ev_times(ref)

    # Calculate AROC on rates.
    rates1, rates2 = [u._Rates[nrate].get_rates(trs, t1s, t2s, ref_ts)
                      for trs in trs_list]
    aroc_res = run_ROC_over_time(rates1, rates2, n_perm)

    return aroc_res


def run_group_ROC_over_time(ulist, trs_list, prd, ref, n_perm=None,
                            nrate=None, verbose=True):
    """Run ROC over list of units over time."""

    # Run unit-wise AROC test in pool.
    params = [(u, prd, ref, trs_list[i], nrate, n_perm, verbose)
              for i, u in enumerate(ulist)]
    aroc_res = util.run_in_pool(run_unit_ROC_over_time, params)

    # Separate AUC and p-values.
    aroc_dict = {u.Name: aroc.auc for u, aroc in zip(ulist, aroc_res)}
    pval_dict = {u.Name: aroc.pval for u, aroc in zip(ulist, aroc_res)}

    # Concat into DF.
    aroc_res = pd.concat(aroc_dict, axis=1).T
    pval_res = pd.concat(pval_dict, axis=1).T

    return aroc_res, pval_res


# %% Meta-functions to run and plot AROC over different trial periods.

def calc_AROC(ulist, trs_list, stims, prd_limits, stim_timings,
              n_perm, nrate, fres, verbose=True,
              rem_all_nan_units=True, rem_any_nan_times=True):
    """Calculate and plot AROC over time between specified sets of trials."""

    aroc_list, pval_list = [], []
    for stim in stims:

        # Set up params.
        prd = stim + ' half'
        ref = stim + ' on'

        # Calculate AROC DF.
        aroc, pval = run_group_ROC_over_time(ulist, trs_list[stim], prd, ref,
                                             n_perm, nrate, verbose)
        aroc_list.append(aroc)
        pval_list.append(pval)

    # Concatenate stimulus-specific results.
    offsets = [None, stim_timings.loc['S2', 'on']]
    truncate_prds = [prd_limits.loc[stim, t] for t in ['start', 'stop']]
    aroc = util.concat_stim_prd_res(aroc_list, offsets, truncate_prds,
                                    rem_all_nan_units, rem_any_nan_times)
    pval = util.concat_stim_prd_res(pval_list, offsets, truncate_prds,
                                    rem_all_nan_units, rem_any_nan_times)

    # Save results.
    if fres is not None:
        aroc_res = {'aroc': aroc, 'pval': pval}
        util.write_objects(aroc_res, fres)

    return aroc, pval


def plot_AROC_heatmap(aroc, stims, stim_timings, title, cmap='jet', ffig=None):
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
    pauc.plot_auc_heatmap(aroc, cmap=cmap, events=events,
                          xlbl_freq=500, ylbl_freq=50, xlab=xlab, ylab=ylab,
                          title=title, ffig=ffig)
