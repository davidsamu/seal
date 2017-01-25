# -*- coding: utf-8 -*-
"""
Functions to test statistical difference between samples and across
time-series.

@author: David Samu
"""


import numpy as np
import scipy as sp
import pandas as pd


# Constants.
min_sample_size = 10


# %% Basic statistical tests.

def t_test(x, y, paired=False, equal_var=False, nan_policy='propagate'):
    """
    Run t-test between two related (paired) or independent (unpaired) samples.
    """

    # Remove any NaN values.
    if paired:
        idx = np.logical_and(~np.isnan(x), ~np.isnan(y))
        x, y = x[idx], y[idx]

    # Insufficient sample size.
    xvalid, yvalid = [v[~np.isnan(v)] for v in (x, y)]
    if min(len(xvalid), len(yvalid)) < min_sample_size:
        return np.nan, np.nan

    if paired:
        stat, pval = sp.stats.ttest_rel(x, y, nan_policy=nan_policy)
    else:
        stat, pval = sp.stats.ttest_ind(xvalid, yvalid, equal_var=equal_var)

    return stat, pval


def wilcoxon_test(x, y, zero_method='wilcox', correction=False):
    """
    Run Wilcoxon test, testing the null-hypothesis that
    two related paired samples come from the same distribution. In particular,
    it tests whether the distribution of the differences x-y is symmetric
    about zero. It is a non-parametric version of the paired T-test.

    Note: Because the normal approximation is used for the calculation,
          the samples used should be large. A typical rule is to require
          that n > 20.
    """

    # Remove any NaN values. Test is always paired!
    idx = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x, y = x[idx], y[idx]

    # Insufficient sample size.
    if min(len(x), len(y)) < min_sample_size:
        return np.nan, np.nan

    stat, pval = sp.stats.wilcoxon(x, y, zero_method=zero_method,
                                   correction=correction)
    return stat, pval


def mann_whithney_u_test(x, y, use_continuity=True, alternative='two-sided'):
    """Run Mann-Whitney (aka unpaired Wilcoxon) rank test on samples."""

    # Insufficient sample size.
    xvalid, yvalid = [v[~np.isnan(v)] for v in (x, y)]
    if min(len(xvalid), len(yvalid)) < min_sample_size:
        return np.nan, np.nan

    # At least one item should differ from rest
    xv_un, yv_un = np.unique(xvalid), np.unique(yvalid)
    if len(xv_un) == 1 and len(yv_un) == 1 and np.array_equal(xv_un, yv_un):
        return np.nan, np.nan

    stat, pval = sp.stats.mannwhitneyu(xvalid, yvalid,
                                       use_continuity, alternative)

    return stat, pval


# %% Meta-functions testing statistical differences on time series.

def sign_diff(ts1, ts2, p, test, **kwargs):
    """
    Return times of significant difference between two sets of time series.

    ts1, ts2: time series stored in DataFrames, columns are time samples.
    """

    # Get intersection of time vector.
    tvec = np.intersect1d(ts1.columns, ts2.columns)

    # Select test.
    if test == 't-test':
        test_func = t_test
    elif test == 'wilcoxon':
        test_func = wilcoxon_test
    elif test == 'mann_whitney_u':
        test_func = mann_whithney_u_test
    else:
        print('Unrecognised test name: ' + str(test) + ', running t-test.')
        test_func = t_test

    # Calculate p-values and times of significant difference.
    pvals = pd.Series([test_func(ts1[t], ts2[t], **kwargs)[1]
                       for t in tvec], index=tvec)
    tsign = pvals < p

    return pvals, tsign


def periods(t_on_ser, min_len=None):
    """Return list of periods where t_on is True and with minimum length."""

    if not len(t_on_ser.index):
        return []

    # Init data.
    tvec = np.array(t_on_ser.index)
    t_on = np.array(t_on_ser)

    # Starts of periods.
    tstarts = np.insert(t_on, 0, False)
    istarts = np.logical_and(tstarts[:-1] == False, tstarts[1:] == True)

    # Ends of periods.
    tends = np.insert(t_on, -1, False)
    iends = np.logical_and(tends[:-1] == True, tends[1:] == False)

    # Zip (start, end) pairs of periods.
    pers = [(t1, t2) for t1, t2 in zip(tvec[istarts], tvec[iends])]

    # Drop periods shorter than minimum length.
    if min_len is not None:
        pers = [(t1, t2) for t1, t2 in pers if t2-t1 >= min_len]

    return pers


def sign_periods(ts1, ts2, pval, test, min_len=None, **kwargs):
    """
    Return list of periods of significantly difference
    between two sets of time series (row: samples, columns: time points).
    """

    # Indices of significant difference.
    tsign = sign_diff(ts1, ts2, pval, test, **kwargs)[1]

    # Periods of significant difference.
    sign_periods = periods(tsign, min_len)

    return sign_periods
