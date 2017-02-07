#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to perform decoding analyses and process results.

@author: David Samu
"""

import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold

from seal.util import util

seed = None


# %% Core decoding functions.

def run_logreg(X, y, Cs=10, cv_obj=None, ncv=5, multi_class=None):
    """
    Run logistic regression with number of cross-validation folds (ncv) and
    internal regularization over a number of regularisation parameters (Cs).
    """

    # Init.
    yvals = y.unique()
    if len(yvals) < 2:
        warnings.warn('Number of different values in y is less then 2!')
        return np.nan, np.nan, np.nan

    if multi_class is None:
        multi_class = 'orv' if len(yvals) == 2 else 'multinomial'

    if cv_obj is None:
        cv_obj = KFold(n_splits=ncv, shuffle=True, random_state=seed)

    # Fit logistic regression.
    LRCV = LogisticRegressionCV(Cs=Cs, cv=cv_obj, multi_class=multi_class)
    LRCV.fit(X, y)

    # Extract results for best fit.
    C = LRCV.C_[0]   # regularization parameter of best result
    i_best = np.where(LRCV.Cs_ == C)[0]  # index of best results
    acc = LRCV.scores_[yvals[0]][:, i_best][:, 0]  # accuracy of each CV fold
    weights = LRCV.coef_[0]

    return acc, weights, C


def run_logreg_across_time(rates, vtarget, corr_trs=None, Cs=10, ncv=5):
    """Run logistic regression analysis across trial time."""

    # Init.
    uids = rates.index.get_level_values(0).unique()

    # Correct and error trials and targets.
    if corr_trs is None:
        corr_trs = pd.Series(True, index=vtarget.index)
    err_trs = ~corr_trs
    corr_trg, err_trg = [vtarget[trs] for trs in [corr_trs, err_trs]]

    # Check that we have enough trials to split into folds during CV.
    vcounts = corr_trg.value_counts()
    if (vcounts < ncv).any():
        warnings.warn('Not enough trials to do decoding with CV')
        return None, None, None, None, None

    # Run logistic regression at each time point.
    LRparams = []
    for t, rt in rates.items():
        rtmat = rt.unstack().T
        corr_rates, err_rates = [rtmat.loc[trs] for trs in [corr_trs, err_trs]]
        LRparams.append((corr_rates, corr_trg, Cs, None, ncv))
    Acc, Weights, C = zip(*util.run_in_pool(run_logreg, LRparams))

    # Put results into series and dataframes.
    tvec = rates.columns
    C = pd.Series(list(C), index=tvec)
    Acc = pd.DataFrame(list(Acc), index=tvec)
    Weights = pd.DataFrame(list(Weights), index=tvec, columns=uids)

    return Acc, Weights, C
