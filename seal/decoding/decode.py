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

from seal.util import util


# %% Core decoding functions.

def run_logreg(X, y, Cs=10, ncv=5):
    """
    Run logistic regression with number of cross-validation (nCV) and
    internal regularization over a number of regularisation parameters (Cs).
    """

    # Fit logistic regression.
    LRCV = LogisticRegressionCV(Cs=Cs, cv=ncv)
    LRCV.fit(X, y)

    # Extract results for best fit.
    C = LRCV.C_[0]   # regularization parameter of best result
    i_best = np.where(LRCV.Cs_ == C)[0]  # index of best results
    acc = LRCV.scores_[1][:, i_best][:, 0]  # accuracy of each CV fold
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
    ntrg1, ntrg2 = corr_trg.value_counts()[::-1]
    if ntrg1 < ncv or ntrg2 < ncv:
        warnings.warn('Not enough trials to do decoding, ' +
                      '{} trg1, {} trg2, {} CVs'.format(ntrg1, ntrg2, ncv))
        return None, None, None, None, None

    # Run logistic regression at each time point.
    LRparams = []
    for t, rt in rates.items():
        rtmat = rt.unstack().T
        corr_rates, err_rates = [rtmat.loc[trs] for trs in [corr_trs, err_trs]]
        LRparams.append((corr_rates, corr_trg, Cs, ncv))
    Acc, Weights, C = zip(*util.run_in_pool(run_logreg, LRparams))

    # Put results into series and dataframes.
    tvec = rates.columns
    C = pd.Series(list(C), index=tvec)
    Acc = pd.DataFrame(list(Acc), index=tvec)
    Weights = pd.DataFrame(list(Weights), index=tvec, columns=uids)

    return Acc, Weights, C, ntrg1, ntrg2
