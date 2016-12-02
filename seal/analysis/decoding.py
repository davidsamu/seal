#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 13:06:55 2016

Functions for performing and processing decoding analyses.

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
    C = LRCV.C_[0]
    idx_bestC = np.where(LRCV.Cs_ == C)[0]
    perf = LRCV.scores_[1][:, idx_bestC][:, 0]
    weights = LRCV.coef_[0]

    return perf, weights, C


# TODO: extend with error trial analysis.
def run_logreg_across_time(FRdf, target_vec, corr_trs=None, Cs=10, ncv=5):
    """Run logistic regression analysis across trial time."""

    # Init.
    uidxs = FRdf.index.get_level_values(0).unique()

    # Correct and error trials and targets.
    if corr_trs is None:
        corr_trs = pd.Series(True, index=target_vec.index)
    err_trs = ~corr_trs
    corrTrg, errTrg = [target_vec[trs] for trs in [corr_trs, err_trs]]

    # Check that we have enough trials to split into folds during CV.
    ntrg1, ntrg2 = corrTrg.value_counts()[::-1]
    if ntrg1 < ncv or ntrg2 < ncv:
        warnings.warn('Not enough trials to decode' +
                      ', ntrg1: {}, ntrg2: {}, ncv: {}'.format(ntrg1, ntrg2, ncv))
        return None, None

    # Run logistic regression for each time point.
    LRparams = []
    for t, FR in FRdf.items():
        FRmat = pd.concat({uidx: FR.loc[uidx] for uidx in uidxs}, axis=1)
        corrFR, errFR = [FRmat.loc[trs] for trs in [corr_trs, err_trs]]
        LRparams.append((corrFR, corrTrg, Cs, ncv))
    Perf, Weights, C = zip(*util.run_in_pool(run_logreg, LRparams))

    # Put results into series and dataframes.
    tvec = FRdf.columns
    C = pd.Series(list(C), index=tvec)
    Perf = pd.DataFrame(list(Perf), index=tvec)
    Weights = pd.DataFrame(list(Weights), index=tvec, columns=uidxs)

    return Perf, Weights, C, ntrg1, ntrg2
