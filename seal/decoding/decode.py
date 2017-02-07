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

def run_logreg(X, y, cv_obj=None, ncv=5, multi_class=None, solver=None, Cs=10):
    """
    Run logistic regression with number of cross-validation folds (ncv) and
    internal regularization over a number of regularisation parameters (Cs).
    """

    # Init.
    yvals = np.unique(y)
    if len(yvals) < 2:
        warnings.warn('Number of different values in y is less then 2!')
        return np.nan, np.nan, np.nan

    binary = (len(yvals) == 2)
    if multi_class is None:
        multi_class = 'orv' if binary else 'multinomial'

    if solver is None:
        solver = 'lbfgs' if len(y) < 500 else 'sag'

    if cv_obj is None:
        cv_obj = KFold(n_splits=ncv, shuffle=True, random_state=seed)

    # Fit logistic regression.
    LRCV = LogisticRegressionCV(solver=solver, Cs=Cs, cv=cv_obj,
                                multi_class=multi_class, refit=True)
    LRCV.fit(X, y)

    # Get results for best fit.
    classes = LRCV.classes_
    C = LRCV.C_[0]   # should be the same for all class (as refit=True)

    # Prediction score of each CV fold.
    i_best = np.where(LRCV.Cs_ == C)[0][0]  # index of reg. giving best result
    # Scores should be the same across different classes (multinomial case).
    score = LRCV.scores_[yvals[0]][:, i_best].squeeze()

    # Coefficients (weights) of features by predictors.
    coef = LRCV.coef_

    return score, coef, C, classes


def run_logreg_across_time(rates, vtarget, corr_trs=None, ncv=5):
    """Run logistic regression analysis across trial time."""

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
    uids = []
    for t, rt in rates.items():
        rtmat = rt.unstack().T
        corr_rates, err_rates = [rtmat.loc[trs] for trs in [corr_trs, err_trs]]
        LRparams.append((corr_rates, corr_trg, None, ncv))
        uids.append(rtmat.columns)
    Scores, Coefs, C, Classes = zip(*util.run_in_pool(run_logreg, LRparams))

    # Put results into series and dataframes.
    tvec = rates.columns
    # Best regularisation parameter value
    C = pd.Series(list(C), index=tvec)
    # Prediction scores over time.
    ScoresDF = pd.DataFrame.from_records(Scores, index=tvec).T
    # Coefficients (unit by value) over time.
    coef_ser = {t: pd.DataFrame(Coefs[i], columns=uids[i],
                                index=Classes[i]).unstack()
                for i, t in enumerate(tvec)}
    CoefsDF = pd.concat(coef_ser, axis=1)

    return ScoresDF, CoefsDF, C
