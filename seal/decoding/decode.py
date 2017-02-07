#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to perform decoding analyses and process results.

@author: David Samu
"""

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold

from seal.util import util

# For reproducable (deterministic) results.
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
    Scores = pd.DataFrame.from_records(Scores, index=tvec).T
    # Coefficients (unit by value) over time.
    coef_ser = {t: pd.DataFrame(Coefs[i], columns=uids[i],
                                index=Classes[i]).unstack()
                for i, t in enumerate(tvec)}
    Coefs = pd.concat(coef_ser, axis=1)

    return Scores, Coefs, C


# %% Utility functions for getting file names, and import / export data.

def res_fname(res_dir, feat, nrate, ncv, sep_err_trs):
    """Return full path to decoding result with given parameters."""

    feat_str = util.format_to_fname(str(feat))
    err_str = 'w{}_err'.format('o' if sep_err_trs else '')
    ncv_str = 'ncv_{}'.format(ncv)
    fres = '{}{}_{}_{}_{}.data'.format(res_dir, feat_str, nrate,
                                       ncv_str, err_str)
    return fres


def fig_fname(res_dir, feat, nrate, ncv, sep_err_trs, ext='png'):
    """Return full path to decoding result with given parameters."""

    fres = res_fname(res_dir, feat, nrate, ncv, sep_err_trs)
    ffig = os.path.splitext(fres)[0] + '.' + ext

    return ffig


def fig_title(res_dir, feat, nrate, ncv, sep_err_trs):
    """Return title for decoding result figure with given parameters."""

    feat_str = util.format_to_fname(str(feat))
    err_str = 'error trials ' + ('excl.' if sep_err_trs else 'incl.')
    title = ('Decoding {}\n'.format(feat_str) +
             'FR kernel: {}, {}\n'.format(nrate, err_str) +
             'Logistic regression with {}-fold CV'.format(ncv))

    return title


def load_res(res_dir, feat, nrate, ncv, sep_err_trs):
    """Load decoding results."""

    fres = res_fname(res_dir, feat, nrate, ncv, sep_err_trs)
    dec_res = util.read_objects(fres, ['Scores', 'Coefs', 'C'])

    return dec_res
