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

verbose = True


# %% Core decoding functions.

def run_logreg(X, y, cv_obj=None, ncv=5, multi_class=None, solver=None, Cs=10):
    """
    Run logistic regression with number of cross-validation folds (ncv) and
    internal regularization over a number of regularisation parameters (Cs).
    """

    # Remove missing values from data.
    idx = np.logical_and(np.all(~np.isnan(X), 1), [iy is not None for iy in y])
    X, y = X[idx], y[idx]

    # Init data params.
    classes, vcounts = np.unique(y, return_counts=True)
    ntrials, nfeatures = X.shape
    nclasses = len(classes)

    # Init results.
    score = np.nan * np.zeros(ncv)
    coef = np.nan * np.zeros((nclasses, nfeatures))
    C = np.nan

    # Check that there's at least two classes.
    if nclasses < 2:
        if verbose:
            warnings.warn('Number of different values in y is less then 2!')
        return score, coef, C, classes

    # Check that we have enough trials to split into folds during CV.
    if np.any(vcounts < ncv):
        if verbose:
            warnings.warn('Not enough trials to split into folds during CV')
        return score, coef, C, classes

    # Init LogRegCV parameters.
    binary = (nclasses == 2)
    if multi_class is None:
        multi_class = 'ovr' if binary else 'multinomial'

    if solver is None:
        solver = 'lbfgs' if len(y) < 500 else 'sag'

    if cv_obj is None:
        cv_obj = KFold(n_splits=ncv, shuffle=True, random_state=seed)

    # Fit logistic regression.
    LRCV = LogisticRegressionCV(solver=solver, Cs=Cs, cv=cv_obj,
                                multi_class=multi_class, refit=True)
    LRCV.fit(X, y)

    # Get results for best fit.
    classes = [list(LRCV.scores_.keys())[0]] if binary else LRCV.classes_
    C = LRCV.C_[0]   # should be the same for all class (as refit=True)

    # Prediction score of each CV fold.
    i_best = np.where(LRCV.Cs_ == C)[0][0]  # index of reg. giving best result
    # Scores should be the same across different classes (multinomial case).
    score = LRCV.scores_[classes[0]][:, i_best].squeeze()

    # Coefficients (weights) of features by predictors.
    coef = LRCV.coef_

    return score, coef, C, classes


def run_logreg_across_time(rates, vfeat, corr_trs=None, ncv=5):
    """Run logistic regression analysis across trial time."""

    # Correct and error trials and targets.
    if corr_trs is None:
        corr_trs = pd.Series(True, index=vfeat.index)
    err_trs = ~corr_trs
    corr_feat, err_feat = [vfeat[trs] for trs in [corr_trs, err_trs]]

    # Check that we have enough trials to split into folds during CV.
    vcounts = corr_feat.value_counts()
    if (vcounts < ncv).any():
        if verbose:
            warnings.warn('Not enough trials to do decoding with CV')
        return None, None, None, None, None

    # Run logistic regression at each time point.
    LRparams = []
    uids = []
    for t, rt in rates.items():
        rtmat = rt.unstack().T
        corr_rates, err_rates = [rtmat.loc[trs] for trs in [corr_trs, err_trs]]
        LRparams.append((corr_rates, corr_feat, None, ncv))
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
