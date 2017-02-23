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

from seal.util import util, ua_query

# For reproducable (deterministic) results.
seed = None

verbose = True


# %% Core decoding functions.

def run_logreg(X, y, n_pshfl=0, cv_obj=None, ncv=5, multi_class=None,
               solver=None, Cs=None, class_weight='balanced'):
    """
    Run logistic regression with number of cross-validation folds (ncv) and
    internal regularization over a number of regularisation parameters (Cs).
    """

    # Remove missing values from data.
    idx = np.logical_and(np.all(~np.isnan(X), 1), [yi is not None for yi in y])
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

    if Cs is None:
        Cs = [1]   # no regularisation by default

    # Fit logistic regression.
    sc=[]
    for i in range(10):
        print(i)
        LRCV = LogisticRegressionCV(solver=solver, Cs=Cs, cv=cv_obj,
                                    multi_class=multi_class, refit=True,
                                    class_weight=class_weight)
        LRCV.fit(X, y)

        # Get results for best fit.
        classes = [list(LRCV.scores_.keys())[0]] if binary else LRCV.classes_
        C = LRCV.C_[0]   # should be the same for all classes (as refit=True)

        # Prediction score of each CV fold.
        i_best = np.where(LRCV.Cs_ == C)[0][0]  # index of reg. giving best result
        # Scores should be the same across different classes (multinomial case).
        score = LRCV.scores_[classes[0]][:, i_best].squeeze()

        sc.append(score)
    np.array(sc).mean()

    # Coefficients (weights) of features by predictors.
    coef = LRCV.coef_

    # Run decoding on rate matrix with trials shuffled within units.
    pshfl_score = []
    for i in range(n_pshfl):

        Xpshfld = pop_shfl(X, y)
        # TODO: move this to separate function along with above?
        LRCV.fit(Xpshfld, y)
        # Prediction score of each CV fold.
        classes = [list(LRCV.scores_.keys())[0]] if binary else LRCV.classes_
        i_best = np.where(LRCV.Cs_ == LRCV.C_[0])[0][0]
        pss = LRCV.scores_[classes[0]][:, i_best].squeeze()
        pshfl_score.append(pss.mean())

    return score, coef, C, classes, pshfl_score


def pop_shfl(X, y):
    """Return X predictors shuffled within columns for each y level."""

    ncols = X.shape[1]
    Xc = np.array(X).copy()
    # For trials corresponding to given y value.
    for v, idxs in y.index.groupby(y).items():
        # For each column (predictor) independently.
        for ifeat in range(ncols):
            # Shuffle trials of predictor (feature).
            Xc[idxs, ifeat] = Xc[np.random.permutation(idxs), ifeat]

    return Xc


# %% Wrappers to run decoding over time and different stimulus periods.

def run_logreg_across_time(rates, vfeat, n_pshfl=0, corr_trs=None, ncv=5):
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
        LRparams.append((corr_rates, corr_feat, n_pshfl, None, ncv))
        uids.append(rtmat.columns)

    Scores, Coefs, C, Classes, PshflScore = zip(*util.run_in_pool(run_logreg,
                                                                  LRparams))

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
    # Population shuffled score.
    pd.DataFrame.from_records(PshflScore, index=tvec).T
    PshflScore

    return Scores, Coefs, C, PshflScore


def run_prd_pop_dec(UA, rec, task, uids, trs, sfeat, prd, ref_ev, nrate,
                    n_pshfl, sep_err_trs, ncv):
    """Run logistic regression analysis on population for time period."""

    # Get target vector.
    TrData = ua_query.get_trial_params(UA, rec, task)
    vfeat = TrData.loc[trs].copy()[sfeat].squeeze()

    # Get FR matrix.
    rates = ua_query.get_rate_matrix(UA, rec, task, uids, prd,
                                     ref_ev, nrate, trs)

    # Separate correct trials from error trials, if requested.
    corr_trs = TrData.correct[vfeat.index] if sep_err_trs else None

    # Run decoding.
    Scores, Coefs, C = run_logreg_across_time(rates, vfeat, n_pshfl,
                                              corr_trs, ncv)

    return Scores, Coefs, C, vfeat


def run_pop_dec(UA, rec, task, uids, trs, prd_pars, nrate, n_pshfl,
                sep_err_trs, ncv):
    """Run population decoding on multiple periods across the trials."""

    lScores, lCoefs, lC, lnclasses = [], [], [], []
    stims = prd_pars.index
    for stim in stims:
        print('    ' + stim)

        prd, ref_ev, sfeat = prd_pars.loc[stim, ['prd', 'ref_ev', 'feat']]

        # Run decoding.
        try:
            res = run_prd_pop_dec(UA, rec, task, uids, trs, sfeat, prd, ref_ev,
                                  nrate, n_pshfl, sep_err_trs, ncv)
            Scores, Coefs, C, vfeat = res
            lScores.append(Scores)
            lCoefs.append(Coefs)
            lC.append(C)
            lnclasses.append(len(vfeat))
        except:
            print('Decoding {} - {} - {} failed'.format(rec, task,
                                                        stim))
            continue

    # No successfully decoded stimulus period.
    if not len(lScores):
        print('No stimulus period decoding finished successfully.')
        return

    # Concatenate stimulus-specific results.
    rem_all_nan_units, rem_any_nan_times = True, True
    offsets = list(prd_pars.stim_start)
    truncate_prds = [list(prd_pars.loc[stim, ['prd_start', 'prd_stop']])
                     for stim in stims]

    res = [util.concat_stim_prd_res(r, offsets, truncate_prds,
                                    rem_all_nan_units, rem_any_nan_times)
           for r in (lScores, lCoefs, lC)]
    Scores, Coefs, C = res

    # Prepare results.
    res_dict = {'Scores': Scores, 'Coefs': Coefs, 'C': C,
                'nunits': len(uids), 'ntrials': len(vfeat),
                'nvals': lnclasses, 'prd_pars': prd_pars}

    return res_dict


# %% Utility functions for getting file names, and import / export data.

def res_fname(res_dir, feat, nrate, ncv, n_pshfl, sep_err_trs):
    """Return full path to decoding result with given parameters."""

    feat_str = util.format_to_fname(str(feat))
    ncv_str = 'ncv_{}'.format(ncv)
    pshfl_str = 'np_shfl_{}'.format(n_pshfl)
    err_str = 'w{}_err'.format('o' if sep_err_trs else '')
    fres = '{}{}_{}_{}_{}_{}.data'.format(res_dir, feat_str, nrate,
                                          ncv_str, pshfl_str, err_str)
    return fres


def fig_fname(res_dir, feat, nrate, ncv, n_pshfl, sep_err_trs, ext='png'):
    """Return full path to decoding result with given parameters."""

    fres = res_fname(res_dir, feat, nrate, ncv, n_pshfl, sep_err_trs)
    ffig = os.path.splitext(fres)[0] + '.' + ext

    return ffig


def fig_title(res_dir, feat, nrate, ncv, n_pshfl, sep_err_trs):
    """Return title for decoding result figure with given parameters."""

    feat_str = util.format_to_fname(str(feat))
    cv_str = 'Logistic regression with {}-fold CV'.format(ncv)
    err_str = 'error trials ' + ('excl.' if sep_err_trs else 'incl.')
    pshfl_str = '# population shuffles: {}'.format(n_pshfl)
    title = ('Decoding {}\n'.format(feat_str) + cv_str +
             'FR kernel: {}, {}, {}\n'.format(nrate, err_str, pshfl_str))

    return title


def load_res(res_dir, feat, nrate, ncv, n_pshfl, sep_err_trs):
    """Load decoding results."""

    fres = res_fname(res_dir, feat, nrate, ncv, n_pshfl, sep_err_trs)
    dec_res = util.read_objects(fres, ['Scores', 'Coefs', 'C'])

    return dec_res
