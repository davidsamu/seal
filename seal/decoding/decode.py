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

def fit_LRCV(LRCV, X, y):
    """
    Fit cross-validated logistic regression model on data and return results.
    """

    LRCV.fit(X, y)

    # Get results for best fit.
    classes = list(LRCV.scores_.keys()) if is_binary(y) else LRCV.classes_
    C = LRCV.C_[0]   # should be the same for all classes (as refit=True)

    # Prediction score of each CV fold.
    i_best = np.where(LRCV.Cs_ == C)[0][0]  # index of reg. giving best result
    # Scores should be the same across different classes (multinomial case).
    score = LRCV.scores_[list(LRCV.scores_.keys())[0]][:, i_best].squeeze()

    return classes, C, score


def run_logreg(X, y, n_pshfl=0, cv_obj=None, ncv=5, Cs=None,
               multi_class=None, solver=None, class_weight='balanced'):
    """
    Run logistic regression with number of cross-validation folds (ncv) and
    internal regularization over a number of regularisation parameters (Cs).
    """

    # Remove missing values from data.
    idx = np.logical_and(np.all(~np.isnan(X), 1),
                         [yi is not None for yi in y])
    X, y = np.array(X[idx]), np.array(y[idx])

    # Init data params.
    classes, vcounts = np.unique(y, return_counts=True)
    ntrials, nfeatures = X.shape
    nclasses = len(classes)
    binary = is_binary(y)

    # Deal with binary case.
    class_names = [classes[1]] if binary else classes
    nclasspars = 1 if binary else nclasses

    # Init results.
    score = np.nan * np.zeros(ncv)
    coef = np.nan * np.zeros((nclasspars, nfeatures))
    C = np.nan
    score_shfld = np.nan * np.zeros((n_pshfl, ncv))

    # Check that there's at least two classes.
    if nclasses < 2:
        if verbose:
            warnings.warn('Number of different values in y is less then 2!')
        return score, coef, C, class_names, score_shfld

    # Check that we have enough trials to split into folds during CV.
    if np.any(vcounts < ncv):
        if verbose:
            warnings.warn('Not enough trials to split into folds during CV')
        return score, coef, C, class_names, score_shfld

    # Init LogRegCV parameters.
    if multi_class is None:
        multi_class = 'ovr' if binary else 'multinomial'

    if solver is None:
        solver = 'lbfgs' if len(y) < 500 else 'sag'

    if cv_obj is None:
        cv_obj = KFold(n_splits=ncv, shuffle=True, random_state=seed)

    if Cs is None:
        Cs = [1]   # no regularisation by default

    # Create LogRegress solver.
    LRCV = LogisticRegressionCV(solver=solver, Cs=Cs, cv=cv_obj,
                                multi_class=multi_class, refit=True,
                                class_weight=class_weight)

    # Fit logistic regression.
    class_names, C, score = fit_LRCV(LRCV, X, y)

    # Coefficients (weights) of features by predictors.
    coef = LRCV.coef_

    # Run decoding on rate matrix with trials shuffled within units.
    score_shfld = np.array([fit_LRCV(LRCV, pop_shfl(X, y), y)[2]
                            for i in range(n_pshfl)])

    return score, coef, C, class_names, score_shfld


# %% Utility functions for model fitting.

def is_binary(y):
    """Is it a binary or multinomial classification?"""

    binary = len(np.unique(y)) == 2
    return binary


def pop_shfl(X, y):
    """Return X predictors shuffled within columns for each y level."""

    ncols = X.shape[1]
    Xc = X.copy()

    # For trials corresponding to given y value.
    for v in np.unique(y):
        idxs = np.where(y == v)[0]
        # For each column (predictor) independently.
        for ifeat in range(ncols):
            # Shuffle trials of predictor (feature).
            Xc[idxs, ifeat] = Xc[np.random.permutation(idxs), ifeat]

    return Xc


def zscore_by_cond(X, vzscore_by):
    """Z-score rate values by condition levels within each unit."""

    Xc = X.copy()
    for v, idxs in X.index.groupby(vzscore_by).items():
        vX = Xc.loc[idxs, :]
        mean = vX.mean()
        std = vX.std(ddof=0)
        Xc.loc[idxs, :] = (vX - mean)/std

    return Xc


def separate_by_cond(X, vcond):
    """Separate rate values by condition levels."""

    Xsep = [X.loc[idxs, :] for v, idxs in X.index.groupby(vcond).items()]

    return Xsep


# %% Wrappers to run decoding over time and different stimulus periods.

def run_logreg_across_time(rates, vfeat, vzscore_by=None,
                           n_pshfl=0, corr_trs=None, ncv=5, Cs=10):
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
        return

    # Prepare data for running analysis in pool.
    LRparams = []
    t_uids = []
    for t, rt in rates.items():

        rtmat = rt.unstack().T  # get rates and format to (trial x unit) matrix
        if vzscore_by is not None:  # z-score by condition level
            rtmat = zscore_by_cond(rtmat, vzscore_by)

        corr_rates, err_rates = [rtmat.loc[trs] for trs in [corr_trs, err_trs]]
        LRparams.append((corr_rates, corr_feat, n_pshfl, None, ncv, Cs))
        t_uids.append(rtmat.columns)

    # Run logistic regression at each time point.
    res = zip(*util.run_in_pool(run_logreg, LRparams))
    lScores, lCoefs, lC, lClasses, lShfldScore = res

    # Put results into series and dataframes.
    tvec = rates.columns
    # Best regularisation parameter value.
    C = pd.Series(list(lC), index=tvec)
    # Prediction scores over time.
    Scores = pd.DataFrame.from_records(lScores, index=tvec).T
    # Coefficients (unit by value) over time.
    coef_ser = {t: pd.DataFrame(lCoefs[i], columns=t_uids[i],
                                index=lClasses[i]).unstack()
                for i, t in enumerate(tvec)}
    Coefs = pd.concat(coef_ser, axis=1)
    # Population shuffled score.
    if n_pshfl and ncv:
        ShfldScore = np.array(lShfldScore).reshape(-1, n_pshfl*ncv)
        ShfldScore = pd.DataFrame.from_records(ShfldScore, index=tvec).T
    else:
        ShfldScore = None

    return Scores, Coefs, C, ShfldScore


def run_prd_pop_dec(UA, rec, task, uids, trs, sfeat, zscore_by, prd,
                    ref_ev, nrate, n_pshfl, sep_err_trs, ncv, Cs):
    """Run logistic regression analysis on population for time period."""

    # Get target vector.
    TrData = ua_query.get_trial_params(UA, rec, task)
    vfeat = TrData.loc[trs].copy()[sfeat].squeeze()

    # Init levels of separation and z-scoring condition.
    vzscore_by = (None if zscore_by in (None, np.nan) else
                  TrData.loc[trs].copy()[zscore_by].squeeze())

    # Get FR matrix.
    rates = ua_query.get_rate_matrix(UA, rec, task, uids, prd,
                                     ref_ev, nrate, trs)

    # Separate correct trials from error trials, if requested.
    corr_trs = TrData.correct[vfeat.index] if sep_err_trs else None

    # Run decoding.
    res = run_logreg_across_time(rates, vfeat, vzscore_by,
                                 n_pshfl, corr_trs, ncv, Cs)

    return res


def run_pop_dec(UA, rec, task, uids, trs, prd_pars, nrate, n_pshfl,
                sep_err_trs, ncv, Cs):
    """Run population decoding on multiple periods across given trials."""

    lScores, lCoefs, lC, lShfldScores = [], [], [], []
    stims = prd_pars.index
    for stim in stims:
        print('    ' + stim)

        pars = ['prd', 'ref_ev', 'feat', 'cond_by', 'zscore_by']
        prd, ref_ev, sfeat, sep_by, zscore_by = prd_pars.loc[stim, pars]

        # Run decoding.
#        try:
        res = run_prd_pop_dec(UA, rec, task, uids, trs, sfeat, zscore_by, prd,
                              ref_ev, nrate, n_pshfl, sep_err_trs, ncv, Cs)
        Scores, Coefs, C, ShfldScores = res
        lScores.append(Scores)
        lCoefs.append(Coefs)
        lC.append(C)
        lShfldScores.append(ShfldScores)
#        except:
#            print('Decoding {} - {} - {} failed'.format(rec, task, stim))
#            continue

    # No successfully decoded stimulus period.
    if not len(lScores):
        print('No stimulus period decoding finished successfully.')
        return

    # Concatenate stimulus-specific results.
    rem_all_nan_units, rem_any_nan_times = True, True
    tshifts = list(prd_pars.stim_start)
    truncate_prds = [list(prd_pars.loc[stim, ['prd_start', 'prd_stop']])
                     for stim in stims]

    res = [util.concat_stim_prd_res(r, tshifts, truncate_prds,
                                    rem_all_nan_units, rem_any_nan_times)
           for r in (lScores, lCoefs, lC, lShfldScores)]
    Scores, Coefs, C, ShfldScores = res

    # Prepare results.
    res_dict = {'Scores': Scores, 'Coefs': Coefs, 'C': C,
                'ShfldScores': ShfldScores, 'nunits': len(uids),
                'ntrials': len(trs), 'prd_pars': prd_pars}

    return res_dict


# %% Utility functions for getting file names, and import / export data.

def res_fname(res_dir, feat, nrate, ncv, n_pshfl, sep_err_trs, cond,
              n_most_DS):
    """Return full path to decoding result with given parameters."""

    feat_str = util.format_to_fname(str(feat))
    ncv_str = 'ncv_{}'.format(ncv)
    pshfl_str = 'npshfl_{}'.format(n_pshfl)
    err_str = 'w{}_err'.format('o' if sep_err_trs else '')
    cond_str = ('by_'+util.format_feat_name(cond, True)
                if cond is not None else 'uncond')
    nDS_str = ('top_{}_units'.format(n_most_DS)
               if n_most_DS != 0 else 'all_units')
    fres = '{}{}/{}_{}_{}_{}_{}_{}.data'.format(res_dir, feat_str, nrate,
                                                ncv_str, pshfl_str, err_str,
                                                cond_str, nDS_str)
    return fres


def fig_fname(res_dir, feat, nrate, ncv, n_pshfl, sep_err_trs, cond, n_most_DS,
              ext='png'):
    """Return full path to decoding result with given parameters."""

    fres = res_fname(res_dir, feat, nrate, ncv, n_pshfl,
                     sep_err_trs, cond, n_most_DS)
    ffig = os.path.splitext(fres)[0] + '.' + ext
    return ffig


def fig_title(res_dir, feat, nrate, ncv, n_pshfl, sep_err_trs, cond,
              n_most_DS):
    """Return title for decoding result figure with given parameters."""

    feat_str = util.format_to_fname(str(feat))
    cv_str = 'Logistic regression with {}-fold CV'.format(ncv)
    err_str = 'error trials ' + ('excl.' if sep_err_trs else 'incl.')
    pshfl_str = '# population shuffles: {}'.format(n_pshfl)
    cond_str = ('' if cond is None else
                'conditioned by ' + util.format_feat_name(cond, True))
    nDS_str = ('using {} most DS units'.format(n_most_DS)
               if n_most_DS != 0 else 'all units')
    title = ('Decoding {}\n'.format(feat_str) + cv_str +
             '\nFR kernel: {}, {}'.format(nrate, err_str) +
             '\n{}, {}, {}'.format(pshfl_str, nDS_str, cond_str))
    return title


def load_res(res_dir, feat, nrate, ncv, n_pshfl, sep_err_trs, cond, nDS_str):
    """Load decoding results."""

    fres = res_fname(res_dir, feat, nrate, ncv, n_pshfl,
                     sep_err_trs, cond, nDS_str)
    dec_res = util.read_objects(fres, ['Scores', 'Coefs', 'C'])

    return dec_res
