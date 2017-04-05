#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to perform decoding analyses and process results.

@author: David Samu
"""

import warnings

import numpy as np
import pandas as pd

from quantities import deg

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score

from seal.analysis import direction, stats
from seal.decoding import decutil
from seal.util import util, ua_query, constants


# For reproducable (deterministic) results.
seed = 8257  # just a random number

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


def run_logreg(X, y, n_perm=0, n_pshfl=0, cv_obj=None, ncv=5, Cs=None,
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
    res = [('score', np.nan * np.zeros(ncv)), ('class_names', class_names),
           ('coef', np.nan * np.zeros((nclasspars, nfeatures))), ('C', np.nan),
           ('perm', pd.Series(np.nan, index=['mean', 'std', 'pval'])),
           ('psdo', pd.Series(np.nan, index=['mean', 'std', 'pval']))]
    res = util.series_from_tuple_list(res)

    # Check that there's at least two classes.
    if nclasses < 2:
        if verbose:
            warnings.warn('Number of different values in y is less then 2!')
        return res

    # Check that we have enough trials to split into folds during CV.
    if np.any(vcounts < ncv):
        if verbose:
            warnings.warn('Not enough trials to split into folds during CV')
        return res

    # Init LogRegCV parameters.
    if multi_class is None:
        multi_class = 'ovr' if binary else 'multinomial'

    if solver is None:
        solver = 'lbfgs' if len(y) < 500 else 'sag'

    if cv_obj is None:
        cv_obj = StratifiedKFold(n_splits=ncv, shuffle=True,
                                 random_state=seed)

    if Cs is None:
        Cs = [1]   # no regularisation by default

    # Create LogRegress solver.
    LRCV = LogisticRegressionCV(solver=solver, Cs=Cs, cv=cv_obj,
                                multi_class=multi_class, refit=True,
                                class_weight=class_weight)

    # Fit logistic regression.
    class_names, C, score = fit_LRCV(LRCV, X, y)
    res['C'] = C
    res['score'] = score

    # Coefficients (weights) of features by predictors.
    coef = LRCV.coef_
    res['coef'] = coef

    # Run permutation testing.
    if n_perm > 0:
        r = permutation_test_score(LRCV, X, y, scoring='accuracy', cv=cv_obj,
                                   n_permutations=n_perm, random_state=seed)
        _, perm_scores, perm_p = r
        res['perm']['mean'] = perm_scores.mean()
        res['perm']['std'] = perm_scores.std()
        res['perm']['pval'] = perm_p

    # Run decoding on rate matrix with trials shuffled within units.
    if n_pshfl > 0:
        shfld_scores = np.array([fit_LRCV(LRCV, pop_shfl(X, y), y)[2]
                                 for i in range(n_pshfl)]).mean(1)
        res['psdo']['mean'] = shfld_scores.mean()
        res['psdo']['std'] = shfld_scores.std()
        res['psdo']['pval'] = stats.perm_pval(score.mean(), shfld_scores)

    return res


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

def run_logreg_across_time(rates, vfeat, vzscore_by=None, n_perm=0,
                           n_pshfl=0, corr_trs=None, ncv=5, Cs=None):
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

        corr_rates, err_rates = [rtmat.loc[trs]
                                 for trs in [corr_trs, err_trs]]
        LRparams.append((corr_rates, corr_feat, n_perm,
                         n_pshfl, None, ncv, Cs))
        t_uids.append(rtmat.columns)

    # Run logistic regression at each time point.
    res = zip(*util.run_in_pool(run_logreg, LRparams))
    lScores, lClasses, lCoefs, lC, lPerm, lPsdo = res

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
    # Permutation and population shuffling results.
    Perm = pd.concat(lPerm, axis=1, keys=tvec)
    Psdo = pd.concat(lPsdo, axis=1, keys=tvec)

    # Collect results.
    res = [('Scores', Scores), ('Coefs', Coefs), ('C', C),
           ('Perm', Perm), ('Psdo', Psdo)]
    res = util.series_from_tuple_list(res)

    return res


def run_prd_pop_dec(UA, rec, task, stim, uids, trs, feat, zscore_by,
                    even_by, PDD_offset, PPDc, PADc, prd, ref_ev, nrate,
                    n_perm, n_pshfl, sep_err_trs, ncv, Cs, tstep):
    """Run logistic regression analysis on population for time period."""

    # Init.
    TrData = ua_query.get_trial_params(UA, rec, task)

    # Homogenize occurance of values of a feature (same or different as the
    # one to decode), by dropping some trials.
    if not util.is_null(even_by):
        # ntrs: number of trials of least frequent feature value.
        vfeven = TrData[even_by][trs]
        ntrs = vfeven.value_counts().min()
        # Select ntrs number of trials from beginning of recording.
        ltrs = [grp.index[:ntrs] for v, grp in vfeven.groupby(vfeven)]
        trs = pd.Int64Index(pd.Series(np.concatenate(ltrs)).sort_values())

    # Select only trials where direction is PDD/PAD +- offset.
    if not util.is_null(PDD_offset):
        ppds = [d for d in [PPDc, PADc]]
        dd = pd.Series({float(d): np.min([direction.deg_diff(d, pd)
                                          for pd in ppds])
                        for d in constants.all_dirs})
        dirs = dd.index[dd <= PDD_offset]
        pdd_trs = TrData.index[TrData[(stim, 'Dir')].isin(dirs)]
        trs = trs[trs.isin(pdd_trs)]

    # Get target vector.
    vfeat = TrData[feat][trs].squeeze()

    # Binarize direction target vector to decode if PDD/PAD is requested.
    if ('Dir' in feat) and (not util.is_null(PDD_offset)):
        vbin = [np.argmin([direction.deg_diff(d*deg, pd) for pd in ppds])
                for d in vfeat]
        vfeat = pd.Series(vbin, vfeat.index)

    # Init levels of separation and z-scoring condition.
    vzscore_by = (None if zscore_by in (None, np.nan) else
                  TrData.loc[trs].copy()[zscore_by].squeeze())

    # Get FR matrix.
    rates = ua_query.get_rate_matrix(UA, rec, task, uids, prd,
                                     ref_ev, nrate, trs, tstep)

    # Separate correct trials from error trials, if requested.
    corr_trs = TrData.correct[vfeat.index] if sep_err_trs else None

    # Run decoding.
    rates = rates.iloc[:, :2]# REMOVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
    dec_res = run_logreg_across_time(rates, vfeat, vzscore_by, n_perm,
                                     n_pshfl, corr_trs, ncv, Cs)

    # Add # units, trials and classes to results.
    nunits = len(dec_res['Coefs'].index.get_level_values(0).unique())
    ncls = len(dec_res['Coefs'].index.get_level_values(1).unique())
    ntrs = (sum(corr_trs) if corr_trs is not None else
            len(rates.index.get_level_values(1).unique()))
    if ncls == 1:  # binary case
        ncls = 2
    res = {'dec_res': dec_res, 'nunits': nunits, 'ntrs': ntrs, 'ncls': ncls}

    return res


def run_pop_dec(UA, rec, task, uids, trs, prd_pars, nrate, n_perm, n_pshfl,
                sep_err_trs, ncv, Cs, tstep, PPDc, PADc):
    """Run population decoding on multiple periods across given trials."""

    # Init.
    r = {'Scores': [], 'Coefs': [], 'C': [], 'Perm': [], 'Psdo': []}
    nunits, ntrs, ncls = pd.Series(), pd.Series(), pd.Series()

    stims = prd_pars.index
    for stim in stims:
        # print('    ' + stim)

        # Get params.
        pars_names = ['prd', 'ref_ev', 'feat', 'sep_by',
                      'zscore_by', 'even_by', 'PDD_offset']
        pars = prd_pars.loc[stim, pars_names]
        prd, ref_ev, feat, sep_by, zscore_by, even_by, PDD_offset = pars

        # Run decoding.
        res = run_prd_pop_dec(UA, rec, task, stim, uids, trs, feat, zscore_by,
                              even_by, PDD_offset, PPDc, PADc, prd, ref_ev,
                              nrate, n_perm, n_pshfl, sep_err_trs, ncv, Cs,
                              tstep)

        # Extract decording results.
        if res['dec_res'] is not None:
            r = {resname: rres + [res['dec_res'][resname]]
                 for resname, rres in r.items()}

        # Collect parameters.
        nunits[stim] = res['nunits']
        ntrs[stim] = res['ntrs']
        ncls[stim] = res['ncls']

    # No successfully decoded stimulus period.
    if not len(r['Scores']):
        print('No stimulus period decoding finished successfully.')
        return

    # Concatenate stimulus-specific results.
    rem_all_nan_units, rem_any_nan_times = True, True
    tshifts = list(prd_pars.stim_start)
    truncate_prds = [list(prd_pars.loc[stim, ['prd_start', 'prd_stop']])
                     for stim in stims]

    res = {rn: util.concat_stim_prd_res(rr, tshifts, truncate_prds,
                                        rem_all_nan_units, rem_any_nan_times)
           for rn, rr in r.items()}

    # Add # units, trials and classes.
    res['nunits'] = nunits
    res['ntrials'] = ntrs
    res['nclasses'] = ncls

    return res


def dec_recs_tasks(UA, RecInfo, recs, tasks, feat, stims, sep_by, zscore_by,
                   even_by, PDD_offset, res_dir, nrate, tstep, ncv, Cs,
                   n_perm, n_pshfl, sep_err_trs, n_most_DS, PPDres):
    """Run decoding across tasks and recordings."""

    print('\nDecoding: ' + util.format_feat_name(feat))

    # Set up decoding params.
    prd_pars = util.init_stim_prds(stims, feat, sep_by, zscore_by,
                                   even_by, PDD_offset)

    fres = decutil.res_fname(res_dir, 'results', tasks, feat, nrate, ncv, Cs,
                             n_perm, n_pshfl, sep_err_trs, sep_by, zscore_by,
                             even_by, PDD_offset, n_most_DS, tstep)
    rt_res = {}
    for rec in recs:
        print('\n' + rec)
        for task in tasks:

            # Skip recordings that are missing or undecodable.
            if (((rec, task) not in RecInfo.index) or
                not RecInfo.loc[(rec, task), 'nunits']):
                continue

            # Let's not decode saccade and correct/incorrect for passive task.
            if ('Pas' in task) and (feat in ['saccade', 'correct']):
                continue

            # Init.
            print('  ' + task)
            rt_res[(rec, task)] = {}

            # Init units, trials and trial params.
            recinfo = RecInfo.loc[(rec, task)]
            uids = [(rec, ic, iu) for ic, iu in recinfo.units]
            inc_trs = recinfo.trials
            PPDc, PADc = PPDres.loc[(rec, task), ('PPDc', 'PADc')]

            # Select n most DS units (or all if n_most_DS is 0).
            utids = [uid + (task, ) for uid in uids]
            n_most_DS_utids = ua_query.select_n_most_DS_units(UA, utids,
                                                              n_most_DS)
            uids = [utid[:3] for utid in n_most_DS_utids]

            # Split by value condition (optional).
            TrData = ua_query.get_trial_params(UA, rec, task)
            ltrs = (inc_trs.groupby(TrData[sep_by].loc[inc_trs])
                    if not util.is_null(sep_by) else {'all': inc_trs})
            ltrs = pd.Series(ltrs)

            # Decode feature in each period.
            tr_res = {}
            for v, trs in ltrs.items():
                tr_res[v] = run_pop_dec(UA, rec, task, uids, trs, prd_pars,
                                        nrate, n_perm, n_pshfl, sep_err_trs,
                                        ncv, Cs, tstep, PPDc, PADc)
            rt_res[(rec, task)] = tr_res

    # Save results.
    util.write_objects({'rt_res': rt_res}, fres)
