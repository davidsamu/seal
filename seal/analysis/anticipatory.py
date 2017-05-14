# -*- coding: utf-8 -*-
"""
Functions related to analyzing stimulus anticipatory activity.

@author: David Samu
"""

from itertools import combinations

import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns

from quantities import ms

from seal.analysis import stats
from seal.util import util
from seal.plot import putil


# %% Core methods to calculate anticipatory activity.

def regr_unit_anticipation(u, nrate, prd, t1_offset, t2_offset, max_len):
    """Calculate anticipatory results for unit."""

    # Init.
    print(u.Name)
    trs = u.inc_trials()
    t1s, t2s = u.pr_times(prd, trs, concat=False)
    t1s = t1s + t1_offset
    t2s = t2s + t2_offset

    rates = u._Rates[nrate].get_rates(trs, t1s, t2s, t1s)
    rates = rates.loc[:, rates.columns <= max_len]

    # Fit linear regression.
    lrates = rates.unstack()
    lrates.dropna(inplace=True)
    x = np.array(lrates.index.get_level_values(0)) / 1000  # ms -> sec
    y = np.array(lrates)
    fit = sp.stats.linregress(x, y)
    fit_res = {fld: getattr(fit, fld) for fld in
               ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']}

    # Test difference from baseline.
    bs_rates = u.get_prd_rates('baseline', trs)
    bs_rates = util.remove_dim_from_series(bs_rates)
    t1s = t2s - 100*ms  # 200 ms interval preceding stimulus
    prestim_rates = u._Spikes.rates(trs, t1s, t2s)
    prestim_rates = util.remove_dim_from_series(prestim_rates)
    fit_res['base_diff_pval'] = stats.wilcoxon_test(bs_rates, prestim_rates)[1]

    return fit_res


# %% Misc and wrapper functions.

def test_prd_anticipation(ulists, nrate, prd, t1_offset, t2_offset, max_len):
    """Tests anticipatory activity before given stimulus."""

    # Get slopes and difference of end-period rate from baseline.
    d_res = dict()
    for gname, ulist in ulists.items():
        params = [(u,  nrate, prd, t1_offset, t2_offset, max_len)
                  for i, u in enumerate(ulist)]
        fit_res = util.run_in_pool(regr_unit_anticipation, params)
        d_res[gname] = pd.DataFrame(fit_res, index=ulist.index)

    # Format results.
    fit_res = pd.concat(d_res)
    fit_res.index = fit_res.index.set_names(['group']+fit_res.index.names[1:])
    fit_res['group'] = fit_res.index.get_level_values('group')
    fit_res['sign_base_diff'] = fit_res.base_diff_pval < 0.01

    return fit_res


def test_anticipation(ulists, nrate):
    """Tests anticipatory activity before both stimuli."""

    # Set up period parameters to analyze. Period lengths are the same: 700 ms.
    # t1_offset is chosen so that rates during delay are roughly at minimum.
    # t2_offset is chosen so as to avoid overlap with stimuli.
    off_preS = int(nrate[1:])/2 * ms
    prd_pars = pd.DataFrame.from_items([('t1_offset', [250*ms, 750*ms]),
                                        ('t2_offset', [-off_preS, -off_preS]),
                                        ('max_len', [700*ms, 700*ms])],
                                       ['fixation', 'delay'], 'index').T

    fit_res = {prd: test_prd_anticipation(ulists, nrate, prd,
                                          t1off, t2off, max_len)
               for prd, (t1off, t2off, max_len) in prd_pars.iterrows()}

    return fit_res


def plot_mean_rates(mRates, aa_res_dir, tasks=None, task_lbls=None,
                    xlim=None, ci=68):
    """Plot mean rates across tasks."""

    # Init.
    if tasks is None:
        tasks = mRates.keys()

    # Plot mean activity.
    lRates = []
    for task in tasks:
        lrates = pd.DataFrame(mRates[task].unstack(), columns=['rate'])
        lrates['task'] = task
        lRates.append(lrates)
    lRates = pd.concat(lRates)
    lRates['time'] = lRates.index.get_level_values(0)
    lRates['unit'] = lRates.index.get_level_values(1)

    if task_lbls is not None:
        lRates.task.replace(task_lbls, inplace=True)

    # Plot as time series.
    putil.set_style('notebook', 'whitegrid')
    fig = putil.figure()
    ax = putil.axes()

    sns.tsplot(lRates, time='time', value='rate', unit='unit',
               condition='task', ci=ci, ax=ax)

    putil.plot_periods(ax=ax)

    putil.set_labels(ax, xlab='time since S1 onset')
    putil.set_limits(ax, xlim=xlim)
    putil.hide_legend_title(ax)

    # Save plot.
    ffig = aa_res_dir + 'aa_curves.png'
    putil.save_fig(ffig, fig)


def plot_slope_diffs_btw_groups(fit_res, prd, res_dir, groups=None,
                                figsize=None):
    """Plot differences in slopes between group pairs."""

    # Test pair-wise difference from each other.
    if groups is None:
        groups = fit_res['group'].unique()
    empty_df = pd.DataFrame(np.nan, index=groups, columns=groups)
    pw_diff_v = empty_df.copy()
    pw_diff_p = empty_df.copy()
    pw_diff_a = empty_df.copy()
    for grp1, grp2 in combinations(groups, 2):
        # Get slopes in each task.
        slp_t1 = fit_res.slope[fit_res.group == grp1]
        slp_t2 = fit_res.slope[fit_res.group == grp2]
        # Mean difference value.
        diff = slp_t1.mean() - slp_t2.mean()
        pw_diff_v.loc[grp1, grp2] = diff
        # Do test for statistical difference.
        stat, pval = stats.mann_whithney_u_test(slp_t1, slp_t2)
        pw_diff_p.loc[grp1, grp2] = pval
        # Annotation DF plot.
        a1, a2 = ['{:.2f}{}'.format(v, util.star_pvalue(pval))
                  for v in (diff, -diff)]
        pw_diff_a.loc[grp1, grp2] = a1
    # Plot and save figure of mean pair-wise difference.
    fig = putil.figure(figsize=figsize)
    sns.heatmap(pw_diff_v, annot=pw_diff_a, fmt='', linewidths=0.5, cbar=False)
    title = 'Mean difference in {} slopes (sp/s / s)'.format(prd)
    putil.set_labels(title=title)
    ffig = res_dir + '{}_anticipatory_slope_pairwise_diff.png'.format(prd)
    putil.save_fig(ffig, fig)

    return pw_diff_v
