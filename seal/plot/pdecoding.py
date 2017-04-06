# -*- coding: utf-8 -*-

"""
Functions to plot decoding results.

@author: David Samu
"""

import numpy as np
import pandas as pd
import seaborn as sns

from seal.analysis import stats
from seal.decoding import decutil
from seal.plot import putil, pplot
from seal.util import constants, util


tlab = 'Time since S1 onset (ms)'
ylab_scr = 'decoding accuracy'
tlim = [-1000, 3500]
ylim_scr = [0, 1]
fformat = 'png'

verbose = False


def plot_score_set(Scores, ax=None, time='time', value='score', unit='fold',
                   color='b'):
    """Plot decoding scores over time as time series."""

    # Prepare data
    lScores = pd.DataFrame(Scores.unstack(), columns=[value])
    lScores[time] = lScores.index.get_level_values(0)
    lScores[unit] = lScores.index.get_level_values(1)
    lScores.index = np.arange(len(lScores.index))

    # Plot as time series.
    sns.tsplot(lScores, time=time, value=value,
               unit=unit, color=color, ax=ax)

    return ax


def plot_mean_std_sdiff(x, ymean, ystd, pval, pth=0.01, color='b', lw=4,
                        ax=None):
    """Plot mean +- std and significant difference for permuted results."""

    # Plot mean +- std.
    mean_kws = {'color': color, 'alpha': 0.5}
    band_kws = {'color': color, 'alpha': 0.2}
    pplot.mean_err(x, ymean, ystd, mean_kws=mean_kws,
                   band_kws=band_kws, ax=ax)

    # Add bars for significance periods.
    # Two tailed test: p >= 1-pth also counts as significant!
    tsign = pd.concat([pval <= pth, pval >= 1-pth], axis=1).any(axis=1)
    sign_prds = stats.periods(tsign)
    putil.plot_signif_prds(sign_prds, color=color,
                           linewidth=lw, ax=ax)

    return ax


def plot_scores(ax, Scores, Perm=None, Psdo=None, nvals=None, prds=None,
                col='b', perm_col='grey', psdo_col='g', xlim=None,
                ylim=ylim_scr, xlab=tlab, ylab=ylab_scr, title='',
                ytitle=1.04):
    """Plot decoding accuracy results."""

    lgn_patches = []

    # Plot permuted results (if exist).
    if not util.is_null(Perm) and not Perm.isnull().all().all():
        x, pval = Perm.columns, Perm.loc['pval']
        ymean, ystd = Perm.loc['mean'], Perm.loc['std']
        plot_mean_std_sdiff(x, ymean, ystd, pval, pth=0.1, lw=6,
                            color=perm_col, ax=ax)
        lgn_patches.append(putil.get_artist('permuted', perm_col))

    # Plot population shuffled results (if exist).
    if not util.is_null(Psdo) and not Psdo.isnull().all().all():
        x, pval = Psdo.columns, Psdo.loc['pval']
        ymean, ystd = Psdo.loc['mean'], Psdo.loc['std']
        plot_mean_std_sdiff(x, ymean, ystd, pval, pth=0.01, lw=3,
                            color=psdo_col, ax=ax)
        lgn_patches.append(putil.get_artist('pseudo-population', psdo_col))

    # Plot scores.
    plot_score_set(Scores, ax, color=col)
    lgn_patches.append(putil.get_artist('synchronous', col))

    # Add legend.
    lgn_patches = lgn_patches[::-1]
    putil.set_legend(ax, handles=lgn_patches)

    # Add chance level line.
    # This currently plots all nvals combined across stimulus period!
    if nvals is not None:
        chance_lvl = 1.0 / nvals
        putil.add_chance_level(ax=ax, ylevel=chance_lvl)

    # Add stimulus periods.
    if prds is not None:
        putil.plot_periods(prds, ax=ax)

    # Set axis limits.
    putil.set_limits(ax, xlim, ylim)

    # Format plot.
    putil.set_labels(ax, xlab, ylab, title, ytitle)


def plot_weights(ax, Coefs, prds=None, xlim=None, xlab=tlab,
                 ylab='unit coefficient', title='', ytitle=1.04):
    """Plot decoding weights."""

    # Unstack dataframe with results.
    lCoefs = pd.DataFrame(Coefs.unstack().unstack(), columns=['coef'])
    lCoefs['time'] = lCoefs.index.get_level_values(0)
    lCoefs['value'] = lCoefs.index.get_level_values(1)
    lCoefs['uid'] = lCoefs.index.get_level_values(2)
    lCoefs.index = np.arange(len(lCoefs.index))

    # Plot time series.
    sns.tsplot(lCoefs, time='time', value='coef', unit='value',
               condition='uid', ax=ax)

    # Add chance level line and stimulus periods.
    putil.add_chance_level(ax=ax, ylevel=0)
    putil.plot_periods(prds, ax=ax)

    # Set axis limits.
    putil.set_limits(ax, xlim)

    # Format plot.
    putil.set_labels(ax, xlab, ylab, title, ytitle)
    putil.hide_legend(ax)


def plot_scores_weights(recs, stims, res_dir, par_kws):
    """
    Plot prediction scores and model weights for given recording and analysis.
    """

    # Init.
    putil.set_style('notebook', 'ticks')
    n_most_DS = par_kws['n_most_DS']
    tasks = par_kws['tasks']

    # Remove Passive if plotting Saccade or Correct.
    if par_kws['feat'] in ['saccade', 'correct']:
        tasks = tasks[~tasks.str.contains('Pas')]

    # Load results.
    rt_res = decutil.load_res(res_dir, **par_kws)[n_most_DS]

    # Create figures.
    # For prediction scores.
    fig_scr, _, axs_scr = putil.get_gs_subplots(nrow=len(recs),
                                                ncol=len(tasks),
                                                subw=8, subh=6,
                                                create_axes=True)

    # For unit weights (coefficients).
    fig_wgt, _, axs_wgt = putil.get_gs_subplots(nrow=len(recs),
                                                ncol=len(tasks),
                                                subw=8, subh=6,
                                                create_axes=True)

    for irec, rec in enumerate(recs):
        if verbose:
            print('\n' + rec)
        for itask, task in enumerate(tasks):
            if verbose:
                print('    ' + task)

            # Init figures.
            ax_scr = axs_scr[irec, itask]
            ax_wgt = axs_wgt[irec, itask]

            # Check if any result exists for rec-task combination.
            if (((rec, task) not in rt_res.keys()) or
               (not len(rt_res[(rec, task)].keys()))):
                ax_scr.axis('off')
                ax_wgt.axis('off')
                continue

            # Init data.
            res = rt_res[(rec, task)]
            vals = [v for v in res.keys() if not util.is_null(res[v])]
            cols = sns.color_palette('hls', len(vals))
            lnunits, lntrs, lncls,  = [], [], []
            for v, col in zip(vals, cols):
                # Basic results.
                vres = res[v]
                Scores = vres['Scores']
                Coefs = vres['Coefs']
                Perm = vres['Perm']
                Psdo = vres['Psdo']
                # Decoding params.
                lnunits.append(vres['nunits'])
                lntrs.append(vres['ntrials'])
                lncls.append(vres['nclasses'])
                # Plot decoding accuracy.
                plot_scores(ax_scr, Scores, Perm, Psdo, col=col)

            # Add labels.
            uni_lnunits = np.unique(np.array(lnunits).flatten())
            if len(uni_lnunits) > 1 and verbose:
                print('Different number of units found.')
            nunits = uni_lnunits[0]
            title = '{} {}, {} units'.format(rec, task, nunits)
            putil.set_labels(ax_scr, tlab, ylab_scr, title, ytitle=1.04)

            # Add chance level line.
            uni_ncls = np.unique(np.array(lncls).flatten())
            if len(uni_ncls) > 1 and verbose:
                print('Different number of classes found.')
            for nvals in uni_ncls:
                chance_lvl = 1.0 / nvals
                putil.add_chance_level(ax=ax_scr, ylevel=chance_lvl)

            # Plot stimulus periods.
            prds = [[stim] + list(constants.fixed_tr_prds.loc[stim])
                    for stim in stims]
            putil.plot_periods(prds, ax=ax_scr)

            # Plot unit weights over time.
            plot_weights(ax_wgt, Coefs, prds, tlim, tlab, title=title)

    # Match axes across decoding plots.
    # [putil.sync_axes(axs_scr[:, itask], sync_y=True)
    #  for itask in range(axs_scr.shape[1])]

    # Save plots.
    title = decutil.fig_title(res_dir, **par_kws)
    fs_title = 'large'
    w_pad, h_pad = 3, 3

    # Performance.
    ffig = decutil.fig_fname(res_dir, 'score', 'pdf', **par_kws)
    putil.save_fig(ffig, fig_scr, title, fs_title, w_pad=w_pad, h_pad=h_pad)

    # Weights.
    ffig = decutil.fig_fname(res_dir, 'weight', 'pdf', **par_kws)
    putil.save_fig(ffig, fig_wgt, title, fs_title, w_pad=w_pad, h_pad=h_pad)


def plot_score_multi_rec(recs, stims, res_dir, par_kws):
    """Plot prediction scores for multiple recordings."""

    # Init.
    putil.set_style('notebook', 'ticks')
    n_most_DS = par_kws['n_most_DS']
    tasks = par_kws['tasks']

    # Remove Passive if plotting Saccade or Correct.
    if par_kws['feat'] in ['saccade', 'correct']:
        tasks = tasks[~tasks.str.contains('Pas')]

    # Load results.
    rt_res = decutil.load_res(res_dir, **par_kws)[n_most_DS]

    # Create figure.
    ret = putil.get_gs_subplots(nrow=1, ncol=len(tasks),
                                subw=8, subh=6, create_axes=True)
    fig_scr, _, axs_scr = ret

    print('\nPlotting multi-recording results...')
    for itask, task in enumerate(tasks):
        if verbose:
            print('    ' + task)
        ax_scr = axs_scr[0, itask]

        dict_lScores = {}
        for irec, rec in enumerate(recs):

            # Check if results exist for rec-task combination.
            if (((rec, task) not in rt_res.keys()) or
               (not len(rt_res[(rec, task)].keys()))):
                continue

            # Init data.
            res = rt_res[(rec, task)]
            cols = sns.color_palette('hls', len(res.keys()))
            lncls = []
            for v, col in zip(res.keys(), cols):
                vres = res[v]
                if vres is None:
                    continue

                Scores = vres['Scores']
                lncls.append(vres['nclasses'])

                # Unstack dataframe with results.
                lScores = pd.DataFrame(Scores.unstack(), columns=['score'])
                lScores['time'] = lScores.index.get_level_values(0)
                lScores['fold'] = lScores.index.get_level_values(1)
                lScores.index = np.arange(len(lScores.index))

                dict_lScores[(rec, v)] = lScores

        if not len(dict_lScores):
            ax_scr.axis('off')
            continue

        # Concatenate accuracy scores from every recording.
        all_lScores = pd.concat(dict_lScores)
        all_lScores['rec'] = all_lScores.index.get_level_values(0)
        all_lScores.index = np.arange(len(all_lScores.index))

        # Plot decoding results.
        nrec = len(all_lScores['rec'].unique())
        title = '{}, {} recordings'.format(task, nrec)
        ytitle = 1.0
        prds = [[stim] + list(constants.fixed_tr_prds.loc[stim])
                for stim in stims]

        # Plot time series.
        palette = sns.color_palette('muted')
        sns.tsplot(all_lScores, time='time', value='score', unit='fold',
                   condition='rec', color=palette, ax=ax_scr)

        # Add chance level line.
        # This currently plots a chance level line for every nvals,
        # combined across stimulus period!
        uni_ncls = np.unique(np.array(lncls).flatten())
        if len(uni_ncls) > 1 and verbose:
            print('Different number of classes found.')
        for nvals in uni_ncls:
            chance_lvl = 1.0 / nvals
            putil.add_chance_level(ax=ax_scr, ylevel=chance_lvl)

        # Add stimulus periods.
        if prds is not None:
            putil.plot_periods(prds, ax=ax_scr)

        # Set axis limits.
        putil.set_limits(ax_scr, tlim, ylim_scr)

        # Format plot.
        putil.set_labels(ax_scr, tlab, ylab_scr, title, ytitle)

    # Save figure.
    title = decutil.fig_title(res_dir, **par_kws)
    fs_title = 'large'
    w_pad, h_pad = 3, 3
    ffig = decutil.fig_fname(res_dir, 'all_scores', fformat, **par_kws)
    putil.save_fig(ffig, fig_scr, title, fs_title, w_pad=w_pad, h_pad=h_pad)


def plot_scores_across_nunits(recs, stims, res_dir, list_n_most_DS, par_kws):
    """
    Plot prediction score results across different number of units included.
    """

    # Init.
    putil.set_style('notebook', 'ticks')
    tasks = par_kws['tasks']

    # Remove Passive if plotting Saccade or Correct.
    if par_kws['feat'] in ['saccade', 'correct']:
        tasks = tasks[~tasks.str.contains('Pas')]

    # Load all results to plot.
    dict_rt_res = decutil.load_res(res_dir, list_n_most_DS, **par_kws)

    # Create figures.
    fig_scr, _, axs_scr = putil.get_gs_subplots(nrow=len(recs),
                                                ncol=len(tasks),
                                                subw=8, subh=6,
                                                create_axes=True)
    # Do plotting per recording and task.
    for irec, rec in enumerate(recs):
        if verbose:
            print('\n' + rec)
        for itask, task in enumerate(tasks):
            if verbose:
                print('    ' + task)

            ax_scr = axs_scr[irec, itask]

            # Init data.
            dict_lScores = {}
            cols = sns.color_palette('hls', len(dict_rt_res.keys()))
            lncls = []
            for (n_most_DS, rt_res), col in zip(dict_rt_res.items(), cols):

                # Check if results exist for rec-task combination.
                if (((rec, task) not in rt_res.keys()) or
                    (not len(rt_res[(rec, task)].keys()))):
                    continue

                res = rt_res[(rec, task)]
                for v, col in zip(res.keys(), cols):
                    vres = res[v]
                    Scores = vres['Scores']
                    lncls.append(vres['nclasses'])

                    # Unstack dataframe with results.
                    lScores = pd.DataFrame(Scores.unstack(), columns=['score'])
                    lScores['time'] = lScores.index.get_level_values(0)
                    lScores['fold'] = lScores.index.get_level_values(1)
                    lScores.index = np.arange(len(lScores.index))

                    # Get number of units tested.
                    nunits = vres['nunits']
                    uni_nunits = nunits.unique()
                    if len(uni_nunits) > 1 and verbose:
                        print('Different number of units found.')
                    nunits = uni_nunits[0]

                    # Collect results.
                    dict_lScores[(nunits, v)] = lScores

            # Skip rest if no data is available.
            # Check if any result exists for rec-task combination.
            if not len(dict_lScores):
                ax_scr.axis('off')
                continue

            # Concatenate accuracy scores from every recording.
            all_lScores = pd.concat(dict_lScores)
            all_lScores['n_most_DS'] = all_lScores.index.get_level_values(0)
            all_lScores.index = np.arange(len(all_lScores.index))

            # Plot decoding results.
            nnunits = len(all_lScores['n_most_DS'].unique())
            title = '{} {}, {} sets of units'.format(rec, task, nnunits)
            ytitle = 1.0
            prds = [[stim] + list(constants.fixed_tr_prds.loc[stim])
                    for stim in stims]

            # Plot time series.
            palette = sns.color_palette('muted')
            sns.tsplot(all_lScores, time='time', value='score', unit='fold',
                       condition='n_most_DS', color=palette, ax=ax_scr)

            # Add chance level line.
            # This currently plots a chance level line for every nvals,
            # combined across stimulus period!
            uni_ncls = np.unique(np.array(lncls).flatten())
            if len(uni_ncls) > 1 and verbose:
                print('Different number of classes found.')
            for nvals in uni_ncls:
                chance_lvl = 1.0 / nvals
                putil.add_chance_level(ax=ax_scr, ylevel=chance_lvl)

            # Add stimulus periods.
            if prds is not None:
                putil.plot_periods(prds, ax=ax_scr)

            # Set axis limits.
            putil.set_limits(ax_scr, tlim, ylim_scr)

            # Format plot.
            putil.set_labels(ax_scr, tlab, ylab_scr, title, ytitle)

    # Match axes across decoding plots.
    # [putil.sync_axes(axs_scr[:, itask], sync_y=True)
    #  for itask in range(axs_scr.shape[1])]

    # Save plots.
    list_n_most_DS_str = [str(i) if i != 0 else 'all' for i in list_n_most_DS]
    par_kws['n_most_DS'] = ', '.join(list_n_most_DS_str)
    title = decutil.fig_title(res_dir, **par_kws)
    fs_title = 'large'
    w_pad, h_pad = 3, 3

    par_kws['n_most_DS'] = '_'.join(list_n_most_DS_str)
    ffig = decutil.fig_fname(res_dir, 'score_nunits', fformat, **par_kws)
    putil.save_fig(ffig, fig_scr, title, fs_title, w_pad=w_pad, h_pad=h_pad)


def plot_combined_rec_mean(recs, stims, res_dir, par_kws,
                           list_n_most_DS, list_min_nunits,
                           n_boot=1e4, ci=95):
    """Test and plot results combined across sessions."""

    # Init.
    putil.set_style('notebook', 'ticks')
    vkey = 'all'

    # This should be made more explicit!
    prds = [[stim] + list(constants.fixed_tr_prds.loc[stim])
            for stim in stims]

    # Load all results to plot.
    dict_rt_res = decutil.load_res(res_dir, list_n_most_DS, **par_kws)

    # Create figures.
    fig_scr, _, axs_scr = putil.get_gs_subplots(nrow=len(dict_rt_res),
                                                ncol=len(list_min_nunits),
                                                subw=8, subh=6,
                                                create_axes=True)

    # Query data.
    allScores = {}
    allnunits = {}
    for n_most_DS, rt_res in dict_rt_res.items():
        # Get accuracy scores.
        dScores = {(rec, task): res[vkey]['Scores'].mean()
                   for (rec, task), res in rt_res.items()
                   if res[vkey] is not None}
        allScores[n_most_DS] = pd.concat(dScores, axis=1).T
        # Get number of units.
        allnunits[n_most_DS] = {(rec, task): res[vkey]['nunits'].iloc[0]
                                for (rec, task), res in rt_res.items()
                                if res[vkey] is not None}
        # Get # values (for baseline plotting.)
        all_nvals = pd.Series({(rec, task): res[vkey]['nclasses'].iloc[0]
                               for (rec, task), res in rt_res.items()
                               if res[vkey] is not None})
        un_nvals = all_nvals.unique()
        if len(un_nvals) > 1 and verbose:
            print('Found multiple # of classes to decode: {}'.format(un_nvals))
        nvals = un_nvals[0]

    allnunits = pd.DataFrame(allnunits)

    # Plot mean performance across recordings and
    # test significance by bootstrapping.
    for inmost, n_most_DS in enumerate(list_n_most_DS):
        Scores = allScores[n_most_DS]
        nunits = allnunits[n_most_DS]

        for iminu, min_nunits in enumerate(list_min_nunits):

            ax_scr = axs_scr[inmost, iminu]

            # Select only recordings with minimum number of units.
            sel_rt = nunits.index[nunits >= min_nunits]
            nScores = Scores.loc[sel_rt].copy()

            # Nothing to plot.
            if nScores.empty:
                ax_scr.axis('off')
                continue

            # Prepare data.
            dtasks = nScores.index.get_level_values(1).unique()  # in data
            dScores = {task: pd.DataFrame(nScores.xs(task, level=1).unstack(),
                                          columns=['accuracy'])
                       for task in dtasks}
            lScores = pd.concat(dScores, axis=0)
            lScores['time'] = lScores.index.get_level_values(1)
            lScores['task'] = lScores.index.get_level_values(0)
            lScores['rec'] = lScores.index.get_level_values(2)
            lScores.index = np.arange(len(lScores.index))

            # Add altered task names for legend plotting.
            nrecs = {task: len(nScores.xs(task, level=1)) for task in dtasks}
            my_format = lambda x: '{} (n={})'.format(x, nrecs[x])
            lScores['task_nrecs'] = lScores['task'].apply(my_format)

            # Plot as time series.
            sns.tsplot(lScores, time='time', value='accuracy', unit='rec',
                       condition='task_nrecs', ci=ci, n_boot=n_boot, ax=ax_scr)

            # Add chance level line.
            chance_lvl = 1.0 / nvals
            putil.add_chance_level(ax=ax_scr, ylevel=chance_lvl)

            # Add stimulus periods.
            putil.plot_periods(prds, ax=ax_scr)

            # Format plot.
            title = ('{} most DS units'.format(n_most_DS)
                     if n_most_DS != 0 else 'all units')
            title += (', recordings with at least {} units'.format(min_nunits)
                      if (min_nunits > 1 and len(list_min_nunits) > 1) else '')
            ytitle = 1.0
            putil.set_labels(ax_scr, tlab, ylab_scr, title, ytitle)

    # Match axes across decoding plots.
    [putil.sync_axes(axs_scr[inmost, :], sync_y=True)
     for inmost in range(axs_scr.shape[0])]

    # Save plots.
    list_n_most_DS_str = [str(i) if i != 0 else 'all' for i in list_n_most_DS]
    par_kws['n_most_DS'] = ', '.join(list_n_most_DS_str)
    title = decutil.fig_title(res_dir, **par_kws)
    title += '\n{}% CE with {} bootstrapped subsamples'.format(ci, int(n_boot))
    fs_title = 'large'
    w_pad, h_pad = 3, 3

    par_kws['n_most_DS'] = '_'.join(list_n_most_DS_str)
    ffig = decutil.fig_fname(res_dir, 'combined_score', fformat, **par_kws)
    putil.save_fig(ffig, fig_scr, title, fs_title, w_pad=w_pad, h_pad=h_pad)
