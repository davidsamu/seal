# -*- coding: utf-8 -*-

"""
Functions to plot decoding results.

@author: David Samu
"""

import numpy as np
import pandas as pd
import seaborn as sns

from seal.decoding import decode
from seal.plot import putil
from seal.util import util, constants


tlab = 'Time since S1 onset (ms)'
ylab_scr = 'decoding accuracy'
tlim = [-1000, 3500]
ylim_scr = [0, 1]


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


def plot_scores(ax, Scores, ShfldScores, nvals=None, prds=None, col='b',
                shflcol='g', xlim=None, ylim=ylim_scr, xlab=tlab,
                ylab=ylab_scr, title='', ytitle=1.04):
    """Plot decoding accuracy results."""

    # Plot scores.
    plot_score_set(Scores, ax, color=col)

    # Plot shuffled scores.
    if (ShfldScores is not None) and not ShfldScores.isnull().all().all():
        plot_score_set(ShfldScores, ax, color=shflcol)

        # Add legend.
        lgn_patches = [putil.get_artist('synchronous', col),
                       putil.get_artist('pseudo-population', shflcol)]
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


def plot_scores_weights(recs, tasks, stims, feat, cond, zscore, res_dir,
                        nrate, ncv, n_pshfl, sep_err_trs):
    """
    Plot prediction scores and model weights for given recording and analysis.
    """

    # Init.
    putil.set_style('notebook', 'ticks')

    # Load results.
    prd_pars = util.init_stim_prds(stims, feat, cond, zscore,
                                   constants.fixed_tr_prds)
    fres = decode.res_fname(res_dir+'results/', feat, nrate, ncv, n_pshfl,
                            sep_err_trs, cond)
    rt_res = util.read_objects(fres, 'rt_res')

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
        print('\n' + rec)
        for itask, task in enumerate(tasks):
            print('    ' + task)

            # Init figures.
            ax_scr = axs_scr[irec, itask]
            ax_wgt = axs_wgt[irec, itask]

            # Check if any result exists for rec-task combination.
            if (((rec, task) not in rt_res.keys()) or
               (not len(rt_res[(rec, task)].keys()))):
                print('No result found for {} - {}.'.format(rec, task))

                ax_scr.axis('off')
                ax_wgt.axis('off')
                continue

            # Init data.
            res = rt_res[(rec, task)]
            cols = sns.color_palette('hls', len(res.keys()))

            for v, col in zip(res.keys(), cols):
                vres = res[v]
                Scores = vres['Scores']
                Coefs = vres['Coefs']
                # C = vres['C']
                ShfldScores = vres['ShfldScores']
                nunits = vres['nunits']
                # ntrials = vres['ntrials']
                # prd_pars = vres['prd_pars']
                nvals = len(Coefs.index.get_level_values(1).unique())
                if nvals == 1:  # binary case
                    nvals = 2

                # Plot decoding accuracy.
                plot_scores(ax_scr, Scores, ShfldScores, col=col)

            # Add labels.
            title = '{} {}, {} units'.format(rec, task, nunits)
            putil.set_labels(ax_scr, tlab, ylab_scr, title, ytitle=1.04)

            # Add chance level line and stimulus periods.
            chance_lvl = 1.0 / nvals
            putil.add_chance_level(ax=ax_scr, ylevel=chance_lvl)
            prds = [[stim] + list(constants.fixed_tr_prds.loc[stim])
                    for stim in prd_pars.index]
            putil.plot_periods(prds, ax=ax_scr)

            # Plot unit weights over time.
            plot_weights(ax_wgt, Coefs, prds, tlim, tlab, title=title)

    # Match axes across decoding plots.
    # [putil.sync_axes(axs_scr[:, itask], sync_y=True)
    #  for itask in range(axs_scr.shape[1])]

    # Save plots.
    title = decode.fig_title(res_dir, feat, nrate, ncv,
                             n_pshfl, sep_err_trs, cond)
    fs_title = 'large'
    ytitle = 1.05
    w_pad, h_pad = 3, 3

    # Performance.
    ffig = decode.fig_fname(res_dir + 'score_', feat, nrate, ncv, n_pshfl,
                            sep_err_trs, cond)
    putil.save_fig(ffig, fig_scr, title, ytitle, fs_title,
                   w_pad=w_pad, h_pad=h_pad)

    # Weights.
    ffig = decode.fig_fname(res_dir + 'weight_', feat, nrate, ncv, n_pshfl,
                            sep_err_trs, cond)
    putil.save_fig(ffig, fig_wgt, title, ytitle, fs_title,
                   w_pad=w_pad, h_pad=h_pad)

# TODO: update this
def plot_score_weight_multi_rec(recs, tasks, stims, feat, cond, zscore,
                                res_dir, nrate, ncv, n_pshfl, sep_err_trs):
    """
    Plot prediction scores and model weights of analysis for multiple
    recordings.
    """

    # Init.
    putil.set_style('notebook', 'ticks')

    # Load results.
    prd_pars = util.init_stim_prds(stims, feat, cond, zscore,
                                   constants.fixed_tr_prds)
    fres = decode.res_fname(res_dir+'results/', feat, nrate, ncv, n_pshfl,
                            sep_err_trs, cond)
    rt_res = util.read_objects(fres, 'rt_res')

    # Create figure.
    # For prediction scores.
    ret = putil.get_gs_subplots(nrow=1, ncol=len(tasks),
                                subw=8, subh=6, create_axes=True)
    fig_scr, _, axs_scr = ret

    dict_lScores = {}
    print('\nPlotting multi-recording results...')
    for itask, task in enumerate(tasks):
        print('    ' + task)
        for irec, rec in enumerate(recs):

            # Check if results exist for rec-task combination.
            if (((rec, task) not in rt_res.keys()) or
               (not len(rt_res[(rec, task)].keys()))):
                print('No result found for {} - {}.'.format(rec, task))
                continue

            # Init data.
            res = rt_res[(rec, task)]
            Scores = res['Scores']
            Coefs = res['Coefs']
            # C = res['C']
            # ShfldScores = res['ShfldScores']
            # nunits = res['nunits']
            # ntrials = res['ntrials']
            # prd_pars = res['prd_pars']
            nvals = len(Coefs.index.get_level_values(1).unique())
            if nvals == 1:  # binary case
                nvals = 2

            # Plot accuracy over time.
            # Unstack dataframe with results.
            lScores = pd.DataFrame(Scores.unstack(), columns=['score'])
            lScores['time'] = lScores.index.get_level_values(0)
            lScores['fold'] = lScores.index.get_level_values(1)
            lScores.index = np.arange(len(lScores.index))

            dict_lScores[rec] = lScores

        # Concatenate accuracy scores from every recording.
        all_lScores = pd.concat(dict_lScores)
        all_lScores['rec'] = all_lScores.index.get_level_values(0)
        all_lScores.index = np.arange(len(all_lScores.index))

        # Plot decoding results.
        nrec = len(all_lScores['rec'].unique())
        title = '{}, {} recordings'.format(task, nrec)
        ytitle = 1.0
        prds = [[stim] + list(constants.fixed_tr_prds.loc[stim])
                for stim in prd_pars.index]

        # Plot time series.
        ax_scr = axs_scr[0, itask]
        palette = sns.color_palette('muted')
        sns.tsplot(all_lScores, time='time', value='score', unit='fold',
                   condition='rec', color=palette, ax=ax_scr)

        # Add chance level line.
        # This currently plots all nvals combined across stimulus period!
        if nvals is not None:
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
    title = decode.fig_title(res_dir, feat, nrate, ncv, n_pshfl, sep_err_trs)
    fs_title = 'large'
    ytitle = 1.10
    w_pad, h_pad = 3, 3
    ffig = decode.fig_fname(res_dir + 'all_scores_', feat, nrate, ncv, n_pshfl,
                            sep_err_trs)
    putil.save_fig(ffig, fig_scr, title, ytitle, fs_title,
                   w_pad=w_pad, h_pad=h_pad)
