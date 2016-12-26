#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 12:30:16 2016

Collection of functions for plotting unit activity and direction selectivity
for different sets of trials or trial periods.

@author: David Samu
"""

import scipy as sp
import pandas as pd

from quantities import s

from seal.util import util
from seal.plot import putil, pplot
from seal.object import constants


# %% Quality tests across tasks.

def DS_test(UA, nrate=None, ftempl=None, match_scale=True):
    """Plot responses to all 8 directions and polar plot in the center."""

    # Init plotting theme.
    putil.set_style('notebook', 'white')

    # Init data.
    tasks, uids = UA.tasks(), UA.uids()

    # For each unit over all tasks.
    for uid in uids:

        # Init figure.
        fig, gsp, _ = putil.get_gs_subplots(nrow=1, ncol=len(tasks),
                                            subw=7, subh=7,
                                            create_axes=False)
        task_rate_axs, task_polar_axs = [], []

        # Plot direction response of unit in each task.
        task_gsp_u = zip(tasks, gsp, UA.iter_thru(tasks, [uid], miss=True,
                                                  excl=True))
        for task, sps, u in task_gsp_u:

            if u.is_empty():
                mock_ax = putil.embed_gsp(sps, 1, 1)
                putil.add_mock_axes(fig, mock_ax[0, 0])
                continue

            ax_polar, rate_axs = u.plot_DR(nrate, fig, sps)
            task_rate_axs.extend(rate_axs)
            task_polar_axs.append(ax_polar)

        # Match scale of y axes across tasks.
        if match_scale:
            putil.sync_axes(task_rate_axs, sync_y=True)
            putil.sync_axes(task_polar_axs, sync_y=True)

        # Format and save figure.
        if ftempl is not None:
            uid_str = util.format_uid(uid)
            title = uid_str.replace('_', ' ')
            fname = ftempl.format(uid_str)
            putil.save_gsp_figure(fig, gsp, fname, title,
                                  rect_height=0.92, w_pad=5)


def rate_DS_summary(UA, nrate=None, ftempl=None, match_scale=True):
    """Test unit responses within trails."""

    # Init plotting theme.
    putil.set_style('notebook', 'white')

    # Init data
    tasks, uids = UA.tasks(), UA.uids()
    ntask = len(tasks)

    # For each unit over all tasks.
    for uid in uids:

        # Init figure.
        fig, gsp, _ = putil.get_gs_subplots(nrow=1, ncol=ntask,
                                            subw=7, subh=12,
                                            create_axes=False)
        task_rate_axs, task_polar_axs, task_tuning_axs = [], [], []

        # Plot direction response of unit in each task.
        for i, u in enumerate(UA.iter_thru(tasks, [uid],
                                           miss=True, excl=True)):
            sps = gsp[i]

            if u.is_empty():
                mock_ax = putil.embed_gsp(sps, 1, 1)
                putil.add_mock_axes(fig, mock_ax[0, 0])
                continue

            rate_axs, ax_polar, ax_tuning = u.plot_rate_DS(nrate, fig, sps)
            task_rate_axs.extend(rate_axs)
            task_polar_axs.append(ax_polar)
            task_tuning_axs.append(ax_tuning)

        # Match scale of y axes across tasks.
        if match_scale:
            putil.sync_axes(task_rate_axs, sync_y=True)
            [putil.move_signif_lines(ax) for ax in task_rate_axs]
            putil.sync_axes(task_polar_axs, sync_y=True)
            putil.sync_axes(task_tuning_axs, sync_y=True)

        # Format and save figure.
        if ftempl is not None:
            uid_str = util.format_uid(uid)
            title = uid_str.replace('_', ' ')
            fname = ftempl.format(uid_str)
            putil.save_gsp_figure(fig, gsp, fname, title, rect_height=0.95)


def check_recording_stability(UA, fname):
    """Check stability of recording session across tasks."""

    # Init params.
    periods = constants.tr_prd

    # Init figure.
    fig, gsp, ax_list = putil.get_gs_subplots(nrow=len(periods.index), ncol=1,
                                              subw=10, subh=2.5,
                                              as_array=False)
    colors = putil.get_colors()

    for prd, ax, color in zip(periods.index, ax_list, colors):

        # Calculate and plot firing rate during given period in each trial
        # across session for all units.
        for task in UA.rec_task_order():

            # Get activity of all units in task.
            tr_rates = []
            for u in UA.iter_thru([task]):
                t1s, t2s = u.pr_times(prd, concat=False)
                rates = u.get_prd_rates(t1s=t1s, t2s=t2s, tr_time_idx=True)
                tr_rates.append(util.remove_dim_from_series(rates))
            tr_rates = pd.DataFrame(tr_rates)

            # Plot each rate in task.
            tr_times = tr_rates.columns
            pplot.lines(tr_times, tr_rates.T, zorder=1, alpha=0.5,
                        color=color, ax=ax)

            # Plot mean +- sem rate.
            tr_time = tr_rates.columns
            mean_rate, sem_rate = tr_rates.mean(), tr_rates.sem()
            lower, upper = mean_rate-sem_rate, mean_rate+sem_rate
            lower[lower < 0] = 0  # remove negative values
            ax.fill_between(tr_time, lower, upper, zorder=2,
                            alpha=.5, facecolor='grey', edgecolor='grey')
            pplot.lines(tr_time, mean_rate, lw=2, color='k', ax=ax)

            # Add task start marker lines.
            start_evt = pd.Series()
            start_evt[task] = tr_times.min() * s
            putil.plot_events(start_evt, t_unit=s, lbl_height=0.75,
                              lbl_ha='left', lbl_rotation=0, ax=ax)

            # Add task stats.
            task_lbl = '{}, {} units'.format(task, len(tr_rates.index))

            # Add grand mean FR.
            task_lbl += '\nFR: {:.1f} sp/s'.format(tr_rates.mean().mean())

            # Calculate linear trend to test gradual drift.
            slope, _, _, p_value, _ = sp.stats.linregress(tr_times, mean_rate)
            slope = 3600*slope  # convert to change in spike per hour
            pval = util.format_pvalue(p_value, max_digit=3)
            task_lbl += '\n$\delta$FR: {:.1f} sp/s/h'.format(slope)
            task_lbl += '\n{}'.format(pval)

        # Set limits and add labels to plot.
        xlab = 'Recording time (s)' if prd == periods.index[-1] else None
        putil.set_labels(ax, xlab=xlab, ylab=prd_name)

    # Format and save figure.
    title = 'Recording stability of ' + UA.Name
    putil.save_gsp_figure(fig, gsp, fname, title, rect_height=0.95)
