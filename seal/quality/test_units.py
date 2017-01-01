#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of functions for plotting unit activity and direction selectivity
for different sets of trials or trial periods.

@author: David Samu
"""

import os

import scipy as sp
import pandas as pd

from seal.util import util
from seal.plot import putil, pplot
from seal.object import constants
from seal.quality import test_sorting


# Figure size constants
subw = 7
w_pad = 5


# %% Quality tests across tasks.

def quality_test(UA, ftempl=None, plot_QM=False, match_scales=True):
    """Test and plot quality metrics of recording and spike sorting """

    # Init plotting theme.
    putil.set_style('notebook', 'white')

    # For each unit over all tasks.
    for uid in UA.uids():

        # Init figure.
        if plot_QM:
            fig, gsp, _ = putil.get_gs_subplots(nrow=1, ncol=len(UA.tasks()),
                                                subw=subw, subh=1.6*subw)
            wf_axs, amp_axs, dur_axs, amp_dur_axs, rate_axs = ([], [], [],
                                                               [], [])

        for i, task in enumerate(UA.tasks()):

            # Do quality test.
            u = UA.get_unit(uid, task)
            res = test_sorting.test_qm(u)

            # Plot QC results.
            if plot_QM:

                if res is not None:
                    ax_res = test_sorting.plot_qm(u, fig=fig, sps=gsp[i],
                                                  **res)

                    # Collect axes.
                    ax_wfs, ax_wf_amp, ax_wf_dur, ax_amp_dur, ax_rate = ax_res
                    wf_axs.extend(ax_wfs)
                    amp_axs.append(ax_wf_amp)
                    dur_axs.append(ax_wf_dur)
                    amp_dur_axs.append(ax_amp_dur)
                    rate_axs.append(ax_rate)

                else:
                    mock_ax = putil.embed_gsp(gsp[i], 1, 1)
                    putil.add_mock_axes(fig, mock_ax[0, 0])

        if plot_QM:

            # Match scale of y axes across tasks.
            if match_scales:
                putil.sync_axes(wf_axs, sync_x=True, sync_y=True)
                putil.sync_axes(amp_axs, sync_y=True)
                putil.sync_axes(dur_axs, sync_y=True)
                putil.sync_axes(amp_dur_axs, sync_x=True, sync_y=True)
                putil.sync_axes(rate_axs, sync_y=True)
                [putil.move_event_lbls(ax, yfac=0.92) for ax in rate_axs]

            # Save figure.
            if ftempl is not None:
                uid_str = util.format_uid(uid)
                title = uid_str.replace('_', ' ')
                fname = ftempl.format(uid_str)
                putil.save_gsp_figure(fig, gsp, fname, title, rect_height=0.92,
                                      w_pad=w_pad)


def DS_test(UA, ftempl=None, match_scales=False, nrate=None):
    """Plot responses to all 8 directions and polar plot in the center."""

    # Init plotting theme.
    putil.set_style('notebook', 'white')

    # For each unit over all tasks.
    for uid in UA.uids():

        # Init figure.
        fig, gsp, _ = putil.get_gs_subplots(nrow=1, ncol=len(UA.tasks()),
                                            subw=subw, subh=subw)
        task_rate_axs, task_polar_axs = [], []

        # Plot direction response of unit in each task.
        for task, sps in zip(UA.tasks(), gsp):
            u = UA.get_unit(uid, task)

            # Plot DR of unit.
            res = u.plot_DR(nrate, fig, sps)
            if res is not None:
                ax_polar, rate_axs = res
                task_rate_axs.extend(rate_axs)
                task_polar_axs.append(ax_polar)
            else:
                mock_ax = putil.embed_gsp(sps, 1, 1)
                putil.add_mock_axes(fig, mock_ax[0, 0])

        # Match scale of y axes across tasks.
        if match_scales:
            putil.sync_axes(task_rate_axs, sync_y=True)
            putil.sync_axes(task_polar_axs, sync_y=True)

        # Save figure.
        if ftempl is not None:
            uid_str = util.format_uid(uid)
            title = None  # uid_str.replace('_', ' ')
            fname = ftempl.format(uid_str)
            putil.save_gsp_figure(fig, gsp, fname, title,
                                  rect_height=0.92, w_pad=w_pad)


def rate_DS_summary(UA, ftempl=None, match_scales=False, nrate=None):
    """Test unit responses within trails."""

    # Init plotting theme.
    putil.set_style('notebook', 'white')

    # For each unit over all tasks.
    for uid in UA.uids():

        # Init figure.
        fig, gsp, _ = putil.get_gs_subplots(nrow=1, ncol=len(UA.tasks()),
                                            subw=subw, subh=12)
        task_rate_axs, task_polar_axs, task_tuning_axs = [], [], []

        # Plot direction response of unit in each task.
        for task, sps in zip(UA.tasks(), gsp):
            u = UA.get_unit(uid, task)

            res = u.plot_rate_DS(nrate, fig, sps)
            if res is not None:
                rate_axs, ax_polar, ax_tuning = res
                task_rate_axs.extend(rate_axs)
                task_polar_axs.append(ax_polar)
                task_tuning_axs.append(ax_tuning)
            else:
                mock_ax = putil.embed_gsp(sps, 1, 1)
                putil.add_mock_axes(fig, mock_ax[0, 0])

        # Match scale of y axes across tasks.
        if match_scales:
            putil.sync_axes(task_rate_axs, sync_y=True)
            [putil.move_signif_lines(ax) for ax in task_rate_axs]
            putil.sync_axes(task_polar_axs, sync_y=True)
            putil.sync_axes(task_tuning_axs, sync_y=True)

        # Save figure.
        if ftempl is not None:
            uid_str = util.format_uid(uid)
            title = None  # uid_str.replace('_', ' ')
            fname = ftempl.format(uid_str)
            putil.save_gsp_figure(fig, gsp, fname, title, rect_height=0.92,
                                  w_pad=w_pad)


def create_montage(UA, ftempl_qm, ftempl_dr, ftempl_sum, ftempl_mont):
    """Create montage image of figures created during preprocessing."""

    for uid in UA.uids():
        uid_str = util.format_uid(uid)

        # Get file names.
        fqm = ftempl_qm.format(uid_str)
        fdr = ftempl_dr.format(uid_str)
        fsum = ftempl_sum.format(uid_str)
        fmont = ftempl_mont.format(uid_str)

        # Check if figures exist.
        flist = [f for f in (fqm, fdr, fsum) if os.path.isfile(f)]

        # Create output folder.
        util.create_dir(fmont)

        # Create montage image.
        cmd = ('montage {}'.format(' '.join(flist)) +
               ' -tile x3 -geometry +50+100 {}'.format(fmont))
        os.system(cmd)


def rec_stability_test(UA, fname=None):
    """Check stability of recording session across tasks."""

    # Init plotting theme.
    putil.set_style('notebook', 'white')

    # Init params.
    periods = constants.tr_prd

    # Init figure.
    fig, gsp, ax_list = putil.get_gs_subplots(nrow=len(periods.index), ncol=1,
                                              subw=10, subh=2.5)

    for prd, ax in zip(periods.index, ax_list):

        # Calculate and plot firing rate during given period in each trial
        # across session for all units.
        colors = putil.get_colors()
        task_stats = pd.DataFrame(columns=['t_start', 'label'])
        for task, color in zip(UA.rec_task_order(), colors):

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
            mean_rate, sem_rate = tr_rates.mean(), tr_rates.std()
            lower, upper = mean_rate-sem_rate, mean_rate+sem_rate
            lower[lower < 0] = 0  # remove negative values
            ax.fill_between(tr_time, lower, upper, zorder=2, alpha=.5,
                            facecolor='grey', edgecolor='grey')
            pplot.lines(tr_time, mean_rate, lw=2, color='k', ax=ax)

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

            task_stats.loc[task] = (tr_times.min(), task_lbl)

        # Add task labels after all tasks have been plotted (limits are set).
        putil.plot_events(task_stats, lbl_height=0.75, lbl_ha='left',
                          lbl_rotation=0, ax=ax)

        # Format plot.
        xlab = 'Recording time (s)' if prd == periods.index[-1] else None
        putil.set_labels(ax, xlab=xlab, ylab=prd)
        putil.set_spines(ax, left=False)

    # Save figure.
    title = 'Recording stability of ' + UA.Name
    putil.save_gsp_figure(fig, gsp, fname, title, rect_height=0.95)
