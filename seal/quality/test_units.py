#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of functions for plotting unit activity and direction selectivity
for different sets of trials or trial periods.

@author: David Samu
"""

import warnings

import numpy as np
import scipy as sp
import pandas as pd

from seal.util import util
from seal.quality import test_sorting
from seal.plot import putil, pplot, pselectivity, pquality


# Figure size constants
subw = 7
w_pad = 5


# %% Quality tests across tasks.

def get_selection_params(u, UnTrSel=None):
    """Return unit and trial selection parameters of unit provided by user."""

    if UnTrSel is None or u.is_empty():
        return None, None, None

    # Init.
    rec, ch, idx, task = u.get_utid()
    ntrials = len(u.TrialParams.index)

    # Find unit in selection table.
    row = UnTrSel.ix[((UnTrSel.recording == rec) & (UnTrSel.channel == ch) &
                      (UnTrSel['unit index'] == idx) & (UnTrSel.task == task))]
    uname = 'rec {}, ch {}, idx {}, task {}'.format(rec, ch, idx, task)

    # Unit not in table.
    if not len(row.index):
        warnings.warn(uname + ': not found in selection table.')
        return None, None, None

    # If there's more than one match.
    if len(row.index) > 1:
        warnings.warn(uname + ': multiple rows found for unit ' +
                      'in selection table, using first match.')
        row = row.iloc[0:1]

    # Get index of first and last trials to include.
    include = bool(int(row['unit included']))
    first_tr = int(row['first included trial']) - 1  # indexing starts with 0
    last_tr = int(row['last included trial'])   # inclusive --> exclusive

    # Set values outside of range to limits.
    first_tr = max(first_tr, 0)
    last_tr = min(last_tr, ntrials)
    if last_tr == -1:  # -1: include all the way to the end
        last_tr = ntrials

    # Check some simple cases of data inconsistency.
    if include and first_tr >= last_tr:
        warnings.warn(uname + ': index of first included trial is larger or' +
                      ' equal to last in selection table! Excluding unit.')
        include = False

    return include, first_tr, last_tr


def quality_test(UA, ftempl=None, plot_qm=False, fselection=None):
    """Test and plot quality metrics of recording and spike sorting """

    # Init plotting theme.
    putil.set_style('notebook', 'ticks')

    # Import unit&trial selection file.
    UnTrSel = pd.read_excel(fselection) if fselection is not None else None

    # For each unit over all tasks.
    for uid in UA.uids():
        print(uid)

        # Init figure.
        if plot_qm:
            fig, gsp, _ = putil.get_gs_subplots(nrow=1, ncol=len(UA.tasks()),
                                                subw=subw, subh=1.6*subw)
            wf_axs, amp_axs, dur_axs, amp_dur_axs, rate_axs = ([], [], [],
                                                               [], [])

        for i, task in enumerate(UA.tasks()):

            # Do quality test.
            u = UA.get_unit(uid, task)
            include, first_tr, last_tr = get_selection_params(u, UnTrSel)
            res = test_sorting.test_qm(u, include, first_tr, last_tr)

            # Plot QC results.
            if plot_qm:

                if res is not None:
                    ax_res = pquality.plot_qm(u, fig=fig, sps=gsp[i], **res)

                    # Collect axes.
                    ax_wfs, ax_wf_amp, ax_wf_dur, ax_amp_dur, ax_rate = ax_res
                    wf_axs.extend(ax_wfs)
                    amp_axs.append(ax_wf_amp)
                    dur_axs.append(ax_wf_dur)
                    amp_dur_axs.append(ax_amp_dur)
                    rate_axs.append(ax_rate)

                else:
                    putil.add_mock_axes(fig, gsp[i])

        if plot_qm:

            # Match axis scales across tasks.
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


def report_unit_exclusion_stats(UA):
    """Exclude low quality units."""

    exclude = []
    for u in UA.iter_thru(excl=True):
        exclude.append(u.is_excluded())

    # Report unit exclusion results.
    n_tot = len(exclude)
    n_exc, n_inc = sum(exclude), sum(np.invert(exclude))
    perc_exc, perc_inc = 100 * n_exc / n_tot, 100 * n_inc / n_tot
    rep_str = '  {} / {} ({:.1f}%) units {} analysis.'
    print(rep_str.format(n_inc, n_tot, perc_inc, 'included into'))
    print(rep_str.format(n_exc, n_tot, perc_exc, 'excluded from'))


def DR_plot(UA, ftempl=None, match_scales=False):
    """Plot responses to all 8 directions and polar plot in the center."""

    # Init plotting theme.
    putil.set_style('notebook', 'ticks')

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
            res = (pselectivity.plot_DR_3x3(u, fig, sps)
                   if not u.is_excluded() and u.to_plot() else None)
            if res is not None:
                ax_polar, rate_axs = res
                task_rate_axs.extend(rate_axs)
                task_polar_axs.append(ax_polar)
            else:
                putil.add_mock_axes(fig, sps)

        # Match scale of y axes across tasks.
        if match_scales:
            putil.sync_axes(task_polar_axs, sync_y=True)
            putil.sync_axes(task_rate_axs, sync_y=True)
            [putil.adjust_decorators(ax) for ax in task_rate_axs]

        # Save figure.
        if ftempl is not None:
            uid_str = util.format_uid(uid)
            title = uid_str.replace('_', ' ')
            fname = ftempl.format(uid_str)
            putil.save_gsp_figure(fig, gsp, fname, title,
                                  rect_height=0.92, w_pad=w_pad)


def selectivity_summary(UA, ftempl=None, match_scales=False):
    """Test unit responses within trails."""

    # Init plotting theme.
    putil.set_style('notebook', 'ticks')

    # For each unit over all tasks.
    for uid in UA.uids():

        # Init figure.
        fig, gsp, _ = putil.get_gs_subplots(nrow=1, ncol=len(UA.tasks()),
                                            subw=16, subh=32)
        ls_axs, ds_axs = [], []

        # Plot stimulus response summary plot of unit in each task.
        for task, sps in zip(UA.tasks(), gsp):
            u = UA.get_unit(uid, task)

            res = (pselectivity.plot_selectivity(u, fig, sps)
                   if not u.is_excluded() and u.to_plot() else None)
            if res is not None:
                ls_axs.extend(res[0])
                ds_axs.extend(res[1])

            else:
                putil.add_mock_axes(fig, sps)

        # Match scale of y axes across tasks.
        if match_scales:
            for rate_axs in [ls_axs, ds_axs]:
                putil.sync_axes(rate_axs, sync_y=True)
                [putil.adjust_decorators(ax) for ax in rate_axs]

        # Save figure.
        if ftempl is not None:
            uid_str = util.format_uid(uid)
            title = uid_str.replace('_', ' ')
            fname = ftempl.format(uid_str)
            putil.save_gsp_figure(fig, gsp, fname, title, rect_height=0.96,
                                  w_pad=w_pad)


def rec_stability_test(UA, periods=None, fname=None):
    """Check stability of recording session across tasks."""

    # Init.
    if periods is None:
        periods = ['whole trial', 'fixation']

    # Init figure.
    fig, gsp, axs = putil.get_gs_subplots(nrow=len(periods.index), ncol=1,
                                          subw=10, subh=2.5, create_axes=True,
                                          as_array=False)

    for prd, ax in zip(periods, axs):

        # Calculate and plot firing rate during given period in each trial
        # across session for all units.
        colors = putil.get_colors()
        task_stats = pd.DataFrame(columns=['t_start', 't_stops', 'label'])
        for task, color in zip(UA.tasks(), colors):

            # Get activity of all units in task.
            tr_rates = []
            for u in UA.iter_thru([task]):
                rates = u.get_prd_rates(prd, tr_time_idx=True)
                tr_rates.append(util.remove_dim_from_series(rates))
            tr_rates = pd.DataFrame(tr_rates)

            # Not (non-empty and included) unit during task.
            if not len(tr_rates.index):
                continue

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

            task_stats.loc[task] = (tr_times.min(), tr_times.max(), task_lbl)

        # Set axes limits.
        tmin, tmax = task_stats.t_start.min(), task_stats.t_stops.max()
        putil.set_limits(ax, xlim=(tmin, tmax))

        # Add task labels after all tasks have been plotted.
        putil.plot_events(task_stats[['t_start', 'label']], lbl_height=0.75,
                          lbl_ha='left', lbl_rotation=0, ax=ax)

        # Format plot.
        xlab = 'Recording time (s)' if prd == periods[-1] else None
        putil.set_labels(ax, xlab=xlab, ylab=prd)
        putil.set_spines(ax, left=False)

    # Save figure.
    title = 'Recording stability of ' + UA.Name
    putil.save_gsp_figure(fig, gsp, fname, title, rect_height=0.92)
