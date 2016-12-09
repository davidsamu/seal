#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 12:30:16 2016

Collection of functions for plotting unit activity and direction selectivity
for different sets of trials or trial periods.

@author: David Samu
"""


import numpy as np
import scipy as sp
import pandas as pd

from quantities import s, ms, deg
from collections import OrderedDict as OrdDict

from seal.util import util
from seal.plot import plot
from seal.object import constants


# %% Quality tests across tasks.

def direction_response_test(UA, nrate=None, ftempl=None,
                            match_scale=False):
    """Plot responses to all 8 directions and polar plot in the center."""

    # Init data
    tasks, uids = UA.tasks(), UA.uids()
    ntask = len(tasks)

    # Set up stimulus parameters (time period to plot, color, etc).
    prds = constants.ext_stim_prds.prds
    s1_start, s1_stop = prds.loc['S1', 'start'] + 500*ms, prds.loc['S1', 'end']
    s2_start, s2_stop = prds.loc['S2', 'start'], prds.loc['S2', 'end']
    stims = pd.DataFrame([(s1_start, s1_stop, 'b'), (s2_start, s2_stop, 'g')],
                         index=['S1', 'S2'], columns=['start', 'stop', 'color'])
    stims['len'] = [float(stims.loc[s, 'stop'] - stims.loc[s, 'start'])
                    for s in stims.index]

    # Reorder directions to match with order of axes.
    rr_order = [3, 2, 1, 4, 0, 5, 6, 7]
    dir_order = constants.all_dirs[rr_order]
    dir_order = np.insert(dir_order, 4, np.nan) * deg  # add polar plot

    # For each unit over all tasks.
    for uid in uids:

        # Init figure.
        fig, gsp, _ = plot.get_gs_subplots(nrow=1, ncol=ntask, subw=7, subh=7,
                                           create_axes=False)
        rate_axs, polar_axs = [], []

        # Plot direction response of unit in each task.
        task_gsp_u = zip(tasks, gsp, UA.iter_thru(tasks, [uid], miss=True,
                                                  excl=True))
        for task, sps, u in task_gsp_u:

            task_gsp = plot.embed_gsp(sps, 3, 3)

            for i, u_gsp in enumerate(task_gsp):

                # Polar plot.
                if i == 4:
                    ax_polar = fig.add_subplot(u_gsp, polar=True)
                    if u.is_empty:
                        ax_polar.axis('off')
                    else:
                        # For each stimulus.
                        for stim in stims.index:
                            dres = u.calc_dir_response(stim)
                            dirs = np.array(dres.index)*deg
                            resp = util.dim_series_to_array(dres['mean'])
                            c = stims.loc[stim, 'color']
                            plot.polar_direction_response(dirs, resp, color=c,
                                                          ax=ax_polar)
                        # Remove y-axis ticklabels.
                        plot.hide_ticks(ax_polar, 'y')

                        polar_axs.append(ax_polar)

                # Raster-rate plot.
                else:
                    # For each stimulus.
                    stim_rr_gsp = plot.embed_gsp(u_gsp, 1, stims.shape[0],
                                                 width_ratios=stims.len,
                                                 wspace=0)
                    gsp_stime_list = zip(stims.index, stim_rr_gsp)
                    for stim, stim_rr_gsp in gsp_stime_list:

                        # Prepare plotting.
                        st1, st2, c = stims.loc[stim, ['start', 'stop', 'color']]
                        rr_gsp = plot.embed_gsp(stim_rr_gsp, 2, 1)

                        if u.is_empty:
                            rr_res = plot.empty_raster_rate(fig, rr_gsp, 1)
                            raster_axs, rate_ax = rr_res
                        # Plot raster&rate plot.
                        else:
                            dd_trs = u.trials_by_param_values(stim + 'Dir',
                                                              [dir_order[i]])
                            # Leave axes empty if there's not trial recorded.
                            if not dd_trs[0].num_trials():
                                continue

                            res = u.plot_raster_rate(nrate, dd_trs, st1, st2,
                                                     colors=[c], legend=False,
                                                     no_labels=True, fig=fig,
                                                     outer_gsp=rr_gsp)
                            _, raster_axs, rate_ax = res
                            # Remove y-axis ticklabels from raster plot.
                            [plot.hide_ticks(ax) for ax in raster_axs]

                            # Remove axis spines and ticks.
                            show_xticks = (i == 0)
                            show_yticks = (i == 0) & (stim == stims.index[0])
                            plot.hide_ticks(rate_ax, show_xticks, show_yticks)
                            left_spine = (stim == stims.index[0])
                            plot.show_spines(bottom=True, left=left_spine,
                                             ax=rate_ax)

                            rate_axs.append(rate_ax)

                        # Add task name as title (to 2nd axes).
                        if i == 1 and stim == stims.index[0]:
                            title = task
                            plot.set_labels(title=title, ytitle=1.10,
                                            title_kwargs={'loc': 'right'},
                                            ax=raster_axs[0])

            # Match scale of y axes, only within task.
            if not match_scale:
                plot.sync_axes(rate_axs, sync_y=True)
                rate_axs = []

        # Match scale of y axes across tasks.
        if match_scale:
            plot.sync_axes(rate_axs, sync_y=True)
            plot.sync_axes(polar_axs, sync_y=True)

        # Format and save figure.
        if ftempl is not None:
            uid_str = util.format_rec_ch_idx(uid)
            title = uid_str.replace('_', ' ')
            fname = ftempl.format(uid_str)
            plot.save_gsp_figure(fig, gsp, fname, title,
                                 rect_height=0.92, w_pad=5)


def rate_DS_summary(UA, nrate=None, ftempl=None, match_scale=False):
    """Test unit responses within trails."""

    # Init data
    tasks, uids = UA.tasks(), UA.uids()
    ntask = len(tasks)

    # Set up stimulus parameters (time period to plot, color, etc).
    prds = constants.ext_stim_prds.prds
    s1_start, s1_stop = prds.loc['S1', 'start'] + 500*ms, prds.loc['S1', 'end']
    s2_start, s2_stop = prds.loc['S2', 'start'], prds.loc['S2', 'end']
    stims = pd.DataFrame([(s1_start, s1_stop), (s2_start, s2_stop)],
                         index=['S1', 'S2'], columns=['start', 'stop'])
    stims['len'] = [float(stims.loc[s, 'stop'] - stims.loc[s, 'start'])
                    for s in stims.index]

    # Plotting params.
    legend_kwargs = {'borderaxespad': 0}

    # For each unit over all tasks.
    for uid in uids:

        # Init figure.
        fig, gsp, _ = plot.get_gs_subplots(nrow=1, ncol=ntask,
                                           subw=6, subh=11,
                                           create_axes=False)
        rate_axs, tuning_axs = [], []

        # Plot direction response of unit in each task.
        task_gsp_u = zip(tasks, gsp, UA.iter_thru(tasks, [uid], miss=True,
                                                  excl=True))
        for task, sps, u in task_gsp_u:

            subplots = ['info', 'rr_all_trs', 'DS_tuning', 'rr_pref_anti']
            height_ratios = [0.2, 1, 1, 1]
            task_gsp = plot.embed_gsp(sps, len(subplots), 1,
                                      height_ratios=height_ratios)

            for subplot, u_gsp in zip(subplots, task_gsp):

                # Info header.
                if subplot == 'info':

                    gsp_info = plot.embed_gsp(u_gsp, 1, 1)
                    if u.is_empty:
                        plot.add_mock_axes(fig, gsp_info[0, 0])
                    else:
                        ax = fig.add_subplot(gsp_info[0, 0])
                        plot.unit_info(u, ax=ax)

                # Raster & rate over all trials.
                if subplot == 'rr_all_trs':

                    rr_gsp = plot.embed_gsp(u_gsp, 2, 1)
                    if u.is_empty:
                        plot.empty_raster_rate(fig, rr_gsp, 1)
                    else:
                        t1, t2 = stims.start.min(), stims.stop.max()
                        res = u.plot_raster_rate(nrate, no_labels=True, t1=t1,
                                                 t2=t2, legend_kwargs=legend_kwargs,
                                                 fig=fig, outer_gsp=rr_gsp)
                        fig, raster_axs, rate_ax = res
                        plot.replace_tr_num_with_tr_name(raster_axs[0], 'all trials')
                        rate_axs.append(rate_ax)

                # Direction tuning.
                if subplot == 'DS_tuning':

                    ds_gsp = plot.embed_gsp(u_gsp, 1, 2)
                    if u.is_empty:
                        plot.empty_direction_selectivity(fig, ds_gsp)
                    else:
                        u.test_DS(no_labels=True, fig=fig, outer_gsp=ds_gsp)
                        # tuning_axs.extend([ax_polar, ax_tuning]) ....

                # Raster & rate in pref & anti trials.
                if subplot == 'rr_pref_anti':

                    stim_rr_gsp = plot.embed_gsp(u_gsp, 1, stims.shape[0],
                                                 width_ratios=stims.len,
                                                 wspace=0.1)

                    for i, (stim, row) in enumerate(stims.iterrows()):
                        dd_rr_gsp = plot.embed_gsp(stim_rr_gsp[0, i], 2, 1)
                        if u.is_empty:
                            plot.empty_raster_rate(fig, dd_rr_gsp, 2)
                        else:
                            pname, t1, t2 = [stim+'Dir'], row.start, row.stop
                            dd_trs = u.dir_pref_anti_trials(stim=stim,
                                                            pname=pname)
                            res = u.plot_raster_rate(nrate, trs=dd_trs,
                                                     t1=t1, t2=t2,
                                                     pvals=[0.05], test='t-test',
                                                     legend_kwargs=legend_kwargs,
                                                     no_labels=True, fig=fig,
                                                     outer_gsp=dd_rr_gsp)
                            fig, raster_axs, rate_ax = res
                            rate_axs.append(rate_ax)

                            # Replace y-axis tickmarks with trial set names.
                            for ax, trs in zip(raster_axs, dd_trs):
                                plot.replace_tr_num_with_tr_name(ax, trs.name)

                            # Hide y-axis tickmarks on second and subsequent rate axes.
                            if i > 0:
                                plot.hide_axes(show_x=True, show_y=False, ax=rate_ax)


            # Match scale of y axes, only within task.
            if not match_scale:
                plot.sync_axes(rate_axs, sync_y=True)
                [plot.move_significance_lines(ax) for ax in rate_axs]
                rate_axs = []

        # Match scale of y axes across tasks.
        if match_scale:
            plot.sync_axes(rate_axs, sync_y=True)
            [plot.move_significance_lines(ax) for ax in rate_axs]
            #plot.sync_axes(tuning_axs, sync_y=True)

        # Format and save figure.
        uid_str = util.format_rec_ch_idx(uid)
        title = uid_str.replace('_', ' ')
        fname = ftempl.format(uid_str)
        plot.save_gsp_figure(fig, gsp, fname, title, rect_height=0.95)


def check_recording_stability(UA, fname):
    """Check stability of recording session across tasks."""

    # Init params.
    periods = constants.tr_prds

    # Init figure.
    fig, gsp, ax_list = plot.get_gs_subplots(nrow=periods.n_prds(), ncol=1,
                                             subw=10, subh=2.5, as_array=False)

    # Init task info.
    rec_tasks = UA.rec_task_order()
    task_stats = pd.DataFrame(columns=['nunits', 'start', 'stop'])
    for task in rec_tasks:
        unit_list = UA.unit_list(tasks=[task])
        task_stats.loc[task, 'nunits'] = len(unit_list)
        tparams = unit_list[0].TrialParams
        task_stats.loc[task, 'start'] = tparams['TrialStart'].iloc[0]
        task_stats.loc[task, 'stop'] = tparams['TrialStop'].iloc[-1]

    for (prd_name, (t1, t2)), ax in zip(periods.prds.iterrows(), ax_list):

        # Calculate and plot firing rate during given period in each trial
        # across session for all units.
        all_FR_prd = OrdDict()
        for uid in UA.uids():

            # Get unit activity in all tasks (including empty and rejected
            # ones for color cycle consistency).
            FR_tr_list = [u.get_rates_by_trial(t1=t1, t2=t2)
                          for u in UA.iter_thru(rec_tasks, [uid],
                                                miss=True, excl=True)]

            # Plot each rate in each task.
            colors = plot.get_colors()
            for FR_tr in FR_tr_list:
                color = next(colors)  # for color consistency
                if FR_tr is not None:
                    plot.lines(FR_tr.index, FR_tr, zorder=1, alpha=0.5,
                               color=color, ax=ax)

            # Save FRs for summary plots and stats.
            FR_tr = pd.concat(FR_tr_list)
            FR_tr.index = util.remove_dim_from_array(FR_tr.index)
            all_FR_prd[uid] = FR_tr

        # Add mean +- std FR.
        all_FR = pd.concat(all_FR_prd, axis=1)
        tr_time = all_FR.index
        mean_FR, std_FR = all_FR.mean(axis=1), all_FR.std(axis=1)
        lower, upper = mean_FR-std_FR, mean_FR+std_FR
        lower[lower < 0] = 0  # remove negative values
        ax.fill_between(tr_time, lower, upper, zorder=2,
                        alpha=.5, facecolor='grey', edgecolor='grey')
        plot.lines(tr_time, mean_FR, lw=2, color='k', ax=ax)

        # Add task start marker lines.
        prd_task_stats = OrdDict()
        for task, (n_unit, task_start, task_stop) in task_stats.iterrows():

            # Init.
            tr_idxs = (all_FR.index > task_start) & (all_FR.index <= task_stop)
            meanFR_tr = all_FR.loc[tr_idxs].mean(1)
            task_lbl = '{}\n{} units'.format(task, n_unit)

            # Add grand mean FR.
            meanFR = meanFR_tr.mean()
            task_lbl += '\nFR: {:.1f} sp/s'.format(meanFR)

            # Calculate linear trend to test gradual drift.
            t, fr = meanFR_tr.index, meanFR_tr
            slope, _, r_value, p_value, _ = sp.stats.linregress(t, fr)
            slope = 3600*slope  # convert to change in spike per hour
            pval = util.format_pvalue(p_value, max_digit=3)
            task_lbl += '\n$\delta$FR: {:.1f} sp/h'.format(slope)
            task_lbl += '\n{}'.format(pval)

            prd_task_stats[task_lbl] = task_start

        plot.plot_events(prd_task_stats, t_unit=s, alpha=1.0, color='black',
                         lw=1, linestyle='--', lbl_height=0.75, lbl_ha='left',
                         lbl_rotation=0, ax=ax)

        # Set limits and add labels to plot.
        plot.set_limits(xlim=[None, max(tr_time)], ax=ax)
        plot.set_labels(title=prd_name, xlab='Recording time (s)',
                        ylab=plot.FR_lbl, ytitle=1.1, ax=ax)

    # Format and save figure.
    title = 'Recording stability of ' + UA.Name
    plot.save_gsp_figure(fig, gsp, fname, title, rect_height=0.95)
