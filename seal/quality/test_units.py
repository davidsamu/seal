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

def direction_response_test(UA, tasks=None, nrate=None, ftempl=None,
                            match_scale=True):
    """Plot responses to 8 directions and polar plot in the center."""

    # Init data.
    if tasks is None:
        tasks = UA.tasks()
    uids = UA.uids()
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
        gsp_u_list = zip(gsp, UA.iter_thru(tasks, [uid], ret_empty=True,
                                           ret_excl=True))
        for itask, (sps, u) in enumerate(gsp_u_list):

            task_gsp = plot.embed_gsp(sps, 3, 3)

            for i, u_gsp in enumerate(task_gsp):
                # Polar plot.
                if i == 4:
                    ax_polar = fig.add_subplot(u_gsp, polar=True)
                    if u.is_empty():
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

                        if u.is_empty():
                            rr_res = plot.empty_raster_rate(fig, rr_gsp, 1)
                            raster_axs, rate_ax = rr_res
                        # Plot raster&rate plot.
                        else:
                            dd_trs = u.trials_by_param_values(stim + 'Dir',
                                                              [dir_order[i]])
                            nrate = u.init_nrate(nrate)
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
                            title = tasks[itask]
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


# TODO: update, simplify, refactor, and plot one figure per unit over tasks.
def within_trial_unit_test(UA, nrate, fname, plot_info=True,
                           plot_rr=True, plot_ds=True, plot_dd_rr=True):
    """Test unit responses within trails."""

    def n_plots(plot_names):
        return sum([nplots.nrow[pn] * nplots.ncol[pn] for pn in plot_names])

    # Global plotting params.
    row_data = [('info', (1 if plot_info else 0, 1)),
                ('rr', (2 if plot_rr else 0, 1)),
                ('ds', (1, 2 if plot_ds else 0)),
                ('dd_rr', (3 if plot_dd_rr else 0, 2))]
    nplots = util.make_df(row_data, ('nrow', 'ncol'))
    n_unit_subplots = (plot_info + plot_rr + plot_ds + plot_dd_rr)
    n_unit_plots_total = n_plots(nplots.index)

    # Init S1 and S2 queries.
    stim_df = constants.ext_stim_prds.periods()
    stim_df['dir'] = ['S1Dir', 'S2Dir']
    nstim = len(stim_df.index)
    rr_t1 = stim_df.start.min()
    rr_t2 = stim_df.end.max()

    # Init plotting.
    # TODO: split this by unit!!!!!  And by task???
    tasks = UA.tasks()
    uids = UA.uids()
    ntask = len(tasks)
    nchunit = len(uids)
    Unit_list = UA.unit_list(tasks, uids, return_empty=True)
    fig, gsp, _ = plot.get_gs_subplots(nrow=nchunit, ncol=ntask,
                                       subw=4.5, subh=7, create_axes=False)
    legend_kwargs = {'borderaxespad': 0}

    # Plot within-trial activity of each unit during each task.
    for unit_sps, u in zip(gsp, Unit_list):

        # Container of axes for given unit.
        unit_gsp = plot.embed_gsp(unit_sps, n_unit_subplots, 1)

        irow = 0  # to keep track of row index

        # Plot unit's info header.
        if plot_info:
            mock_gsp_info = plot.embed_gsp(unit_gsp[irow, 0], 1, 1)
            irow += 1
            if u.is_empty():  # add mock subplot
                plot.add_mock_axes(fig, mock_gsp_info[0, 0])
            else:
                ax = fig.add_subplot(mock_gsp_info[0, 0])
                plot.unit_info(u, ax=ax)

        # Plot raster - rate plot.
        if plot_rr:
            rr_gsp = plot.embed_gsp(unit_gsp[irow, 0], 2, 1)
            irow += 1
            if u.is_empty():  # add mock subplot
                plot.empty_raster_rate(fig, rr_gsp, 1)
            else:
                res = u.plot_raster_rate(nrate, no_labels=True, t1=rr_t1,
                                         t2=rr_t2, legend_kwargs=legend_kwargs,
                                         fig=fig, outer_gsp=rr_gsp)
                fig, raster_axs, rate_ax = res
                plot.replace_tr_num_with_tr_name(raster_axs[0], 'all trials')

        # Plot direction selectivity plot.
        if plot_ds:
            ds_gsp = plot.embed_gsp(unit_gsp[irow, 0], 1, 2)
            irow += 1
            if u.is_empty():  # add mock subplot
                plot.empty_direction_selectivity(fig, ds_gsp)
            else:
                u.test_DS(no_labels=True, fig=fig, outer_gsp=ds_gsp)

        # Plot direction selectivity raster - rate plot.
        if plot_dd_rr:
            outer_dd_rr_gsp = plot.embed_gsp(unit_gsp[irow, 0], 1, nstim)
            irow += 1

            for i, (stim, row) in enumerate(stim_df.iterrows()):
                dd_rr_gsp = plot.embed_gsp(outer_dd_rr_gsp[0, i], 2, 1)
                if u.is_empty():  # add mock subplot
                    plot.empty_raster_rate(fig, dd_rr_gsp, 2)
                else:
                    dd_trs = u.dir_pref_anti_trials(stim=stim,
                                                    pname=[row.dir],
                                                    comb_values=True)
                    res = u.plot_raster_rate(nrate, trs=dd_trs,
                                             t1=row.start, t2=row.end,
                                             pvals=[0.05], test='t-test',
                                             legend_kwargs=legend_kwargs,
                                             no_labels=True, fig=fig,
                                             outer_gsp=dd_rr_gsp)
                    fig, raster_axs, rate_ax = res

                    # Replace y-axis tickmarks with trial set names.
                    for ax, trs in zip(raster_axs, dd_trs):
                        plot.replace_tr_num_with_tr_name(ax, trs.name)

                    # Hide y-axis tickmarks on second and subsequent rate axes.
                    if i > 0:
                        plot.hide_axes(show_x=True, show_y=False, ax=rate_ax)

    # Match y-axis scales across tasks.
    # List of axes offset lists to match y limit across.
    # Each value indexes a plot within the unit's plot block.
    yplot_idx = (plot_rr * [[n_plots(['info'])+1]] +
                 plot_ds*[[n_plots(['info', 'rr'])],
                          [n_plots(['info', 'rr'])+1]] +
                 plot_dd_rr * [[n_plots(['info', 'rr', 'ds'])+2,
                                n_plots(['info', 'rr', 'ds'])+5]])
    move_sign_lines = (False, False, False, True)
    for offsets, mv_sg_ln in zip(yplot_idx, move_sign_lines):
        for irow in range(nchunit):
            axs = [fig.axes[n_unit_plots_total*ntask*irow +
                            itask*n_unit_plots_total + offset]
                   for offset in offsets
                   for itask in range(ntask)
                   if not Unit_list[irow*ntask + itask].is_empty()]
            plot.sync_axes(axs, sync_y=True)
            if mv_sg_ln:
                [plot.move_significance_lines(ax) for ax in axs]

    # Add unit names to beginning of each row.
    if not plot_info:
        ylab_kwargs = {'rotation': 0, 'size': 'xx-large', 'ha': 'right'}
        offset = 0  # n_plots(['info', 'rr'])
        for irow, unit_id in enumerate(uids):
            ax = fig.axes[n_unit_plots_total*ntask*irow+offset]
            unit_name = 'ch {} / {}'.format(unit_id[1], unit_id[2]) + 15*' '
            plot.set_labels(ylab=unit_name, ax=ax, ylab_kwargs=ylab_kwargs)

    # Add task names to top of each column.
    if not plot_info:
        title_kwargs = {'size': 'xx-large'}
        for icol, task in enumerate(tasks):
            ax = fig.axes[n_unit_plots_total*icol]
            plot.set_labels(title=task, ax=ax, ytitle=1.30,
                            title_kwargs=title_kwargs)

    # Format and save figure.
    title = 'Within trial activity of ' + UA.Name
    plot.save_gsp_figure(fig, gsp, fname, title, rect_height=0.95)


# TODO: check task order! UA order is not recording order!
def check_recording_stability(UA, fname):
    """Check stability of recording session across tasks."""

    # Init params.
    periods = constants.tr_prds

    # Init figure.
    fig, gsp, ax_list = plot.get_gs_subplots(nrow=periods.index.size, ncol=1,
                                             subw=10, subh=2.5, as_array=False)

    # Init task info dict.
    tasks = UA.tasks()
    task_stats = OrdDict()
    for task in tasks:
        unit_list = UA.unit_list(tasks=[task])
        task_start = unit_list[0].TrialParams['TrialStart'].iloc[0]
        task_stop = unit_list[0].TrialParams['TrialStop'].iloc[-1]
        task_stats[task] = (len(unit_list), task_start, task_stop)

    unit_ids = UA.rec_ch_unit_indices()
    for (prd_name, (t1, t2)), ax in zip(periods.iterrows(), ax_list):
        # Calculate and plot firing rate during given period within each trial
        # across session for all units.
        all_FR_prd = []
        for unit_id in unit_ids:

            # Get all units (including empty ones for color cycle consistency).
            unit_list = UA.unit_list(ch_unit_idxs=[unit_id],
                                     return_empty=True)
            FR_tr_list = [u.get_rates_by_trial(t1=t1, t2=t2)
                          for u in unit_list]

            # Plot each FRs per task discontinuously.
            colors = plot.get_colors()
            for FR_tr in FR_tr_list:
                color = next(colors)
                if FR_tr is None:  # for color consistency
                    continue
                # For (across-task) continuous plot,
                # need to concatenate across tasks first (see below).
                plot.lines(FR_tr.index, FR_tr, ax=ax, zorder=1,
                           alpha=0.5, color=color)

            # Save FRs for summary plots and stats.
            FR_tr = pd.concat([FR_tr for FR_tr in FR_tr_list])
            all_FR_prd.append(FR_tr)

        # Add mean +- std FR.
        all_FR = pd.concat(all_FR_prd, axis=1)
        tr_time = all_FR.index
        mean_FR = all_FR.mean(axis=1)
        std_FR = all_FR.std(axis=1)
        lower, upper = mean_FR-std_FR, mean_FR+std_FR
        lower[lower < 0] = 0
        ax.fill_between(tr_time, lower, upper, zorder=2,
                        alpha=.75, facecolor='grey', edgecolor='grey')
        plot.lines(tr_time, mean_FR, ax=ax, lw=2, color='k')

        # Add task start marker lines.
        prd_task_stats = OrdDict()
        for task, (n_unit, task_start, task_stop) in task_stats.items():

            # Init.
            tr_idxs = (all_FR.index > task_start) & (all_FR.index <= task_stop)
            meanFR_tr = all_FR.loc[tr_idxs].mean(1)
            task_lbl = '{}\nn = {} units'.format(task, n_unit)

            # Add grand mean FR.
            meanFR = meanFR_tr.mean()
            task_lbl += '\nmean FR = {:.1f} sp/s'.format(meanFR)

            # Calculate linear trend to test gradual drift.
            t, fr = meanFR_tr.index, meanFR_tr
            slope, _, r_value, p_value, _ = sp.stats.linregress(t, fr)
            slope = 3600*slope  # convert to change in spike per hour
            pval = util.format_pvalue(p_value, max_digit=3)
            task_lbl += '\n$\delta$FR = {:.1f} sp/hour ({})'.format(slope,
                                                                    pval)

            prd_task_stats[task_lbl] = task_start

        plot.plot_events(prd_task_stats, t_unit=s, alpha=1.0, color='black',
                         lw=1, linestyle='--', lbl_height=0.75, lbl_ha='left',
                         lbl_rotation=0, ax=ax)

        # Set limits and add labels to plot.
        plot.set_limits(xlim=[None, max(tr_time)], ax=ax)
        plot.set_labels(title=prd_name, xlab='Recording time (s)',
                        ylab=plot.FR_lbl, ax=ax)

    # Format and save figure.
    title = 'Recording stability of ' + UA.Name
    plot.save_gsp_figure(fig, gsp, fname, title, rect_height=0.92)
