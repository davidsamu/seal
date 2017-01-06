#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to plot combined (composite) plots..

@author: David Samu
"""

import numpy as np
import pandas as pd

from quantities import deg

from seal.object import constants
from seal.plot import putil, ptuning, prate
from seal.util import util


# %% Utility functions.

def get_stim_pars_to_plot(u):
    """Get stimulus parameters to plot selectivity."""

    stims = pd.DataFrame(index=constants.stim_dur.index)
    stims['prd'] = ['around ' + stim for stim in stims.index]
    stims['dur'] = [u.pr_dur(stims.loc[stim, 'prd']) for stim in stims.index]

    return stims


# %% Functions to plot stimlus feature selectivity.

def plot_SR(u, feat=None, vals=None, stims=None, nrate=None, colors=None,
            add_stim_name=False, fig=None, sps=None, title=None, **kwargs):
    """Plot stimulus response (raster and rate) for mutliple stimuli."""

    if not u.to_plot():
        return

    # Set up stimulus parameters.
    if stims is None:
        stims = get_stim_pars_to_plot(u)

    # Init subplots.
    sps, fig = putil.sps_fig(sps, fig)

    # Create a gridspec for each stimulus.
    wratio = [float(dur) for dur in stims.dur]
    stim_rr_gsp = putil.embed_gsp(sps, 1, len(stims.index),
                                  width_ratios=wratio, wspace=0.1)

    axes_raster, axes_rate = [], []
    for i, stim in enumerate(stims.index):

        rr_sps = stim_rr_gsp[i]

        # Prepare trial set.
        if feat is not None:
            trs = u.trials_by_features(stim, feat, vals)
        else:
            trs = u.ser_inc_trials()

        # Init params.
        if colors is None:
            colcyc = putil.get_colors()
            cols = [next(colcyc) for i in range(len(trs))]

        # Plot response on raster and rate plots.
        prd, ref = stims.loc[stim, 'prd'], stim + ' on'
        _, raster_axs, rate_ax = prate.plot_rr(u, prd, ref, nrate, trs,
                                               cols=cols, fig=fig,
                                               sps=rr_sps, **kwargs)

        # Add stimulus name to rate plot.
        if add_stim_name:
            rate_ax.text(0.02, 0.95, stim, fontsize=10, color='k',
                         va='top', ha='left', transform=rate_ax.transAxes)

        # Add title to first plot.
        if i == 0:
            putil.set_labels(raster_axs[0], title=title)

        # Remove y-axis label from second and later rate plots.
        if i > 0:
            rate_ax.set_ylabel('')

        axes_raster.extend(raster_axs)
        axes_rate.append(rate_ax)

    # Format rate plots.
    for ax in axes_rate[1:]:  # second and later stimuli
        putil.set_spines(ax, bottom=True, left=False)
        putil.hide_ticks(ax, show_x_ticks=True, show_y_ticks=False)

    # Match scale of y axes.
    putil.sync_axes(axes_rate, sync_y=True)
    [putil.move_signif_lines(ax) for ax in axes_rate]

    return axes_raster, axes_rate


def plot_LR(u, **kwargs):
    """Plot location response plot."""

    # Raster & rate in trials sorted by stimulus location.
    title = 'Location selectivity'
    res = plot_SR(u, 'Loc', title=title, **kwargs)
    return res


def plot_DR(u, **kwargs):
    """Plot direction response plot."""

    # Raster & rate in trials sorted by stimulus direction.
    pref_anti_dirs = [u.pref_dir(), u.anti_pref_dir()]
    title = 'Direction selectivity'
    res = plot_SR(u, 'Dir', pref_anti_dirs, title=title, **kwargs)
    return res


def plot_task_relatedness(u, **kwargs):
    """Plot task-relatedness plot."""

    tr_str = '?'
    if 'TaskRelated' in u.QualityMetrics:
        tr_str = ' ' if u.QualityMetrics['TaskRelated'] else 'NOT'
    title = 'Unit is {} task-related'.format(tr_str)

    res = plot_SR(u, title=title, **kwargs)
    return res


# %% Functions to plot stimulus selectivity.

def plot_DR_3x3(u, nrate=None, fig=None, sps=None):
    """Plot 3x3 direction response plot, with polar plot in center."""

    if not u.to_plot():
        return

    # Set up stimulus parameters.
    stims = get_stim_pars_to_plot(u)

    # Init subplots.
    sps, fig = putil.sps_fig(sps, fig)
    gsp = putil.embed_gsp(sps, 3, 3)  # inner gsp with subplots

    # Polar plot.
    putil.set_style('notebook', 'white')
    ax_polar = fig.add_subplot(gsp[4], polar=True)
    for stim in stims.index:  # for each stimuli
        stim_resp = u.get_stim_resp_vals(stim, 'Dir')
        resp_stats = util.calc_stim_resp_stats(stim_resp)
        dirs, resp = np.array(resp_stats.index) * deg, resp_stats['mean']
        c = putil.stim_colors[stim]
        baseline = u.get_baseline()
        ptuning.plot_DR(dirs, resp, color=c, baseline=baseline, ax=ax_polar)
    putil.hide_ticks(ax_polar, 'y')

    # Raster-rate plots.
    putil.set_style('notebook', 'ticks')
    rr_pos = [5, 2, 1, 0, 3, 6, 7, 8]  # Position of each direction.
    rr_dir_pos = pd.Series(constants.all_dirs, index=rr_pos)

    rate_axs = []
    for isp, d in rr_dir_pos.iteritems():

        # Prepare plot formatting.
        first_dir = (isp == 0)

        # Plot stimulus response across stimuli.
        res = plot_SR(u, feat='Dir', vals=[d], stims=stims, nrate=nrate,
                      add_stim_name=first_dir, fig=fig, sps=gsp[isp],
                      no_labels=True)
        draster_axs, drate_axs = res

        # Remove axis ticks.
        for i, ax in enumerate(drate_axs):
            first_stim = (i == 0)
            show_x_tick_lbls = first_dir
            show_y_tick_lbls = first_dir & first_stim
            putil.hide_tick_labels(ax, show_x_tick_lbls, show_y_tick_lbls)

        # Add task name as title (to top center axes).
        if isp == 1:
            putil.set_labels(draster_axs[0], title=u.get_task(),
                             ytitle=1.10, title_kws={'loc': 'right'})

        rate_axs.extend(drate_axs)

    # Match scale of y axes.
    putil.sync_axes(rate_axs, sync_y=True)

    return ax_polar, rate_axs


def plot_selectivity(u, nrate=None, fig=None, sps=None):
    """Plot selectivity summary plot."""

    if not u.to_plot():
        return

    # Set up stimulus parameters.
    stims = get_stim_pars_to_plot(u)

    # Init subplots.
    sps, fig = putil.sps_fig(sps, fig)
    tr_sps, ls_sps, ds_sps = putil.embed_gsp(sps, 3, 1)

    kwargs = {'stims': stims, 'nrate': nrate, 'fig': fig, 'no_labels': False}

    # Plot task-relatedness.
    plot_task_relatedness(u, sps=tr_sps, **kwargs)

    # Plot location selectivity.
    _, ls_rate_axs = plot_LR(u, sps=ls_sps, **kwargs)

    # Plot direction selectivity.
    _, ds_rate_axs = plot_DR(u, sps=ds_sps, **kwargs)

    return ls_rate_axs, ds_rate_axs
