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

def init_prd_pars_to_plot(u):
    """Get parameters of periods to plot selectivity for."""

    rr_prds = constants.rr_prds.copy()
    rr_prds['dur'] = [u.pr_dur(prd) for prd in rr_prds.index]

    return rr_prds


# %% Functions to plot stimlus feature selectivity.

def plot_SR(u, feat=None, vals=None, prd_pars=None, nrate=None, colors=None,
            add_prd_name=False, fig=None, sps=None, title=None, **kwargs):
    """Plot stimulus response (raster and rate) for mutliple stimuli."""

    if not u.to_plot():
        return

    # Set up stimulus parameters.
    if prd_pars is None:
        prd_pars = init_prd_pars_to_plot(u)
    mid_idx = int((len(prd_pars.index)/2.0))

    # Init subplots.
    sps, fig = putil.sps_fig(sps, fig)

    # Create a gridspec for each period.
    wratio = [float(dur) for dur in prd_pars.dur]
    wspace = 0.0  # 0.05 to separate periods
    gsp = putil.embed_gsp(sps, 1, len(prd_pars.index),
                          width_ratios=wratio, wspace=wspace)

    axes_raster, axes_rate = [], []
    for i, (prd, stim, ref, _, dur) in enumerate(prd_pars.itertuples()):

        # Prepare trial set.
        if feat is not None:
            trs = u.trials_by_features(stim, feat, vals)
        else:
            trs = u.ser_inc_trials()

        # Init params.
        if colors is None:
            colcyc = putil.get_colors()
            cols = [next(colcyc) for itrs in range(len(trs))]

        # Plot response on raster and rate plots.
        _, raster_axs, rate_ax = prate.plot_rr(u, prd, ref, nrate, trs,
                                               cols=cols, fig=fig,
                                               sps=gsp[i], **kwargs)

        # Add period name to rate plot.
        if add_prd_name:
            rate_ax.text(0.02, 0.95, prd, fontsize=10, color='k',
                         va='top', ha='left', transform=rate_ax.transAxes)

        # Add title to to raster of middle period.
        if i == mid_idx and title is not None:
            raster_axs[0].set_title(title, ha='right')

        axes_raster.extend(raster_axs)
        axes_rate.append(rate_ax)

    # Format rate plots.

    # Remove y-axis label, spine and ticks from second and later periods.
    for ax in axes_rate[1:]:
        ax.set_ylabel('')
        putil.set_spines(ax, bottom=True, left=False)
        putil.hide_ticks(ax, show_x_ticks=True, show_y_ticks=False)

    # Remove (hide) legend from all but last rate plot.
    [ax.legend().set_visible(False) for ax in axes_rate[:-1]
     if ax.legend() is not None]

    # Set common x label.
    if ('no_labels' not in kwargs) or (not kwargs['no_labels']):
        [ax.set_xlabel('') for ax in axes_rate]
        xlab = putil.t_lbl.format(prd_pars.stim[0] + ' onset')
        axes_rate[mid_idx].set_xlabel(xlab, ha='right')

    # Relabel x axis. First period is reference.
    for ax, lbl_shift in zip(axes_rate, prd_pars.lbl_shift):
        x1, x2 = ax.get_xlim()
        shift = float(lbl_shift)
        tmakrs, tlbls = putil.get_tick_marks_and_labels(x1+shift, x2+shift)
        putil.set_xtick_labels(ax, tmakrs-shift, tlbls)

    # Match scale of rate plots' y axes.
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

    tr_str = '? '
    if 'TaskRelated' in u.QualityMetrics:
        tr_str = '' if u.QualityMetrics['TaskRelated'] else 'NOT '
    title = 'Unit is {}task-related'.format(tr_str)

    res = plot_SR(u, title=title, **kwargs)
    return res


# %% Functions to plot stimulus selectivity.

def plot_DR_3x3(u, nrate=None, fig=None, sps=None):
    """Plot 3x3 direction response plot, with polar plot in center."""

    if not u.to_plot():
        return

    # Set up period parameters.
    prd_pars = init_prd_pars_to_plot(u)

    # Init subplots.
    sps, fig = putil.sps_fig(sps, fig)
    gsp = putil.embed_gsp(sps, 3, 3)  # inner gsp with subplots

    # Polar plot.
    putil.set_style('notebook', 'white')
    ax_polar = fig.add_subplot(gsp[4], polar=True)
    for stim in constants.stim_dur.index:  # for each stimuli
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
    rr_dir_plot_pos = pd.Series(constants.all_dirs, index=rr_pos)

    rate_axs = []
    for isp, d in rr_dir_plot_pos.iteritems():

        # Prepare plot formatting.
        first_dir = (isp == 0)

        # Plot direction response across trial periods.
        res = plot_SR(u, feat='Dir', vals=[d], prd_pars=prd_pars, nrate=nrate,
                      fig=fig, sps=gsp[isp], no_labels=True)
        draster_axs, drate_axs = res

        # Remove axis ticks.
        for i, ax in enumerate(drate_axs):
            first_prd = (i == 0)
            show_x_tick_lbls = first_dir
            show_y_tick_lbls = first_dir & first_prd
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

    # Set up period parameters.
    prd_pars = init_prd_pars_to_plot(u)

    # Init subplots.
    sps, fig = putil.sps_fig(sps, fig)
    tr_sps, ls_sps, ds_sps = putil.embed_gsp(sps, 3, 1, hspace=0.4)

    kwargs = {'prd_pars': prd_pars, 'nrate': nrate, 'fig': fig,
              'no_labels': False}

    # Plot task-relatedness.
    plot_task_relatedness(u, sps=tr_sps, **kwargs)

    # Plot location selectivity.
    _, ls_rate_axs = plot_LR(u, sps=ls_sps, **kwargs)

    # Plot direction selectivity.
    _, ds_rate_axs = plot_DR(u, sps=ds_sps, **kwargs)

    return ls_rate_axs, ds_rate_axs
