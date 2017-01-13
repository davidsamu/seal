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
from seal.analysis import direction, roc
from seal.plot import putil, ptuning, prate, pauc
from seal.util import util


# %% Functions to plot stimlus feature selectivity.

def plot_SR(u, feat=None, vals=None, prd_pars=None, nrate=None, colors=None,
            add_roc=True, add_prd_name=False, fig=None, sps=None, title=None,
            **kwargs):
    """Plot stimulus response (raster, rate and ROC) for mutliple stimuli."""

    if not u.to_plot():
        return

    # Set up stimulus parameters.
    if prd_pars is None:
        prd_pars = u.get_analysis_prds()
    mid_idx = int((len(prd_pars.index)/2.0))

    # Init subplots.
    sps, fig = putil.sps_fig(sps, fig)

    # Create a gridspec for each period.
    wratio = [float(dur) for dur in prd_pars.dur]
    wspace = 0.0  # 0.05 to separate periods
    gsp = putil.embed_gsp(sps, 1, len(prd_pars.index),
                          width_ratios=wratio, wspace=wspace)

    axes_raster, axes_rate, axes_roc = [], [], []
    for i, (prd, stim, ref, _, tcue, dur) in enumerate(prd_pars.itertuples()):

        # Prepare trial set.
        trs = (u.trials_by_features(stim, feat, vals) if feat is not None else
               u.ser_inc_trials())
        plot_roc = add_roc and len(trs) == 2

        # Init subplots.
        if plot_roc:
            rr_sps, roc_sps = putil.embed_gsp(gsp[i], 2, 1, hspace=0.3,
                                              height_ratios=[1, .5])
        else:
            rr_sps = putil.embed_gsp(gsp[i], 1, 1, hspace=0.3)[0]

        # Init params.
        evnts = [tcue] if tcue is not None else None
        if colors is None:
            colcyc = putil.get_colors()
            cols = [next(colcyc) for itrs in range(len(trs))]

        # Plot response on raster and rate plots.
        _, raster_axs, rate_ax = prate.plot_rr(u, prd, ref, evnts, nrate, trs,
                                               cols=cols, fig=fig,
                                               sps=rr_sps, **kwargs)

        # Add period name to rate plot.
        if add_prd_name:
            rate_ax.text(0.02, 0.95, prd, fontsize=10, color='k',
                         va='top', ha='left', transform=rate_ax.transAxes)

        # Add title to raster of middle period.
        if i == mid_idx and title is not None:
            raster_axs[0].set_title(title, ha='right')

        # Plot ROC curve.
        if plot_roc:
            roc_ax = fig.add_subplot(roc_sps)
            # Init rates.
            plot_params = prate.prep_rr_plot_params(u, prd, ref, nrate, trs)
            _, _, (rates1, rates2), stim_prd, _, _ = plot_params
            # Calculate ROC results.
            aroc = roc.run_ROC_over_time(rates1, rates2, n_perm=0)
            # Set up plot params and plot results.
            tvec, auc = aroc.index, aroc.auc
            xlim = rate_ax.get_xlim()
            pauc.plot_auc_over_time(auc, tvec, [stim_prd], evnts,
                                    xlim=xlim, ax=roc_ax)
            axes_roc.append(roc_ax)

        axes_raster.extend(raster_axs)
        axes_rate.append(rate_ax)

    # Format rate and roc plots to make them match across trial periods.

    # Remove y-axis label, spine and ticks from second and later periods.
    for ax in axes_rate[1:] + axes_roc[1:]:
        ax.set_ylabel('')
        putil.set_spines(ax, bottom=True, left=False)
        putil.hide_ticks(ax, show_x_ticks=True, show_y_ticks=False)

    # Remove (hide) legend from all but last rate plot.
    [ax.legend().set_visible(False) for ax in axes_rate[:-1]
     if ax.legend_ is not None]

    # Set common x label.
    if ('no_labels' not in kwargs) or (not kwargs['no_labels']):
        [ax.set_xlabel('') for ax in axes_rate + axes_roc]
        xlab = putil.t_lbl.format(prd_pars.stim[0] + ' onset')
        mid_ax = axes_rate[mid_idx] if not len(axes_roc) else axes_roc[mid_idx]
        mid_ax.set_xlabel(xlab, ha='right')

    # Reformat ticks on x axis. First period is reference.
    for axes in [axes_rate, axes_roc]:
        for ax, lbl_shift in zip(axes, prd_pars.lbl_shift):
            x1, x2 = ax.get_xlim()
            shift = float(lbl_shift)
            tmakrs, tlbls = putil.get_tick_marks_and_labels(x1+shift, x2+shift)
            putil.set_xtick_labels(ax, tmakrs-shift, tlbls)
    # Remove x tick labels from rate axes if roc's are present.
    if len(axes_roc):
        [ax.tick_params(labelbottom='off') for ax in axes_rate]

    # Match scale of rate and roc plots' y axes.
    for axes in [axes_rate, axes_roc]:
        putil.sync_axes(axes, sync_y=True)
        [putil.adjust_decorators(ax) for ax in axes]

    return axes_raster, axes_rate, axes_roc


def plot_LR(u, **kwargs):
    """Plot location response plot."""

    # Raster & rate in trials sorted by stimulus location.
    title = 'Location selectivity'
    res = plot_SR(u, 'Loc', title=title, **kwargs)
    return res


def plot_DR(u, **kwargs):
    """Plot direction response plot."""

    if u.DS.empty:
        u.test_DS()

    # Get DS parameters.
    stim = 'S1'
    pref_anti_dirs = [u.pref_dir(stim), u.anti_pref_dir(stim)]
    t1, t2 = u.DS.TW.loc[stim]

    # Raster & rate in trials sorted by stimulus direction.
    title = 'Direction selectivity'
    res = plot_SR(u, 'Dir', pref_anti_dirs, title=title, **kwargs)

    # Add event lines to interval used to test DS.
    # Slight hardcoding action going on below...
    evts = pd.DataFrame([[t1, 'DS start'], [t2, 'DS end']],
                        columns=['time', 'lbl'])
    for i in range(1, len(res)):
        ax = res[i][0]  # take first of rate and auc plots (showing S1)
        putil.plot_events(evts, add_names=False, color='grey', alpha=0.5,
                          ls='--', lw=1, ax=ax)

    return res


def plot_DSI(u, nrate=None, fig=None, sps=None, prd_pars=None,
             no_labels=False):
    """Plot direction selectivity indices."""

    # Init subplots.
    sps, fig = putil.sps_fig(sps, fig)
    gsp = putil.embed_gsp(sps, 2, 1, hspace=0.6)
    ax_mDSI, ax_wDSI = [fig.add_subplot(igsp) for igsp in gsp]

    # Calculate DSI using maximum - opposite and weighted measure.
    prd_pars = u.get_analysis_prds()
    maxDSI, wghtDSI = [direction.calc_DSI(u, fDSI, prd_pars, nrate)
                       for fDSI in [direction.max_DS, direction.weighted_DS]]

    # Get stimulus periods.
    stim_prds = u.get_stim_prds()

    # Plot DSIs.
    DSI_list = [pd.DataFrame(row).T for name, row in maxDSI.iterrows()]
    prate.rate(DSI_list, maxDSI.index, prds=stim_prds, pval=None, ylab='mDSI',
               title='max - opposite DSI', add_lgn=True, lgn_lbl=None,
               ax=ax_mDSI)

    DSI_list = [pd.DataFrame(row).T for name, row in wghtDSI.iterrows()]
    prate.rate(DSI_list, wghtDSI.index, prds=stim_prds, pval=None, ylab='wDSI',
               title='weighted DSI', add_lgn=True, lgn_lbl=None, ax=ax_wDSI)

    return ax_mDSI, ax_wDSI


def plot_task_relatedness(u, **kwargs):
    """Plot task-relatedness plot."""

    tr_str = '? '
    if 'TaskRelated' in u.QualityMetrics:
        tr_str = '' if u.QualityMetrics['TaskRelated'] else 'NOT '
    title = '{}: unit is {}task-related'.format(u.SessParams.task, tr_str)

    res = plot_SR(u, title=title, **kwargs)
    return res


# %% Functions to plot stimulus selectivity.

def plot_selectivity(u, nrate=None, fig=None, sps=None):
    """Plot selectivity summary plot."""

    if not u.to_plot():
        return

    # Init subplots.
    sps, fig = putil.sps_fig(sps, fig)
    tr_sps, lr_sps, dr_sps = putil.embed_gsp(sps, 3, 1, hspace=0.4)

    kwargs = {'nrate': nrate, 'fig': fig, 'no_labels': False}

    # Plot task-relatedness.
    plot_task_relatedness(u, sps=tr_sps, **kwargs)

    # Plot location-specific activity.
    _, lr_rate_axs, _ = plot_LR(u, sps=lr_sps, **kwargs)

    # Plot direction-specific activity.
    _, dr_rate_axs, _ = plot_DR(u, sps=dr_sps, **kwargs)

    return lr_rate_axs, dr_rate_axs


def plot_DR_3x3(u, nrate=None, fig=None, sps=None):
    """Plot 3x3 direction response plot, with polar plot in center."""

    if not u.to_plot():
        return

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
        res = plot_SR(u, feat='Dir', vals=[d], nrate=nrate, fig=fig,
                      sps=gsp[isp], no_labels=True)
        draster_axs, drate_axs, _ = res

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
    [putil.adjust_decorators(ax) for ax in rate_axs]

    return ax_polar, rate_axs
