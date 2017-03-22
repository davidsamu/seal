#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to plot combined (composite) plots.

@author: David Samu
"""

import numpy as np
import pandas as pd

from quantities import deg, ms

from seal.analysis import direction
from seal.plot import putil, ptuning, prate, pauc
from seal.roc import roccore
from seal.util import util, constants


# %% Functions to plot stimulus feature selectivity.

def plot_SR(u, param=None, vals=None, from_trs=None, prd_pars=None, nrate=None,
            colors=None, add_roc=False, add_prd_name=False, fig=None, sps=None,
            title=None, **kwargs):
    """Plot stimulus response (raster, rate and ROC) for mutliple stimuli."""

    if not u.to_plot():
        return

    # Set up stimulus parameters.
    if prd_pars is None:
        prd_pars = u.get_analysis_prds()

    # Init subplots.
    sps, fig = putil.sps_fig(sps, fig)

    # Create a gridspec for each period.
    wratio = [float(min(dur, lmax)) for dur, lmax in zip(prd_pars.dur,
                                                         prd_pars.max_len)]
    wspace = 0.0  # 0.05 to separate periods
    gsp = putil.embed_gsp(sps, 1, len(prd_pars.index),
                          width_ratios=wratio, wspace=wspace)

    axes_raster, axes_rate, axes_roc = [], [], []

    for i, prd in enumerate(prd_pars.index):

        ppars = prd_pars.loc[prd]
        ref = ppars.ref

        # Prepare trial set.
        if param is None:
            trs = u.ser_inc_trials()
        elif param in u.TrData.columns:
            trs = u.trials_by_param(param, vals)
        else:
            trs = u.trials_by_param((ppars.stim, param), vals)

        if from_trs is not None:
            trs = util.filter_lists(trs, from_trs)

        plot_roc = add_roc and (len(trs) == 2)

        # Init subplots.
        if plot_roc:
            rr_sps, roc_sps = putil.embed_gsp(gsp[i], 2, 1, hspace=0.2,
                                              height_ratios=[1, .4])
        else:
            rr_sps = putil.embed_gsp(gsp[i], 1, 1, hspace=0.3)[0]

        # Init params.
        prds = [constants.ev_stims.loc[ref]]
        evnts = None
        if ('cue' in ppars) and (ppars.cue is not None):
            evnts = [{'time': ppars.cue}]
            evnts[0]['color'] = (ppars.cue_color if 'cue_color' in ppars.index
                                 else putil.cue_colors['all'])

        if colors is None:
            colcyc = putil.get_colors()
            colors = [next(colcyc) for itrs in range(len(trs))]

        # Plot response on raster and rate plots.
        _, raster_axs, rate_ax = prate.plot_rr(u, prd, ref, prds, evnts, nrate,
                                               trs, ppars.max_len, cols=colors,
                                               fig=fig, sps=rr_sps, **kwargs)

        # Add period name to rate plot.
        if add_prd_name:
            rate_ax.text(0.02, 0.95, prd, fontsize=10, color='k',
                         va='top', ha='left', transform=rate_ax.transAxes)

        # Plot ROC curve.
        if plot_roc:
            roc_ax = fig.add_subplot(roc_sps)
            # Init rates.
            plot_params = prate.prep_rr_plot_params(u, prd, ref, nrate, trs)
            _, _, (rates1, rates2), _, _ = plot_params
            # Calculate ROC results.
            aroc = roccore.run_ROC_over_time(rates1, rates2, n_perm=0)
            # Set up plot params and plot results.
            tvec, auc = aroc.index, aroc.auc
            xlim = rate_ax.get_xlim()
            pauc.plot_auc_over_time(auc, tvec, prds, evnts,
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
    [putil.hide_legend(ax) for ax in axes_rate[:-1]]

    # Get middle x point in relative coordinates of first rate axes.
    xranges = np.array([ax.get_xlim() for ax in axes_rate])
    xlens = xranges[:, 1] - xranges[:, 0]
    xmid = xlens.sum() / xlens[0] / 2

    # Add title.
    if title is not None:
        axes_raster[0].set_title(title, x=xmid, y=1.0)

    # Set common x label.
    if ('no_labels' not in kwargs) or (not kwargs['no_labels']):
        [ax.set_xlabel('') for ax in axes_rate + axes_roc]
        xlab = putil.t_lbl.format(prd_pars.stim[0] + ' onset')
        ax = axes_roc[0] if len(axes_roc) else axes_rate[0]
        ax.set_xlabel(xlab, x=xmid)

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


def plot_SR_matrix(u, param, vals=None, sps=None, fig=None):
    """
    Plot stimulus response in matrix layout by target and delay length for
    combined task.

    This function is currently out of date, needs updating!
    """

    # Init params.
    dsplit_prd_pars = constants.tr_half_prds.copy()
    dcomb_prd_pars = constants.tr_third_prds.copy()
    targets = u.TrData['ToReport'].unique()
    dlens = util.remove_dim_from_array(constants.del_lens)

    # Init axes.
    nrow = len(targets) + (len(targets) > 1)
    ncol = len(dlens) + (len(dlens) > 1)
    # Width ratio, depending in delay lengths.
    wratio = None
    if len(dlens) > 1:
        len_wo_delay = float(dsplit_prd_pars.max_len.sum()) - dlens[-1]
        wratio = [len_wo_delay+dlens[0]]  # combined (unsplit) column
        wratio.extend([len_wo_delay + dlen for dlen in dlens])
    gsp = putil.embed_gsp(sps, nrow, ncol, width_ratios=wratio,
                          wspace=0.2, hspace=0.4)
    raster_axs, rate_axs, roc_axs = [], [], []

    # Set up trial splits to plot by.
    mtargets = ['all'] + list(targets if len(targets) > 1 else [])
    mdlens = ['all'] + list(dlens if len(dlens) > 1 else [])
    # No split (all included trials).
    trs_splits = pd.Series([u.inc_trials()], index=[('all', 'all')])
    # Split by report only.
    if len(targets) > 1:
        by_report = u.trials_by_params(['ToReport'])
        by_report.index = [(r, 'all') for r in by_report.index]
        trs_splits = trs_splits.append(by_report)
    # Split by delay length only.
    if len(dlens) > 1:
        by_dlen = u.trials_by_params(['DelayLen'])
        by_dlen.index = [('all', dl) for dl in by_dlen.index]
        trs_splits = trs_splits.append(by_dlen)
    # Split by both report and delay length.
    if len(targets) > 1 and len(dlens) > 1:
        target_dlen_trs = u.trials_by_params(['ToReport', 'DelayLen'])
        trs_splits = trs_splits.append(target_dlen_trs)

    # Plot each split on matrix layout.
    for i, target in enumerate(mtargets):
        for j, dlen in enumerate(mdlens):

            # Init plotting params.
            dlen_str = str(int(dlen)) + ' ms' if util.is_number(dlen) else dlen
            title = 'report: {}    |    delay: {}'.format(target, dlen_str)
            from_trs = trs_splits[(target, dlen)]

            # Periods to plot.
            if dlen == 'all':
                prd_pars = dcomb_prd_pars.copy()
                prd_pars = u.get_analysis_prds(prd_pars, from_trs)
            else:
                prd_pars = dsplit_prd_pars.copy()
                prd_pars = u.get_analysis_prds(prd_pars, from_trs)
                S2_shift = constants.stim_dur['S1'] + dlen*ms
                prd_pars.lbl_shift['S2 half'] = S2_shift
                if len(dlens) > 1:
                    tdiff = (dlen - dlens[-1]) * ms
                    S1_max_len = prd_pars.max_len['S1 half'] + tdiff
                    prd_pars.max_len['S1 half'] = S1_max_len
            if 'cue' in prd_pars.columns:
                prd_pars['cue_color'] = putil.cue_colors[target]

            # Plot response.
            res = plot_SR(u, param, vals, from_trs, prd_pars,
                          title=title, sps=gsp[i, j], fig=fig)
            axes_raster, axes_rate, axes_roc = res

            # Remove superfluous labels.
            if i < len(mtargets)-1:
                [ax.set_xlabel('') for ax in axes_rate]
            if j > 0:
                [ax.set_ylabel('') for ax in axes_rate]

            # Collect axes.
            raster_axs.extend(axes_raster)
            rate_axs.extend(axes_rate)
            roc_axs.extend(axes_roc)

    return raster_axs, rate_axs, roc_axs


def plot_LR(u, sps, fig):
    """Plot location response plot."""

    res = plot_SR(u, 'Loc', title='Location selectivity', sps=sps, fig=fig)

    return res


def plot_DR(u, sps, fig):
    """Plot direction response plot."""

    # Get DS parameters.
    if u.DS.empty:
        u.test_DS()
    stim = 'S2'  # testing S2, because S1 location can change btw tasks
    pref_anti_dirs = [u.pref_dir(stim, 'max'), u.anti_pref_dir(stim, 'max')]
    t1, t2 = u.DS.TW.loc[stim]
    evts = pd.DataFrame([[t1, 'DS start'], [t2, 'DS end']],
                        columns=['time', 'lbl'])

    res = plot_SR(u, 'Dir', pref_anti_dirs, title='Direction selectivity',
                  sps=sps, fig=fig)

    # Add event lines to interval used to test DS.
    # Slight hardcoding action going on below...
    # On first period plots of rate and auc axes (showing S1).
    if u.SessParams.region != 'MT':
        for ax in [res[i][0] for i in (1, 2) if res[i] not in (None, [])]:
            putil.plot_events(evts, add_names=False, color='grey',
                              alpha=0.5, ls='--', lw=1, ax=ax)

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


def plot_all_trials(u, sps, fig):
    """Plot task-relatedness plot."""

    title = putil.get_unit_info_title(u, fullname=False)
    res = plot_SR(u, sps=sps, fig=fig, title=title)

    return res


# %% Functions to plot stimulus selectivity.

def plot_selectivity(u, fig=None, sps=None):
    """Plot selectivity summary plot."""

    if not u.to_plot():
        return

    # Check which stimulus parameters were variable. Don't plot selectivity for
    # variables that were kept constant.
    stims = ('S1', 'S2')
    # Location.
    s1locs, s2locs = [u.TrData[(stim, 'Loc')].unique() for stim in stims]
    plot_lr = False if (len(s1locs) == 1) and (len(s2locs) == 1) else True
    # Direction.
    s1dirs, s2dirs = [u.TrData[(stim, 'Dir')].unique() for stim in stims]
    plot_dr = False if (len(s1dirs) == 1) and (len(s2dirs) == 1) else True

    # Init subplots.
    nsps = 1 + plot_lr + plot_dr  # all trials + location sel + direction sel
    sps, fig = putil.sps_fig(sps, fig)
    gsp = putil.embed_gsp(sps, nsps, 1, hspace=0.3)

    # Plot task-relatedness.
    plot_all_trials(u, gsp[0], fig)
    igsp = 1

    # Plot location-specific activity.
    if plot_lr:
        _, lr_rate_axs, _ = plot_LR(u, gsp[1], fig)
        igsp = igsp + 1
    else:
        lr_rate_axs = []

    # Plot direction-specific activity.
    if plot_dr:
        _, dr_rate_axs, _ = plot_DR(u, gsp[igsp], fig)
    else:
        dr_rate_axs = []

    return lr_rate_axs, dr_rate_axs


def plot_DR_3x3(u, fig=None, sps=None):
    """Plot 3x3 direction response plot, with polar plot in center."""

    if not u.to_plot():
        return

    # Init subplots.
    sps, fig = putil.sps_fig(sps, fig)
    gsp = putil.embed_gsp(sps, 3, 3)  # inner gsp with subplots

    # Polar plot.
    putil.set_style('notebook', 'white')
    ax_polar = fig.add_subplot(gsp[4], polar=True)
    for stim in constants.stim_dur.index:  # for each stimulus
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
        res = plot_SR(u, 'Dir', [d], fig=fig, sps=gsp[isp], no_labels=True)
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
