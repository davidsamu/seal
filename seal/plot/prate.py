#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to plot raster and rate plots.

@author: David Samu
"""

import numpy as np
import pandas as pd

from quantities import ms

from seal.object import constants
from seal.plot import putil


def plot_SR(u, stims, feat=None, vals=None, nrate=None, colors=None,
            add_stim_name=True, fig=None, sps=None, **kwargs):
    """Plot stimulus response (raster and rate) for mutliple stimuli."""

    if not u.to_plot():
        return

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
            # One trial set: stimulus specific color.
            if len(trs) == 1:
                cols = putil.stim_colors[stim]
            else:  # more than one trial set: value specific color
                colcyc = putil.get_colors()
                cols = [next(colcyc) for i in range(len(trs))]

        # Plot response on raster and rate plots.
        prd, ref = stims.loc[stim, 'prd'], stim + ' on'
        res = plot_rr(u, prd, ref, nrate, trs, cols=cols, fig=fig, sps=rr_sps,
                      **kwargs)
        _, raster_axs, rate_ax = res

        # Add stimulus name to rate plot.
        if add_stim_name:
            color = (putil.stim_colors[stim]
                     if vals is None or len(vals) == 1 else 'k')
            rate_ax.text(0.02, 0.95, stim, fontsize=10, color=color,
                         va='top', ha='left',
                         transform=rate_ax.transAxes)

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


def prep_rr_plot_params(u, prd, ref, nrate=None, trs=None):
    """Prepare plotting parameters."""

    # Get trial params.
    t1s, t2s = u.pr_times(prd, concat=False)
    ref_ts = u.ev_times(ref)
    if trs is None:
        trs = u.ser_inc_trials()

    # Get spikes.
    spikes = [u._Spikes.get_spikes(tr, t1s, t2s, ref_ts) for tr in trs]

    # Get rates and rate times.
    nrate = u.init_nrate(nrate)
    rates = [u._Rates[nrate].get_rates(tr, t1s, t2s, ref_ts)
             for tr in trs]

    # Get stimulus periods.
    stim_prd = constants.ev_stim.loc[ref]

    # Get trial set names.
    names = trs.index

    # Get unit's baseline rate.
    baseline = u.get_baseline()

    return trs, spikes, rates, stim_prd, names, baseline


def plot_rr(u, prd, ref, nrate=None, trs=None, no_labels=False,
            rate_kws=dict(), **kwargs):
    """Plot raster and rate plot of unit for selected sets of trials."""

    if not u.to_plot():
        return

    # Set up params.
    plot_params = prep_rr_plot_params(u, prd, ref, nrate, trs)
    trs, spikes, rates, stim_prd, names, baseline = plot_params

    # Set labels.
    if no_labels:
        title = None
        rate_kws.update({'xlab': None, 'ylab': None, 'add_lgn': False})
    else:
        title = u.Name
        rate_kws['xlab'] = prd

    # Plot raster and rate.
    res = raster_rate(spikes, rates, names, prds=[stim_prd], baseline=baseline,
                      title=title, rate_kws=rate_kws, **kwargs)
    fig, raster_axs, rate_ax = res

    return fig, raster_axs, rate_ax


def raster_rate(spk_list, rate_list, names=None, prds=None, cols=None,
                baseline=None, title=None, rs_ylab=True, rate_kws=dict(),
                fig=None, ffig=None, sps=None):
    """Plot raster and rate plots."""

    # Init subplots.
    sps, fig = putil.sps_fig(sps, fig)
    gsp = putil.embed_gsp(sps, 2, 1, height_ratios=[.66, 1], hspace=.15)
    n_sets = max(len(spk_list), 1)  # let's add an empty axes if no data
    gsp_raster = putil.embed_gsp(gsp[0], n_sets, 1, hspace=.15)
    gsp_rate = putil.embed_gsp(gsp[1], 1, 1)

    # Init colors.
    if cols is None:
        col_cyc = putil.get_colors(mpl_colors=True)
        cols = [next(col_cyc) for i in range(n_sets)]

    # Raster plots.
    raster_axs = [fig.add_subplot(gsp_raster[i, 0]) for i in range(n_sets)]
    for i, (spk_trs, ax) in enumerate(zip(spk_list, raster_axs)):
        ylab = names[i] if (rs_ylab and names is not None) else None
        raster(spk_trs, prds=prds, c=cols[i], ylab=ylab, ax=ax)
        putil.hide_axes(ax)
    if len(raster_axs):
        putil.set_labels(raster_axs[0], title=title)  # add title to top raster

    # Rate plot.
    rate_ax = fig.add_subplot(gsp_rate[0, 0])
    rate(rate_list, names, prds=prds, cols=cols, baseline=baseline,
         **rate_kws, ax=rate_ax)

    # Synchronize raster's x axis limits to rate.
    xlim = rate_ax.get_xlim()
    [ax.set_xlim(xlim) for ax in raster_axs]

    # Save and return plot.
    putil.save_fig(fig, ffig)
    return fig, raster_axs, rate_ax


def raster(spk_trains, t_unit=ms, prds=None, size=3.0, c='b', xlim=None,
           title=None, xlab=None, ylab=None, ffig=None, ax=None):
    """Plot rasterplot."""

    # Init.
    ax = putil.axes(ax)

    putil.plot_periods(prds, ax=ax)

    if not len(spk_trains):
        return ax

    # Plot raster.
    for i, spk_tr in enumerate(spk_trains):
        x = spk_tr.rescale(t_unit)
        y = (i+1) * np.ones_like(x)
        ax.scatter(x, y, c=c, s=size, edgecolor='w')

    # Format plot.
    ylim = [0.5, len(spk_trains)+0.5] if len(spk_trains) else [0, 1]
    if xlab is not None:
        xlab = putil.t_lbl.format(xlab)
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title)
    putil.hide_axes(ax, show_x=True)
    putil.hide_spines(ax)

    # Order trials from top to bottom, only after setting axis limits.
    ax.invert_yaxis()

    # Save and return plot.
    putil.save_fig(ffig=ffig)
    return ax


def rate(rate_list, names=None, prds=None, pval=0.05, test='t-test',
         test_kws={}, xlim=None, ylim=None, cols=None, baseline=None,
         title=None, xlab=None, ylab=putil.FR_lbl, add_lgn=True, lgn_lbl='trs',
         ffig=None, ax=None):
    """Plot firing rate."""

    # Init.
    ax = putil.axes(ax)

    # Plot periods and baseline first.
    putil.plot_periods(prds, ax=ax)
    if baseline is not None:
        putil.add_baseline(baseline, ax=ax)

    if not len(rate_list):
        return ax

    if cols is None:
        cols = putil.get_colors()
    if names is None:
        names = len(rate_list) * ['']

    # Iterate through list of rate arrays
    lbl = None
    xmin, xmax, ymax = None, None, None
    for name, rts, col in zip(names, rate_list, cols):

        # Skip empty array (no trials).
        if not rts.shape[0]:
            continue

        # Set line label.
        lbl = str(name)
        if lgn_lbl is not None:
            lbl += ' ({} {})'.format(rts.shape[0], lgn_lbl)

        # Plot mean +- SEM of rate vectors.
        tvec, meanr, semr = rts.columns, rts.mean(), rts.sem()
        ax.plot(tvec, meanr, label=lbl, color=col)
        ax.fill_between(tvec, meanr-semr, meanr+semr, alpha=0.2,
                        facecolor=col, edgecolor=col)

        # Update limits.
        tmin, tmax, rmax = min(tvec), max(tvec), max(meanr+semr)
        xmin = min(xmin, tmin) if xmin is not None else tmin
        xmax = max(xmax, tmax) if xmax is not None else tmax
        ymax = max(ymax, rmax) if ymax is not None else rmax

    # Set ticks, labels and axis limits.
    if xlim is None:
        if xmin == xmax:  # avoid setting identical limits
            xmax = None
        xlim = (xmin, xmax)
    if ylim is None:
        ymax = 1.02 * ymax if (ymax is not None) and (ymax > 0) else None
        ylim = (0, ymax)
    if xlab is not None:
        xlab = putil.t_lbl.format(xlab)
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title)
    t1, t2 = ax.get_xlim()  # in case it was set to None
    tmarks, tlbls = putil.get_tick_marks_and_labels(t1, t2)
    putil.set_xtick_labels(ax, tmarks, tlbls)
    putil.set_max_n_ticks(ax, 7, 'y')

    # Add legend.
    if add_lgn and lbl is not None:
        putil.set_legend(ax, loc=1, borderaxespad=0.5, handletextpad=0.1)

    # Add significance line to top of axes.
    if (pval is not None) and (len(rate_list) == 2):
        r1, r2 = rate_list
        putil.plot_signif_prds(r1, r2, pval, test, test_kws,
                               color='m', linewidth=4.0, ax=ax)

    # Save and return plot.
    putil.save_fig(ffig=ffig)
    return ax
