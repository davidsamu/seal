#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 20:27:41 2016

Functions to plot raster and rate plots.

@author: David Samu
"""

import numpy as np
from quantities import ms

from seal.util import util
from seal.plot import putil


def empty_raster_rate(fig, outer_gsp, nraster):
    """Plot empty raster and rate plots."""

    mock_gsp_rasters = putil.embed_gsp(outer_gsp[0], nraster, 1, hspace=.15)
    mock_raster_axs = [putil.add_mock_axes(fig, mock_gsp_rasters[i, 0])
                       for i in range(nraster)]

    mock_gsp_rate = putil.embed_gsp(outer_gsp[1], 1, 1)
    mock_rate_ax = putil.add_mock_axes(fig, mock_gsp_rate[0, 0])

    return mock_raster_axs, mock_rate_ax


def raster_rate(spk_list, rate_list, tvec, names, t1=None, t2=None, prds=None,
                cols=None, title=None, rs_ylab=True, rate_kws=None,
                fig=None, ffig=None, gsp=None):
    """Plot raster and rate plots."""

    # Create subplots (as nested gridspecs).
    n_sets = len(spk_list)
    fig = putil.figure(fig)
    gsp = putil.gridspec(2, 1, gsp, height_ratios=[1, 1])
    gsp_raster = putil.embed_gsp(gsp[0], n_sets, 1, hspace=.15)
    gsp_rate = putil.embed_gsp(gsp[1], 1, 1)

    # Init colors.
    if cols is None:
        col_cyc = putil.get_colors(mpl_colors=True)
        cols = [next(col_cyc) for i in range(n_sets)]

    # Raster plots.
    raster_axs = [fig.add_subplot(gsp_raster[i, 0]) for i in range(n_sets)]
    for i, (spk_trs, ax) in enumerate(zip(spk_list, raster_axs)):
        ylab = names[i] if rs_ylab else None
        raster(spk_trs, t1, t2, prds=prds, c=cols[i], ylab=ylab, ax=ax)
        putil.hide_axes(ax)
    putil.set_labels(raster_axs[0], title=title)  # add title to top raster

    # Rate plot.
    rate_ax = fig.add_subplot(gsp_rate[0, 0])
    rate(rate_list, tvec, names, t1, t2, prds=prds, cols=cols, **rate_kws,
         ax=rate_ax)

    # Save and return plot.
    putil.save_fig(fig, ffig)
    return fig, raster_axs, rate_ax


def raster(spk_trains, t1=None, t2=None, t_unit=ms, prds=None, size=1.5, c='b',
           title=None, xlab=putil.t_lbl, ylab=None, ffig=None, ax=None):
    """Plot rasterplot."""

    t1, t2 = t1.rescale(t_unit), t2.rescale(t_unit)

    # Init figure.
    ax = putil.axes(ax)
    putil.plot_periods(prds, t_unit, ax=ax)

    # Plot raster.
    for i, spk_tr in enumerate(spk_trains):
        t = spk_tr.rescale(t_unit)
        t = util.values_in_window(t, t1, t2)  # get spikes within time window
        ax.scatter(t, (i+1) * np.ones_like(t), c=c, s=size, edgecolor='w')

    # Format plot.
    ylim = [0.5, len(spk_trains)+0.5] if len(spk_trains) else [0, 1]
    putil.set_limits(ax, [t1, t2], ylim)
    putil.hide_axes(ax, show_x=True)
    putil.hide_spines(ax)
    putil.set_labels(ax, xlab, ylab, title)

    # Order trials from top to bottom, only after setting axis limits.
    ax.invert_yaxis()

    # Save and return plot.
    putil.save_fig(ffig=ffig)
    return ax


def rate(rate_list, tvec, names, t1=None, t2=None, t_unit=ms, prds=None,
         pval=0.05, test='t-test', test_kws={}, xlim=None, ylim=None,
         cols=None, title=None, xlab=putil.t_lbl, ylab=putil.FR_lbl,
         add_lgn=True, lgn_lbl='trs', ffig=None, ax=None):
    """Plot firing rate."""

    # Init time vector and period to plot.
    tvec = tvec.rescale(t_unit)
    t1 = min(tvec) if t1 is None else t1.rescale(t_unit)
    t2 = max(tvec) if t2 is None else t2.rescale(t_unit)

    # Select requested time window.
    idxs = util.indices_in_window(tvec, t1, t2)
    rate_list = [np.array(rts)[:, idxs] for rts in rate_list]
    tvec = tvec[idxs]

    # Init axes and colors.
    ax = putil.axes(ax)
    if cols is None:
        cols = putil.get_colors(mpl_colors=True)

    # Start by highlighting the periods.
    putil.plot_periods(prds, t_unit, ax=ax)

    # Iterate through list of rate arrays.
    for i, (name, rts, col) in enumerate(zip(names, rate_list, cols)):

        # Skip empty array (no trials).
        if not rts.shape[0]:
            continue

        # Set line label.
        lbl = name
        if lgn_lbl is not None:
            lbl += ' ({} {})'.format(rts.shape[0], lgn_lbl)

        # Plot mean +- SEM of rate vectors.
        meanr, semr = util.mean_sem(rts)
        ax.plot(tvec, meanr, label=lbl, color=col)
        ax.fill_between(tvec, meanr-semr, meanr+semr, alpha=0.2,
                        facecolor=col, edgecolor=col)

    # Set ticks, labels and axis limits.
    if ylim is None:
        ylim = (0, None)
    putil.format_plot(ax, [t1, t2], ylim, xlab, ylab, title)
    xtcks = util.values_in_window(putil.t_ticks, t1, t2)
    putil.set_xtick_labels(ax, xtcks)
    putil.set_max_n_ticks(ax, 7, 'y')

    # Add legend.
    putil.set_legend(ax, add_lgn, loc=1, borderaxespad=0.5, handletextpad=0.1)

    # Add significance line to top of axes.
    if (pval is not None) and (len(rate_list) == 2):
        r1, r2 = rate_list
        putil.plot_signif_prds(r1, r2, tvec, pval, test, test_kws,
                               color='m', linewidth=4.0, ax=ax)

    # Save and return plot.
    putil.save_fig(ffig=ffig)
    return ax
