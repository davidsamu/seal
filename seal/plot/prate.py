#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to plot raster and rate plots.

@author: David Samu
"""

import numpy as np
import pandas as pd

from quantities import ms

from seal.util import util
from seal.plot import putil


def raster_rate(spk_list, rate_list, names=None, prds=None, cols=None,
                title=None, rs_ylab=True, rate_kws=dict(), fig=None,
                ffig=None, gsp=None):
    """Plot raster and rate plots."""

    # Create subplots (as nested gridspecs).
    n_sets = max(len(spk_list), 1)  # let's create an empty axes if no data
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
        ylab = names[i] if (rs_ylab and names is not None) else None
        raster(spk_trs, prds=prds, c=cols[i], ylab=ylab, ax=ax)
        putil.hide_axes(ax)
    if len(raster_axs):
        putil.set_labels(raster_axs[0], title=title)  # add title to top raster

    # Rate plot.
    rate_ax = fig.add_subplot(gsp_rate[0, 0])
    rate(rate_list, names, prds=prds, cols=cols, **rate_kws, ax=rate_ax)

    # Synchronize raster's x axis limits to rate.
    xlim = rate_ax.get_xlim()
    [ax.set_xlim(xlim) for ax in raster_axs]

    # Save and return plot.
    putil.save_fig(fig, ffig)
    return fig, raster_axs, rate_ax


def raster(spk_trains, t_unit=ms, prds=None, size=2.0, c='b', xlim=None,
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
         test_kws={}, xlim=None, ylim=None, cols=None, title=None, xlab=None,
         ylab=putil.FR_lbl, add_lgn=True, lgn_lbl='trs', ffig=None, ax=None):
    """Plot firing rate."""

    # Init.
    ax = putil.axes(ax)

    putil.plot_periods(prds, ax=ax)

    if not len(rate_list):
        return ax

    if cols is None:
        cols = putil.get_colors()
    if names is None:
        names = len(rate_list) * ['']

    # Iterate through list of rate arrays
    lbl = None
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

    # Set ticks, labels and axis limits.
    if xlim is None:
        # Use Pandas Series to deal with all sort of extreme cases.
        xlim = (pd.Series([rts.columns.min() for rts in rate_list]).min(),
                pd.Series([rts.columns.max() for rts in rate_list]).max())
    if ylim is None:
        ylim = (0, None)
    if xlab is not None:
        xlab = putil.t_lbl.format(xlab)
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title)
    xlim = ax.get_xlim()  # in case it was set to NaN
    xtcks = util.values_in_window(putil.t_ticks, xlim[0], xlim[1])
    putil.set_xtick_labels(ax, xtcks)
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
