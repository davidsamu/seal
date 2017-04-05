#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to plot raster and rate plots.

@author: David Samu
"""

import numpy as np

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from quantities import ms

from seal.analysis import stats
from seal.util import util
from seal.plot import putil


# Spike marker size for raster plots.
wsp, hsp = 1, .8  # vertical bar


def prep_rr_plot_params(u, prd, ref, nrate=None, trs=None, max_len=None):
    """Prepare plotting parameters."""

    # Get trial params.
    t1s, t2s = u.pr_times(prd, concat=False)

    # Truncate to maximum duration.
    if max_len is not None:
        ilong = t2s - t1s > float(max_len.rescale(ms))
        t2s[ilong] = t1s + max_len

    ref_ts = u.ev_times(ref)
    if trs is None:
        trs = u.ser_inc_trials()

    # Get spikes.
    spikes = [u._Spikes.get_spikes(tr, t1s, t2s, ref_ts) for tr in trs]

    # Get rates and rate times.
    nrate = u.init_nrate(nrate)
    rates = [u._Rates[nrate].get_rates(tr, t1s, t2s, ref_ts) for tr in trs]

    # Get trial set names.
    names = trs.index

    # Get unit's baseline rate.
    baseline = u.get_baseline()

    return trs, spikes, rates, names, baseline


def plot_rr(u, prd, ref, prds=None, evts=None, nrate=None, trs=None,
            max_len=None, no_labels=False, rate_kws=None, title=None,
            **kwargs):
    """Plot raster and rate plot of unit for selected sets of trials."""

    if not u.to_plot():
        return

    if rate_kws is None:
        rate_kws = dict()

    # Set up params.
    plot_params = prep_rr_plot_params(u, prd, ref, nrate, trs, max_len)
    trs, spikes, rates, names, baseline = plot_params

    # Set labels.
    if no_labels:
        rate_kws.update({'xlab': None, 'ylab': None, 'add_lgn': False})
    else:
        rate_kws['xlab'] = prd

    # Plot raster and rate.
    res = raster_rate(spikes, rates, names, prds, evts, baseline=baseline,
                      title=title, rate_kws=rate_kws, **kwargs)
    fig, raster_axs, rate_ax = res

    return fig, raster_axs, rate_ax


def raster_rate(spk_list, rate_list, names=None, prds=None, evts=None,
                cols=None, baseline=None, title=None, rs_ylab=True,
                rate_kws=None, fig=None, ffig=None, sps=None):
    """Plot raster and rate plots."""

    if rate_kws is None:
        rate_kws = dict()

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
    rate(rate_list, names, prds, evts, cols, baseline, **rate_kws, ax=rate_ax)

    # Synchronize raster's x axis limits to rate plot's limits.
    xlim = rate_ax.get_xlim()
    [ax.set_xlim(xlim) for ax in raster_axs]

    # Save and return plot.
    putil.save_fig(ffig, fig)
    return fig, raster_axs, rate_ax


def raster(spk_trains, t_unit=ms, prds=None, c='b', xlim=None,
           title=None, xlab=None, ylab=None, ffig=None, ax=None):
    """Plot rasterplot."""

    # Init.
    ax = putil.axes(ax)

    putil.plot_periods(prds, ax=ax)
    putil.set_limits(ax, xlim)

    # There's nothing to plot.
    if not len(spk_trains):
        return ax

    # Plot raster.
    for i, spk_tr in enumerate(spk_trains):
        x = np.array(spk_tr.rescale(t_unit))
        y = (i+1) * np.ones_like(x)

        # Spike markers are plotted in absolute size (figure coordinates).
        # ax.scatter(x, y, c=c, s=1.8, edgecolor=c, marker='|')

        # Spike markers are plotted in relate size (axis coordinates)
        patches = [Rectangle((xi-wsp/2, yi-hsp/2), wsp, hsp)
                   for xi, yi in zip(x, y)]
        collection = PatchCollection(patches, facecolor=c, edgecolor=c)
        ax.add_collection(collection)

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
    putil.save_fig(ffig)
    return ax


def rate(rate_list, names=None, prds=None, evts=None, cols=None, baseline=None,
         pval=0.05, test='mann_whitney_u', test_kws=None, xlim=None, ylim=None,
         title=None, xlab=None, ylab=putil.FR_lbl, add_lgn=True, lgn_lbl='trs',
         ffig=None, ax=None):
    """Plot firing rate."""

    # Init.
    ax = putil.axes(ax)
    if test_kws is None:
        test_kws = dict()

    # Plot periods and baseline first.
    putil.plot_periods(prds, ax=ax)
    if baseline is not None:
        putil.add_baseline(baseline, ax=ax)
    putil.set_limits(ax, xlim)

    if not len(rate_list):
        return ax

    if cols is None:
        cols = putil.get_colors()
    if names is None:
        names = len(rate_list) * ['']

    # Iterate through list of rate arrays
    xmin, xmax, ymax = None, None, None
    for i, rts in enumerate(rate_list):

        # Init.
        name = names[i]
        col = cols[i]

        # Skip empty array (no trials).
        if not rts.shape[0]:
            continue

        # Set line label. Convert to Numpy array to format floats nicely.
        lbl = str(np.array(name)) if util.is_iterable(name) else str(name)

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
    if add_lgn and len(rate_list):
        putil.set_legend(ax, loc=1, borderaxespad=0.0, handletextpad=0.4,
                         handlelength=0.6)

    # Add significance line to top of axes.
    if (pval is not None) and (len(rate_list) == 2):
        rates1, rates2 = rate_list
        sign_prds = stats.sign_periods(rates1, rates2, pval, test, **test_kws)
        putil.plot_signif_prds(sign_prds, color='m', linewidth=4.0, ax=ax)

    # Plot event markers.
    putil.plot_event_markers(evts, ax=ax)

    # Save and return plot.
    putil.save_fig(ffig)
    return ax
