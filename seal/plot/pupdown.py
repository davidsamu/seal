# -*- coding: utf-8 -*-

"""
Functions to plot up-down states analysis results.

@author: David Samu
"""

import numpy as np

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from seal.plot import putil
from seal.util import constants, ua_query


# Plotting params.
c = 'k'
ec = 'k'
wsp, hsp = 1, .8  # spike marker size


def plot_up_down_raster(UA, task, rec, utids, prd, ref_ev, events, itrs):
    """Plot raster for up-down dynamics analysis."""

    if utids is None:
        utids = UA.utids(tasks=[task], recs=[rec], as_series=True)

    # Query spike times.
    uids = utids.index.droplevel('task')
    Spikes = ua_query.get_spike_times(UA, rec, task, uids, prd, ref_ev)

    # Set params for plotting.
    uids, trs = Spikes.index, Spikes.columns
    plot_trs = trs[itrs]
    ntrs = len(plot_trs)
    nunits = len(uids)
    tr_gap = nunits / 2

    # Plot spike raster.
    putil.set_style('notebook', 'ticks')
    fig = putil.figure(figsize=(10, ntrs))
    ax = fig.add_subplot(111)

    # Per trial, per unit.
    for itr, tr in enumerate(plot_trs):
        for iu, uid in enumerate(uids):

            # Init y level and spike times.
            i = (tr_gap + nunits) * itr + iu
            spk_tr = Spikes.loc[uid, tr]

            # Plot (spike time, y-level) pairs.
            x = np.array(spk_tr.rescale('ms'))
            y = (i+1) * np.ones_like(x)

            patches = [Rectangle((xi-wsp/2, yi-hsp/2), wsp, hsp)
                       for xi, yi in zip(x, y)]
            collection = PatchCollection(patches, facecolor=c, edgecolor=c)
            ax.add_collection(collection)

    # Add stimulus lines.
    putil.plot_events(events, add_names=False, color='grey', alpha=0.5,
                      ls='-', lw=0.5, ax=ax)

    # Add inter-trial shading.
    for itr in range(ntrs+1):
        ymin = itr * (tr_gap + nunits) - tr_gap + 0.5
        ax.axhspan(ymin, ymin+tr_gap, alpha=.05, color='grey')

    # Set tick labels.
    pos = np.arange(ntrs) * (tr_gap + nunits) + nunits/2
    lbls = plot_trs + 1
    putil.set_ytick_labels(ax, pos, lbls)
    # putil.sparsify_tick_labels(ax, 'y', freq=2, istart=1)
    putil.hide_tick_marks(ax, show_x_tick_mrks=True)

    # Format plot.
    xlim = constants.fixed_tr_prds.loc[prd]
    ylim = [-tr_gap/2, ntrs * (nunits+tr_gap)-tr_gap/2]
    xlab = 'Time since {} (ms)'.format(ref_ev)
    ylab = 'Trial number'
    title = '{} {}'.format(rec, task)
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title)
    putil.set_spines(ax, True, False, False, False)

    # Save figure.
    fname = 'results/UpDown/UpDown_dynamics_{}_{}.pdf'.format(rec, task)
    putil.save_fig(fname, fig, dpi=600)
