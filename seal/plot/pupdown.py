# -*- coding: utf-8 -*-

"""
Functions to plot up-down states analysis results.

@author: David Samu
"""

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from seal.plot import putil, pplot
from seal.util import constants


# Plotting params.
c = 'k'
ec = 'k'
wsp, hsp = 1, .8  # spike marker size


# %% Spike raster plots.

def plot_up_down_raster(Spikes, task, rec, itrs):
    """Plot spike raster for up-down dynamics analysis."""

    # Set params for plotting.
    uids, trs = Spikes.index, Spikes.columns
    plot_trs = trs[itrs]
    ntrs = len(plot_trs)
    nunits = len(uids)
    tr_gap = nunits / 2

    # Init figure.
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
    for stim in constants.stim_dur.index:
        t_start, t_stop = constants.fixed_tr_prds.loc[stim]
        events = pd.DataFrame([(t_start, 't_start'), (t_stop, 't_stop')],
                              index=['start', 'stop'],
                              columns=['time', 'label'])
        putil.plot_events(events, add_names=False, color='grey',
                          alpha=0.5, ls='-', lw=0.5, ax=ax)

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
    xlim = constants.fixed_tr_prds.loc['whole trial']
    ylim = [-tr_gap/2, ntrs * (nunits+tr_gap)-tr_gap/2]
    xlab = 'Time since S1 onset (ms)'
    ylab = 'Trial number'
    title = '{} {}'.format(rec, task)
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title)
    putil.set_spines(ax, True, False, False, False)

    # Save figure.
    ffig = 'results/UpDown/UpDown_dynamics_{}_{}.pdf'.format(rec, task)
    putil.save_fig(ffig, fig, dpi=600)


# %% Spike count histrogram plots.

def plot_spike_count_hist(spk_cnt, title, ax, hist=True,
                          kde_alpha=1.0, kde_color='b', kde_lw=2):
    """Plot spike count histogram."""

    # Set histrogram parameters.
    max_spk = spk_cnt.max()
    bins = np.linspace(0, max_spk+1, max_spk+2) - 0.5
    hist_kws = {'edgecolor': 'grey'}
    kde_kws = {'alpha': kde_alpha, 'color': kde_color, 'lw': kde_lw}

    # Plot spike count histogram.
    sns.distplot(spk_cnt, bins=bins, ax=ax, hist=hist,
                 hist_kws=hist_kws, kde_kws=kde_kws)

    # Format plot.
    xlim = None
    ylim = None
    xlab = '# spikes'
    ylab = 'density'
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title)


def plot_prd_spike_count(spk_cnt, prd, sps, fig):
    """Plot spike count results for period."""

    # Init axes.
    ax_total, ax_per_tr, ax_mean = putil.sps_add_axes(fig, sps, 1, 3)

    # Plot total spike count histogram.
    plot_spike_count_hist(spk_cnt.unstack(), prd + ', all trials', ax_total)

    # Per trial.
    cols = sns.color_palette('Spectral', len(spk_cnt.index))
    for (itr, nspk), col in zip(spk_cnt.iterrows(), cols):
        plot_spike_count_hist(nspk, prd + ', per trial', ax_per_tr,
                              False, 1.0, col, 1)

    # Mean over time.
    x = spk_cnt.index
    y = spk_cnt.median(1)
    pplot.lines(x, y, xlim=[x.min(), x.max()], xlab='trial index',
                ylab='mean spike counts', title='median over time', ax=ax_mean)

    # Format figures.
    for ax in (ax_total, ax_per_tr):
        putil.set_limits(ax, xlim=[-0.5, None], ylim=[0, None])

    return ax_total, ax_per_tr, ax_mean


def plot_spike_count_results(bspk_cnts, rec, task, prds, binsize):
    """Plot spike count results on composite plot for multiple periods."""

    # Init figure.
    putil.set_style('notebook', 'ticks')
    fig, gsp, _ = putil.get_gs_subplots(nrow=len(prds), ncol=1,
                                        subw=15, subh=4, create_axes=False)

    # Plot each period.
    for prd, sps in zip(prds, gsp):
        plot_prd_spike_count(bspk_cnts[prd], prd, sps, fig)

    # Save figure.
    title = '{} {}, binsize: {} ms'.format(rec, task, int(binsize))
    ffig = ('results/UpDown/UpDown_spk_cnt_hist_' +
            '{}_{}_bin_{}.png'.format(rec, task, int(binsize)))
    putil.save_fig(ffig, fig, title, ytitle=1.1)
