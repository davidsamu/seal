#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to plot area under the curve (AUC) values over time.

@author: David Samu
"""

import numpy as np

from seal.plot import putil, pplot


def plot_auc_over_time(auc, tvec, prds=None, evts=None, xlim=None, ylim=None,
                       xlab='time', ylab='AUC', title=None, ax=None):
    """Plot AROC values over time."""

    # Init params.
    ax = putil.axes(ax)
    if xlim is None:
        xlim = [min(tvec), max(tvec)]

    # Plot periods first.
    putil.plot_periods(prds, ax=ax)

    # Plot AUC over time.
    pplot.lines(tvec, auc, ylim, xlim, xlab, ylab, title, color='green', ax=ax)

    # Add chance level line.
    putil.add_chance_level(ax=ax)

#    # Set minimum y axis scale.
#    ymin, ymax = ax.get_ylim()
#    ymin, ymax = min(ymin, 0.3), max(ymax, 0.7)
#    ax.set_ylim([ymin, ymax])

    # Set y tick labels.
    if ylim is not None and ylim[0] == 0 and ylim[1] == 1:
        tck_marks = np.linspace(0, 1, 5)
        tck_lbls = np.array(tck_marks, dtype=str)
        tck_lbls[1::2] = ''
        putil.set_ytick_labels(ax, tck_marks, tck_lbls)
    putil.set_max_n_ticks(ax, 5, 'y')

    # Plot event markers.
    putil.plot_event_markers(evts, ax=ax)

    return ax


def plot_auc_heatmap(aroc_mat, cmap='viridis', events=None, xlbl_freq=500,
                     ylbl_freq=10, xlab='time', ylab='unit index',
                     title='AROC over time', ffig=None):
    """Plot ROC AUC of list of units on heatmap."""

    # Plot heatmap.
    yticklabels = np.arange(len(aroc_mat.index)) + 1
    ax = pplot.heatmap(aroc_mat, vmin=0, vmax=1, cmap=cmap, xlab=xlab,
                       ylab=ylab, title=title, yticklabels=yticklabels)

    # Format labels.
    xlbls = aroc_mat.columns.map(str)
    xlbls[aroc_mat.columns % xlbl_freq != 0] = ''
    putil.set_xtick_labels(ax, lbls=xlbls)
    putil.rot_xtick_labels(ax, rot=0, ha='center')
    putil.sparsify_tick_labels(ax, 'y', istart=ylbl_freq-1, freq=ylbl_freq,
                               reverse=True)

    # Plot events.
    if events is not None:
        putil.plot_events(events, add_names=False, color='black', alpha=0.3,
                          ls='-', lw=1, ax=ax)

    # Save plot.
    putil.save_fig(ffig, dpi=300)
