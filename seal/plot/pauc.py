#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to plot area under the curve (AUC) values over time.

@author: David Samu
"""

import numpy as np
import pandas as pd
import seaborn as sns

from seal.util import util
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
                     title='AROC over time', ffig=None, fig=None):
    """Plot ROC AUC of list of units on heatmap."""

    fig = putil.figure(fig)

    # Plot heatmap.
    yticklabels = np.arange(len(aroc_mat.index)) + 1
    ax = pplot.heatmap(aroc_mat, vmin=0, vmax=1, cmap=cmap, xlab=xlab,
                       ylab=ylab, title=title, yticklabels=yticklabels)

    # Format labels.
    xlbls = aroc_mat.columns.map(str)
    xlbls[aroc_mat.columns % xlbl_freq != 0] = ''
    putil.set_xtick_labels(ax, lbls=xlbls)
    putil.rot_xtick_labels(ax, rot=0, ha='center')
    putil.sparsify_tick_labels(fig, ax, 'y', istart=ylbl_freq-1,
                               freq=ylbl_freq, reverse=True)
    putil.hide_tick_marks(ax)
    putil.hide_spines(ax)

    # Plot events.
    if events is not None:
        putil.plot_events(events, add_names=False, color='black', alpha=0.3,
                          ls='-', lw=1, ax=ax)

    # Save plot.
    putil.save_fig(ffig, dpi=300)


def plot_ROC_mean(d_faroc, t1=None, t2=None, ylim=None, colors=None,
                  ylab='AROC', ffig=None):
    """Plot mean ROC curves over given period."""

    # Import results.
    d_aroc = {}
    for name, faroc in d_faroc.items():
        aroc = util.read_objects(faroc, 'aroc')
        d_aroc[name] = aroc.unstack().T

    # Format results.
    laroc = pd.DataFrame(pd.concat(d_aroc), columns=['aroc'])
    laroc['task'] = laroc.index.get_level_values(0)
    laroc['time'] = laroc.index.get_level_values(1)
    laroc['unit'] = laroc.index.get_level_values(2)
    laroc.index = np.arange(len(laroc.index))

    # Init figure.
    fig = putil.figure(figsize=(6, 6))
    ax = sns.tsplot(laroc, time='time', value='aroc', unit='unit',
                    condition='task', color=colors)

    # Highlight stimulus periods.
    putil.plot_periods(ax=ax)

    # Plot mean results.
    [ax.lines[i].set_linewidth(3) for i in range(len(ax.lines))]

    # Add chance level line.
    putil.add_chance_level(ax=ax, alpha=0.8, color='k')
    ax.lines[-1].set_linewidth(1.5)

    # Format plot.
    xlab = 'Time since S1 onset (ms)'
    putil.set_labels(ax, xlab, ylab)
    putil.set_limits(ax, [t1, t2], ylim)
    putil.set_spines(ax, bottom=True, left=True, top=False, right=False)
    putil.set_legend(ax, loc=0)

    # Save plot.
    putil.save_fig(ffig, fig, ytitle=1.05, w_pad=15)
