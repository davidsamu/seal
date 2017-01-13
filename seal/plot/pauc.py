#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to area under the curve (AUC) values over time.

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
    pplot.lines(tvec, auc, ylim, xlim, xlab, ylab, title, ax=ax)

    # Add chance level line.
    putil.add_chance_level(ax=ax)

    # Set y tick labels.
    if ylim is not None and ylim[0] == 0 and ylim[1] == 1:
        tck_marks = np.linspace(0, 1, 5)
        tck_lbls = np.array(tck_marks, dtype=str)
        tck_lbls[1::2] = ''
        putil.set_ytick_labels(ax, tck_marks, tck_lbls)
    putil.set_max_n_ticks(ax, 5, 'y')

    # Plot event markers.
    putil.plot_event_marker(evts, ax=ax)

    return ax
