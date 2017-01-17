#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to plot spike waveforms.

@author: David Samu
"""

import numpy as np

from matplotlib.collections import LineCollection

from seal.plot import putil


def plot_wfs(waveforms, tvec, cols=None, lw=0.1, alpha=0.05, xlim=None,
             ylim=None, title=None, xlab=None, ylab=None, ffig=None, ax=None,
             **kwargs):
    """
    Plot large set of waveforms efficiently as LineCollections.

    waveforms: waveform matrix of (waveform x time sample)
    tvec: vector of sample times
    cols: color matrix of (waveform x RGBA)

    """

    # Init.
    nwfs = waveforms.shape[0]
    ntsamp = tvec.size
    ax = putil.axes(ax)

    if cols is None:
        cols = np.tile(putil.convert_to_rgba('g'), (nwfs, 1))

    # Plot all waveforms efficiently at the same time as LineCollections.
    # Reformat waveform matrix and tvec vector into format:
    # [[(t0, v0), (t1, v1)], [(t0, v0), (t1, v1)], ...]
    wf_col = waveforms.reshape((-1, 1))
    tvec_col = np.tile(np.array(tvec), (nwfs, 1)).reshape((-1, 1))
    tvec_wf_cols = np.hstack((tvec_col, wf_col)).reshape(-1, 1, 2)
    tv_segments = np.hstack([tvec_wf_cols[:-1], tvec_wf_cols[1:]])
    btw_wf_mask = ntsamp * np.arange(1, nwfs) - 1
    tv_segments = np.delete(tv_segments, btw_wf_mask, axis=0)

    # Set color of each segment.
    cols_segments = np.repeat(cols, ntsamp-1, axis=0)

    # Create and add LineCollection to axes.
    lc = LineCollection(tv_segments, linewidths=lw, colors=cols_segments,
                        **kwargs)
    ax.add_collection(lc)
    # Need to update view manually after adding artists manually.
    ax.autoscale_view()

    # Format plot.
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title)

    # Save and return plot.
    putil.save_fig(ffig=ffig)
    return ax
