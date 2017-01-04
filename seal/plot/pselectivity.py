#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to plot combined (composite) plots..

@author: David Samu
"""

import numpy as np
import pandas as pd

from quantities import deg

from seal.object import constants
from seal.plot import putil, ptuning, prate
from seal.util import util


# %% Functions to plot location selectivity.

def plot_LR(u, nrate=None, fig=None, sps=None, **kwargs):
    """Plot location response plot."""

    if not u.to_plot():
        return

    # Set up stimulus parameters.
    stims = pd.DataFrame(index=constants.stim_dur.index)
    stims['prd'] = ['around ' + stim for stim in stims.index]
    stims['dur'] = [u.pr_dur(stims.loc[stim, 'prd']) for stim in stims.index]

    # Init subplots.
    sps, fig = putil.sps_fig(sps, fig)

    # Raster & rate in trials sorted by stimulus location.
    raster_axs, rate_axs = prate.plot_SR(u, stims, 'Loc', None, nrate, None,
                                         True, fig, sps, no_labels=False,
                                         **kwargs)

    return raster_axs, rate_axs


# %% Functions to plot stimulus selectivity.

def plot_DR(u, nrate=None, fig=None, sps=None):
    """Plot 3x3 direction response plot, with polar plot in center."""

    if not u.to_plot():
        return

    # Set up stimulus parameters.
    stims = pd.DataFrame(index=constants.stim_dur.index)
    stims['prd'] = ['around ' + stim for stim in stims.index]
    stims['dur'] = [u.pr_dur(stims.loc[stim, 'prd'])
                    for stim in stims.index]

    # Init subplots.
    sps, fig = putil.sps_fig(sps, fig)
    gsp = putil.embed_gsp(sps, 3, 3)  # inner gsp with subplots

    # Polar plot.
    putil.set_style('notebook', 'white')
    ax_polar = fig.add_subplot(gsp[4], polar=True)
    for stim in stims.index:  # for each stimuli
        stim_resp = u.get_stim_resp_vals(stim, 'Dir')
        resp_stats = util.calc_stim_resp_stats(stim_resp)
        dirs, resp = np.array(resp_stats.index) * deg, resp_stats['mean']
        c = putil.stim_colors[stim]
        baseline = u.get_baseline()
        ptuning.plot_DR(dirs, resp, color=c, baseline=baseline, ax=ax_polar)
    putil.hide_ticks(ax_polar, 'y')

    # Raster-rate plots.
    putil.set_style('notebook', 'ticks')
    rr_pos = [5, 2, 1, 0, 3, 6, 7, 8]  # Position of each direction.
    rr_dir_pos = pd.Series(constants.all_dirs, index=rr_pos)

    rate_axs = []
    for isp, d in rr_dir_pos.iteritems():

        # Prepare plot formatting.
        first_dir = (isp == 0)

        # Plot stimulus response across stimuli.
        res = prate.plot_SR(u, stims, 'Dir', [d], nrate, None, first_dir, fig,
                            gsp[isp], no_labels=True)
        draster_axs, drate_axs = res

        # Remove axis ticks.
        for i, ax in enumerate(drate_axs):
            first_stim = (i == 0)
            show_x_tick_lbls = first_dir
            show_y_tick_lbls = first_dir & first_stim
            putil.hide_tick_labels(ax, show_x_tick_lbls, show_y_tick_lbls)

        # Add task name as title (to top center axes).
        if isp == 1:
            putil.set_labels(draster_axs[0], title=u.get_task(),
                             ytitle=1.10, title_kws={'loc': 'right'})

        rate_axs.extend(drate_axs)

    # Match scale of y axes.
    putil.sync_axes(rate_axs, sync_y=True)

    return ax_polar, rate_axs


def plot_rate_DS(u, nrate=None, fig=None, sps=None):
    """Plot rate and direction selectivity summary plot."""

    if not u.to_plot():
        return

    # Set up stimulus parameters.
    stims = pd.DataFrame(index=constants.stim_dur.index)
    stims['prd'] = ['around ' + stim for stim in stims.index]
    stims['dur'] = [u.pr_dur(stims.loc[stim, 'prd']) for stim in stims.index]

    # Init subplots.
    sps, fig = putil.sps_fig(sps, fig)
    gsp = putil.embed_gsp(sps, 3, 1)
    all_rr_sps, ds_sps, pa_rr_sps = [g for g in gsp]
    all_rate_ax, dir_rate_ax = [], []

    # Raster & rate over all trials.
    sraster_axs, srate_axs = prate.plot_SR(u, stims, nrate=nrate, fig=fig,
                                           sps=all_rr_sps, no_labels=True)
    all_rate_ax.extend(srate_axs)

    # Direction tuning.
    ax_polar, ax_tuning = ptuning.plot_DS(u, no_labels=True, fig=fig,
                                          sps=ds_sps)

    # Raster & rate in pref and anti trials.
    stim = stims.index[0]
    pa_dir = [u.pref_dir(stim), u.anti_pref_dir(stim)]
    res = prate.plot_SR(u, stims, 'Dir', pa_dir, nrate, None, True, fig,
                        pa_rr_sps, no_labels=True)
    draster_axs, drate_axs = res
    dir_rate_ax.extend(drate_axs)

    return all_rate_ax, dir_rate_ax, ax_polar, ax_tuning
