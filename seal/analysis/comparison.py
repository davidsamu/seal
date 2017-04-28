# -*- coding: utf-8 -*-
"""
Functions related to analyzing stimulus comparison effects.

@author: David Samu
"""

import numpy as np
import pandas as pd
import seaborn as sns

from seal.plot import putil
from seal.roc import rocutil


def import_CE_ROC_results(ulists, tasks, CE_feat_pars, nrate, tstep, n_perm,
                          offsets, min_len, pth, vth_hi, vth_lo, aroc_res_dir):
    """Import CE ROC results."""

    eff_pars = [('high', 'S > D'), ('low', 'D > S')]
    prefix, btw_str, t1, t2 = CE_feat_pars
    d_eff_t_res = {}
    for nlist in ulists:
        ares_dir = aroc_res_dir + 'CE/{}/'.format(nlist)
        task = tasks[nlist]
        ftable = rocutil.aroc_table_fname(ares_dir, task, nrate, tstep, n_perm,
                                          offsets, 'S2', min_len, pth, vth_hi,
                                          vth_lo)
        d_eff_t_res[nlist] = pd.read_excel(ftable)
    eff_t_res = pd.concat(d_eff_t_res)

    return eff_pars, eff_t_res


def plot_CE_time_distribution(ulists, eff_t_res, eff_pars, aroc_res_dir,
                              bins=None):
    """Plot distribution of ROC comparison effect timing across groups."""

    # Init.
    putil.set_style('notebook', 'white')
    if bins is None:
        bins = np.arange(2000, 2600, 50)
    fig, _, axs = putil.get_gs_subplots(nrow=1, ncol=len(eff_pars),
                                        subw=5, subh=4,
                                        create_axes=True, as_array=False)

    # Plot CE timing distribution for each unit group.
    for (eff_dir, eff_lbl), ax in zip(eff_pars, axs):
        etd = eff_t_res.loc[eff_t_res.effect_dir == eff_dir, 'time']
        for nlist in ulists:
            tvals = etd.loc[nlist]
            lbl = '{} (n={})'.format(nlist, len(tvals))
            sns.distplot(tvals, bins, label=lbl, ax=ax)
        putil.set_labels(ax, 'effect timing (ms since S1 onset)', '', eff_lbl)

    # Format plots.
    [ax.legend() for ax in axs]
    [putil.hide_tick_labels(ax, show_x_tick_lbls=True) for ax in axs]
    putil.sync_axes(axs, sync_y=True)

    # Save plot.s
    ffig = aroc_res_dir + 'CE/CE_timing_distributions.png'
    putil.save_fig(ffig, fig)
