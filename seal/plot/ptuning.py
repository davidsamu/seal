#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Functions to plot tuning curve and stimulus selectivity.

@author: David Samu
"""

import numpy as np
import pandas as pd
from quantities import deg, rad, s

from seal.analysis import tuning, direction
from seal.plot import putil, pplot


def plot_DS(u, no_labels=False, ftempl=None, **kwargs):
    """Plot direction selectivity results."""

    if not u.to_plot():
        return

    # Test DS if it hasn't been yet.
    if not len(u.DS.index):
        u.test_DS()

    baseline = u.get_baseline()

    # Set up plot params.
    if no_labels:  # minimise labels on plot
        title = None
        kwargs['labels'] = False
        kwargs['DR_legend'] = True
        kwargs['tuning_legend'] = False
    else:
        title = u.Name

    ffig = None if ftempl is None else ftempl.format(u.name_to_fname())
    ax_DR, ax_tuning = plot_DR_tuning(u.DS, title=title, baseline=baseline,
                                      ffig=ffig, **kwargs)
    return ax_DR, ax_tuning


def plot_DR_tuning(DS, title=None, labels=True, baseline=None, DR_legend=True,
                   tuning_legend=True, ffig=None, fig=None, sps=None):
    """Plot direction selectivity on polar plot and tuning curve."""

    # Init subplots.
    sps, fig = putil.sps_fig(sps, fig)
    gsp = putil.embed_gsp(sps, 1, 2, wspace=0.3)
    ax_DR = fig.add_subplot(gsp[0], polar=True)
    ax_tuning = fig.add_subplot(gsp[1])
    DR_patches = []
    tuning_patches = []

    stims = DS.DSI.index
    for stim in stims:

        # Extract params and results.
        a, b, x0, sigma, FWHM, R2, RMSE = DS.TP.loc[stim]
        DSI = DS.DSI.wDS[stim]
        PD, cPD = DS.PD.loc[(stim, 'weighted'), ['PD', 'cPD']]
        dirs = np.array(DS.DR[stim].index) * deg
        SR, SRsem = [DS.DR[stim][col] for col in ('mean', 'sem')]

        # Center direction  and response stats.
        dirsc = direction.center_to_dir(dirs, PD)

        # Generate data points for plotting fitted tuning curve.
        x, y = tuning.gen_fit_curve(tuning.gaus, -180, 180,
                                    a=a, b=b, x0=x0, sigma=sigma)

        # Plot DR polar plot.
        color = putil.stim_colors.loc[stim]
        ttl = 'Direction response' if labels else None
        plot_DR(dirs, SR, DSI, PD, baseline, title=ttl, color=color, ax=ax_DR)

        # Collect parameters of DR plot (stimulus - response).
        s_pd = str(float(round(PD, 1)))
        s_pd_c = str(int(cPD)) if not np.isnan(float(cPD)) else 'nan'
        lgd_lbl = '{}:   {:.3f}'.format(stim, DSI)
        lgd_lbl += '     {:>5}$^\circ$ --> {:>3}$^\circ$ '.format(s_pd, s_pd_c)
        DR_patches.append(putil.get_artist(lgd_lbl, color))

        # Calculate and plot direction tuning curve.
        xticks = np.arange(-180, 180+1, 45)
        plot_tuning(x, y, dirsc, SR, SRsem, color, baseline, xticks,
                    ax=ax_tuning)

        # Collect parameters tuning curve fit.
        s_a, s_b, s_x0, s_sigma, s_FWHM = [str(float(round(p, 1)))
                                           for p in (a, b, x0, sigma, FWHM)]
        s_R2 = format(R2, '.2f')
        lgd_lbl = '{}:{}{:>6}{}{:>6}'.format(stim, 5 * ' ', s_a, 5 * ' ', s_b)
        lgd_lbl += '{}{:>6}{}{:>6}'.format(5 * ' ', s_x0, 8 * ' ', s_sigma)
        lgd_lbl += '{}{:>6}{}{:>6}'.format(8 * ' ', s_FWHM, 8 * ' ', s_R2)
        tuning_patches.append(putil.get_artist(lgd_lbl, color))

    # Format tuning curve plot (only after all stimuli has been plotted!).
    xlab = 'Difference from preferred direction (deg)' if labels else None
    ylab = putil.FR_lbl if labels else None
    xlim = [-180-5, 180+5]  # degrees
    ylim = [0, None]
    ttl = 'Tuning curve' if labels else None
    putil.format_plot(ax_tuning, xlim, ylim, xlab, ylab, ttl)
    putil.add_zero_line('y', ax=ax_tuning)  # add zero reference line

    # Set super title.
    if title is not None:
        fig.suptitle(title, y=1.1, fontsize='x-large')

    # Set legends.
    ylegend = -0.38 if labels else -0.15
    fr_on = False if labels else True
    lgd_kws = dict([('fancybox', True), ('shadow', False), ('frameon', fr_on),
                    ('framealpha', 1.0), ('loc', 'lower center'),
                    ('bbox_to_anchor', [0., ylegend, 1., .0]),
                    ('prop', {'family': 'monospace'})])
    DR_lgn_ttl = 'DSI'.rjust(20) + 'PD'.rjust(14) + 'PD8'.rjust(14)
    tuning_lgd_ttl = ('a (sp/s)'.rjust(35) + 'b (sp/s)'.rjust(15) +
                      'x0 (deg)'.rjust(13) + 'sigma (deg)'.rjust(15) +
                      'FWHM (deg)'.rjust(15) + 'R-squared'.rjust(15))

    lgn_params = [(DR_legend, DR_lgn_ttl, DR_patches, ax_DR),
                  (tuning_legend, tuning_lgd_ttl, tuning_patches, ax_tuning)]

    for (plot_legend, lgd_ttl, patches, ax) in lgn_params:
        if not plot_legend:
            continue
        if not labels:  # customize for summary plot
            lgd_ttl = None
        lgd = putil.set_legend(ax, handles=patches, title=lgd_ttl, **lgd_kws)
        lgd.get_title().set_ha('left')
        if lgd_kws['frameon']:
            lgd.get_frame().set_linewidth(.5)

    # Save figure.
    if ffig is not None:
        putil.save_figure(fig, ffig, rect_height=0.55)

    return ax_DR, ax_tuning


def plot_DR(dirs, resp, DSI=None, PD=None, baseline=None, plot_type='line',
            complete_missing_dirs=False, color='b', title=None, ffig=None,
            ax=None):
    """
    Plot response to each directions on polar plot, with a vector pointing to
    preferred direction (PD) with length DSI.
    Use plot_type to change between sector ('bar') and connected ('line') plot
    types.
    """

    ax = putil.axes(ax, polar=True)

    # Plot baseline.
    if baseline is not None:
        putil.add_baseline(baseline, ax=ax)

    # Remove NaNs.
    all_dirs = dirs
    not_nan = np.array(~pd.isnull(dirs) & ~pd.isnull(resp))
    dirs, resp = dirs[not_nan], np.array(resp[not_nan])

    # Prepare data.
    # Complete missing directions with 0 response.
    if complete_missing_dirs:
        for i, d in enumerate(all_dirs):
            if d not in dirs:
                dirs = np.insert(dirs, i, d) * dirs.units
                resp = np.insert(resp, i, 0) * 1/s

    rad_dirs = dirs.rescale(rad)

    # Plot response to each directions on polar plot.
    if plot_type == 'bar':  # sector plot
        ndirs = all_dirs.size
        left_rad_dirs = rad_dirs - np.pi/ndirs  # no need for this in MPL 2.0?
        w = 2*np.pi / ndirs                     # same with edgecolor and else?
        pplot.bars(left_rad_dirs, resp, width=w, alpha=0.50, color=color, lw=1,
                   edgecolor='w', title=title, ytitle=1.08, ax=ax)
    else:  # line plot
        rad_dirs, resp = [np.append(v, [v[0]]) for v in (rad_dirs, resp)]
        pplot.lines(rad_dirs, resp, color=color,  marker='o', lw=1, ms=4,
                    mew=0, title=title, ytitle=1.08, ax=ax)
        ax.fill(rad_dirs, resp, color=color, alpha=0.15)

    # Add arrow representing PD and weighted DSI.
    if DSI is not None and PD is not None:
        rho = np.max(resp) * DSI
        xy = (float(PD.rescale(rad)), rho)
        arr_props = dict(facecolor=color, edgecolor='k', shrink=0.0, alpha=0.5)
        ax.annotate('', xy, xytext=(0, 0), arrowprops=arr_props)

    # Remove spines and tick marks, maximize tick labels.
    putil.set_spines(ax, False, False)
    putil.hide_tick_marks(ax)

    # Save and return plot.
    putil.save_fig(ffig=ffig)
    return ax


def plot_tuning(xfit, yfit, vals=None, meanr=None, semr=None, color='b',
                baseline=None, xticks=None, xlim=None, ylim=None, xlab=None,
                ylab=None, title=None, ffig=None, ax=None, **kwargs):
    """Plot tuning curve, optionally with data samples."""

    ax = putil.axes(ax)

    # Plot baseline.
    if baseline is not None:
        putil.add_baseline(baseline, ax=ax)

    # Plot fitted curve.
    pplot.lines(xfit, yfit, color=color, ax=ax)

    # Plot data samples.
    if meanr is not None and semr is not None:
        pplot.errorbar(vals, meanr, yerr=semr, fmt='o', color=color,
                       ax=ax, **kwargs)

    # Set x axis ticks.
    if xticks is not None:
        putil.set_xtick_labels(ax, xticks)
    elif vals is not None:
        putil.set_xtick_labels(ax, vals)

    # Format plot.
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title)

    # Save and return plot.
    putil.save_fig(ffig=ffig)
    return ax
