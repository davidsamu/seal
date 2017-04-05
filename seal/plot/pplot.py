#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of general purpose plotting functions.

@author: David Samu
"""

import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns

from seal.util import util
from seal.plot import putil


def scatter(x, y, is_sign=None, c='b', bc='w', alpha=0.5, xlim=None,
            ylim=None, xlab=None, ylab=None, title=None, ytitle=None,
            polar=False, ffig=None, ax=None, **kwargs):
    """
    Plot paired data on scatter plot.
    Color settings:
        - c:  face color of foreground points (is_sign == True)
        - bc: face color of background points (is_sign == False)
    """

    # Init.
    ax = putil.axes(ax, polar=polar)
    cols = c
    if (is_sign is not None) and isinstance(c, str) and isinstance(bc, str):
        cols = putil.get_cmat(is_sign, c, bc)  # get point-specific color array

    ax.scatter(x, y, c=cols, alpha=alpha, **kwargs)  # plot colored points

    # Format and save figure.
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title, ytitle)
    putil.save_fig(ffig)

    return ax


def joint_scatter(x, y, is_sign=None, kind='reg', stat_func=util.pearson_r,
                  c='b', xlim=None, ylim=None, xlab=None, ylab=None,
                  title=None, ytitle=None, ffig=None, **kwargs):
    """
    Plot paired data on scatter plot with
        - marginal distributions added to the side
        - linear regression on center scatter
        - N, r and p values reported

    Additional parameters for scatter (e.g. size) can be passed as kwargs.
    """

    # Create scatter plot and distribution plots on the side.
    g = sns.jointplot(x, y, color=c, kind=kind, stat_func=stat_func,
                      xlim=xlim, ylim=ylim)
    ax = g.ax_joint  # scatter axes

    # Make non-significant points hollow (white face color).
    if is_sign is not None or kwargs is not None:
        ax.collections[0].set_visible(False)    # hide scatter points
        scatter(x, y, is_sign, c=c, ax=ax, **kwargs)

    # Add N to legend.
    leg_txt = g.ax_joint.get_legend().texts[0]
    new_txt = 'n = {}\n'.format(len(x)) + leg_txt.get_text()
    leg_txt.set_text(new_txt)

    # Format and save figure.
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title, ytitle)
    putil.save_fig(ffig)

    return ax


def lines(x, y, ylim=None, xlim=None, xlab=None, ylab=None, title=None,
          ytitle=None, polar=False, ffig=None, ax=None, **kwargs):
    """Plot simple lines."""

    # Plot data.
    ax = putil.axes(ax, polar=polar)
    ax.plot(x, y, **kwargs)

    # Format and save figure.
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title, ytitle)
    putil.save_fig(ffig)

    return ax


def band(x, ylower, yupper, ylim=None, xlim=None, xlab=None, ylab=None,
         title=None, ytitle=None, polar=False, ffig=None, ax=None, **kwargs):
    """Plot highlighted band area."""

    # Plot data.
    ax = putil.axes(ax, polar=polar)
    ax.fill_between(x, ylower, yupper, **kwargs)

    # Format and save figure.
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title, ytitle)
    putil.save_fig(ffig)

    return ax


def mean_err(x, ymean, ystd, ylim=None, xlim=None, xlab=None, ylab=None,
             title=None, ytitle=None, polar=False, ffig=None, ax=None,
             mean_kws=None, band_kws=None):
    """Plot mean and highlighted band area around it."""

    # Init params.
    if mean_kws is None:
        mean_kws = dict()
    if band_kws is None:
        band_kws = dict()

    # Init data.
    ylower = ymean - ystd
    yupper = ymean + ystd

    # Plot data.
    ax = putil.axes(ax, polar=polar)
    lines(x, ymean, ax=ax, **mean_kws)
    band(x, ylower, yupper, ax=ax, **band_kws)

    # Format and save figure.
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title, ytitle)
    putil.save_fig(ffig)

    return ax


def bars(x, y, ylim=None, xlim=None, xlab=None, ylab=None, title=None,
         ytitle=None, polar=False, ffig=None, ax=None, **kwargs):
    """Plot bar plot."""

    # Plot data.
    ax = putil.axes(ax, polar=polar)
    ax.bar(x, y, **kwargs)

    # Format and save figure.
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title, ytitle)
    putil.save_fig(ffig)

    return ax


def errorbar(x, y, yerr, ylim=None, xlim=None, xlab=None, ylab=None,
             title=None, ytitle=None, polar=False, ffig=None, ax=None,
             **kwargs):
    """Plot error bars."""

    # Plot data.
    ax = putil.axes(ax, polar=polar)
    ax.errorbar(x, y, yerr, **kwargs)

    # Format and save figure.
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title, ytitle)
    putil.save_fig(ffig)

    return ax


def cat_hist(vals, xlim=None, ylim=None, xlab=None, ylab='n', title=None,
             ytitle=None, polar=False, ffig=None, ax=None, **kwargs):
    """Plot histogram of categorical data."""

    # Plot data.
    ax = putil.axes(ax, polar=polar)
    ax = sns.countplot(vals, ax=ax)

    # Format and save figure.
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title, ytitle)
    putil.save_fig(ffig)

    return ax


def multi_hist(vals, xlim=None, ylim=None, xlab=None, ylab='n', title=None,
               ytitle=None, polar=False, ffig=None, ax=None, **kwargs):
    """Plot histogram of multiple samples side by side."""

    # Plot data.
    ax = putil.axes(ax, polar=polar)
    ax.hist(vals, **kwargs)

    # Format and save figure.
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title, ytitle)
    putil.save_fig(ffig)

    return ax


def heatmap(mat, vmin=None, vmax=None, cmap=None, cbar=True, cbar_ax=None,
            annot=None, square=False, xlab=None, ylab=None, title=None,
            ytitle=None, xlim=None, ylim=None, xticklabels=True,
            yticklabels=True, ffig=None, ax=None):
    """Plot rectangular data as heatmap."""

    # Plot data.
    ax = putil.axes(ax)
    sns.heatmap(mat, vmin, vmax, cmap, annot=annot, cbar=cbar, cbar_ax=cbar_ax,
                square=square, xticklabels=xticklabels,
                yticklabels=yticklabels, ax=ax)

    # Format and save figure.
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title, ytitle)
    putil.save_fig(ffig)

    return ax


def plot_task_violin(res, tasks, x, y, npval=None, pth=0.01, color='grey',
                     ylim=None, ylab=None, ffig=None):
    """Plot results on violin plots."""

    # Test difference from zero in each task.
    ttest_res = {task: sp.stats.ttest_1samp(tres[y], 0)
                 for task, tres in res.groupby(x)}
    ttest_res = pd.DataFrame.from_dict(ttest_res, 'index')

    # Binarize significance test.
    res['is_sign'] = res[npval] < pth if npval is not None else True
    res['direction'] = np.sign(res[y])

    # Set up figure and plot data.
    fig = putil.figure()
    ax = putil.axes()
    putil.add_baseline(ax=ax)
    sns.violinplot(x=x, y=y, data=res, inner=None, order=tasks, ax=ax)
    sns.swarmplot(x=x, y=y, hue='is_sign', data=res, color=color,
                  order=tasks, hue_order=[True, False], ax=ax)
    putil.set_labels(ax, ylab=ylab)
    putil.set_limits(ax, ylim=ylim)
    putil.hide_legend(ax)

    # Add annotations.
    ymin, ymax = ax.get_ylim()
    ylvl = ymax
    for i, task in enumerate(tasks):
        tres = res.loc[res.task == task]
        # Mean.
        mean_str = 'Mean:\n' if i == 0 else '\n'
        mean_str += '{:.2f}'.format(tres[y].mean())
        # Non-zero test of distribution.
        str_pval = util.format_pvalue(ttest_res.loc[task, 'pvalue'])
        mean_str += '\n({})'.format(str_pval)
        # Stats on difference from baseline.
        nnonsign, ntot = (~tres.is_sign).sum(), len(tres)
        npos, nneg = [sum(tres.is_sign & (tres.direction == d))
                      for d in (1, -1)]
        sign_to_report = [('+', npos), ('=', nnonsign), ('-', nneg)]
        nsign_str = ''
        for symb, n in sign_to_report:
            prc = str(int(round(100*n/ntot)))
            nsign_str += '\n{} {:>3} / {} ({:>2}%)'.format(symb, int(n),
                                                           ntot, prc)
        lbl = '{}\n\n{}'.format(mean_str, nsign_str)
        ax.text(i, ylvl, lbl, fontsize='smaller', va='bottom', ha='center')

    # Save plot.
    putil.save_fig(ffig, fig)

    return fig, ax
