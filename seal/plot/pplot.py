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

from seal.analysis import stats
from seal.util import util
from seal.plot import putil



def sign_hist(v, pvals=None, pth=0.01, bins=None, scol='g', nscol='k',
              ax=None):
    """Plot histogram of significant and non-significant values stacked."""

    # Init.
    ax = putil.axes(ax)
    vnonsig = pvals >= pth

    # Plot all values and then non-significant values only.
    sns.distplot(v, kde=False, bins=bins, color=scol, ax=ax)
    sns.distplot(v[vnonsig], kde=False, bins=bins, color=nscol, ax=ax)

    # Add vertical zero line.
    ax.axvline(0, color='gray', lw=1, ls='dashed')

    # Format plot.
    sns.despine(ax=ax)

    return ax


def sign_scatter(v1, v2, pvals=None, pth=0.01, scol='g', nscol='k',
                 id_line=False, fit_reg=False, ax=None):
    """Plot scatter plot with significant points highlighted."""

    # Init.
    ax = putil.axes(ax)
    s_pars = (True, scol, {'alpha': 1.0})
    ns_pars = (False, nscol, {'alpha': 0.8})

    # Binarize significance stats.
    vsig = (pvals < pth if pvals is not None
            else pd.Series(True, index=v1.index))

    # Plot significant and non-significant points.
    for b, c, a in [ns_pars, s_pars]:
        if (vsig==b).any():
            sns.regplot(v1.loc[vsig==b], v2.loc[vsig==b], fit_reg=fit_reg,
                        color=c, scatter_kws=a, ax=ax)

    # Format plot.
    sns.despine(ax=ax)

    # Add identity line.
    if id_line:
        v_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
        putil.set_limits(ax, [0, v_max], [0, v_max])
        putil.add_identity_line(ax=ax, equal_xy=True)

    return ax


def scatter(x, y, is_sign=None, c='b', bc='w', nc='grey', ec='k', alpha=0.5,
            xlim=None, ylim=None, xlab=None, ylab=None, title=None,
            ytitle=None, polar=False, id_line=True,
            ffig=None, ax=None, **kwargs):
    """
    Plot paired data on scatter plot.
    Color settings:
        - c:  face color of significant points (is_sign == True)
        - bc: face color of non-significant points (is_sign == False)
        - nc: face color of untested/untestable points (is_sign == None)
    """

    # Init.
    ax = putil.axes(ax, polar=polar)
    cols = c
    # Get point-specific color array.
    if (is_sign is not None) and isinstance(c, str) and isinstance(bc, str):
        cols = putil.get_cmat(is_sign, c, bc, nc)

    # Plot colored points.
    ax.scatter(x, y, c=cols, edgecolor=ec, alpha=alpha, **kwargs)

    # Add identity line.
    if id_line:
        putil.add_identity_line(equal_xy=True, zorder=99, ax=ax)

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


def cat_mean(df, x, y, add_stats=True, fstats=None, bar_ylvl=None, ci=68,
             add_mean=True, mean_ylvl=None, ylbl=None,
             fig=None, ax=None, ffig=None):
    """Plot mean of two categorical dataset."""

    # Init.
    if fig is None and ax is None:
        fig = putil.figure(figsize=(3, 3))
    if ax is None:
        ax = putil.axes()
    if fstats is None:
        fstats = stats.mann_whithney_u_test

    # Plot means as bars.
    sns.barplot(x=x, y=y, data=df, ci=ci, ax=ax, palette=palette,
                errwidth=errwidth, **kwargs)

    # Get plotted vectors.
    ngrps = [t.get_text() for t in ax.get_xticklabels()]
    v1, v2 = [df.loc[df[x] == ngrp, y] for ngrp in ngrps]

    # Add significance bar.
    if add_stats:
        _, pval = fstats(v1, v2)
        pval_str = util.format_pvalue(pval)
        if bar_ylvl is None:
            bar_ylvl = 1.1 * max(v1.mean()+stats.sem(v1),
                                  v2.mean()+stats.sem(v2))
        lines([0.1, 0.9], [bar_ylvl, bar_ylvl], color='grey', ax=ax)
        ax.text(0.5, 1.01*bar_ylvl, pval_str, fontsize='medium',
                fontstyle='italic', va='bottom', ha='center')

    # Add mean values.
    for vec, xpos in [(v1, 0.2), (v2, 1.2)]:
        mstr = '{:.2f}'.format(vec.mean())
        ypos = 1.005 * vec.mean()
        ax.text(xpos, ypos, mstr, fontstyle='italic', fontsize='smaller',
                va='bottom', ha='center')

    # Format plot.
    sns.despine()
    putil.hide_legend_title(ax)
    putil.set_labels(ax, '', ylbl)
    putil.sparsify_tick_labels(fig, ax, 'y', freq=2)
    # Save plot.
    putil.save_fig(ffig, fig)

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


def plot_group_violin(res, x, y, groups=None, npval=None, pth=0.01,
                      color='grey', ylim=None, ylab=None, ffig=None):
    """Plot group-wise results on violin plots."""

    if groups is None:
        groups = res['group'].unique()

    # Test difference from zero in each groups.
    ttest_res = {group: sp.stats.ttest_1samp(gres[y], 0)
                 for group, gres in res.groupby(x)}
    ttest_res = pd.DataFrame.from_dict(ttest_res, 'index')

    # Binarize significance test.
    res['is_sign'] = res[npval] < pth if npval is not None else True
    res['direction'] = np.sign(res[y])

    # Set up figure and plot data.
    fig = putil.figure()
    ax = putil.axes()
    putil.add_baseline(ax=ax)
    sns.violinplot(x=x, y=y, data=res, inner=None, order=groups, ax=ax)
    sns.swarmplot(x=x, y=y, hue='is_sign', data=res, color=color,
                  order=groups, hue_order=[True, False], ax=ax)
    putil.set_labels(ax, xlab='', ylab=ylab)
    putil.set_limits(ax, ylim=ylim)
    putil.hide_legend(ax)

    # Add annotations.
    ymin, ymax = ax.get_ylim()
    ylvl = ymax
    for i, group in enumerate(groups):
        gres = res.loc[res.group == group]
        # Mean.
        mean_str = 'Mean:\n' if i == 0 else '\n'
        mean_str += '{:.2f}'.format(gres[y].mean())
        # Non-zero test of distribution.
        str_pval = util.format_pvalue(ttest_res.loc[group, 'pvalue'])
        mean_str += '\n({})'.format(str_pval)
        # Stats on difference from baseline.
        nnonsign, ntot = (~gres.is_sign).sum(), len(gres)
        npos, nneg = [sum(gres.is_sign & (gres.direction == d))
                      for d in (1, -1)]
        sign_to_report = [('+', npos), ('=', nnonsign), ('-', nneg)]
        nsign_str = ''
        for symb, n in sign_to_report:
            prc = str(int(round(100*n/ntot)))
            nsign_str += '{} {:>3} / {} ({:>2}%)\n'.format(symb, int(n),
                                                           ntot, prc)
        lbl = '{}\n\n{}'.format(mean_str, nsign_str)
        ax.text(i, ylvl, lbl, fontsize='smaller', va='bottom', ha='center')

    # Save plot.
    putil.save_fig(ffig, fig)

    return fig, ax
