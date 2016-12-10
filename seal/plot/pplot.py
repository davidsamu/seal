#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 20:35:17 2016

Collection of general purpose plotting functions.

@author: David Samu
"""


import seaborn as sns

from seal.util import util
from seal.plot import putil


def scatter(x, y, is_sign, kind='reg', stat_func=util.pearson_r, c='b',
            xlim=None, ylim=None, xlab=None, ylab=None,
            title=None, ytitle=None, ffig=None):
    """Plot two vectors on scatter plot."""

    # Create scatter plot and distribution plots on the side.
    g = sns.jointplot(x, y, color=c, kind=kind, stat_func=stat_func,
                      xlim=xlim, ylim=ylim)
    ax = g.ax_joint  # scatter axes

    # Make non-significant points hollow (white face color).
    if is_sign is not None:
        ax.collections[0].set_visible(False)    # hide scatter points
        cols = putil.get_cmat(is_sign, c, 'w')  # point-specific color array
        ax.scatter(x, y, c=cols, edgecolors=c)  # add colored scatter points

    # Add N to legend.
    leg_txt = g.ax_joint.get_legend().texts[0]
    new_txt = 'n = {}\n'.format(len(x)) + leg_txt.get_text()
    leg_txt.set_text(new_txt)

    # Format and save figure.
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title, ytitle)
    putil.save_fig(ffig=ffig)

    return ax


def lines(x, y, ylim=None, xlim=None, xlab=None, ylab=None, title=None,
          ytitle=None, polar=False, ffig=None, ax=None, **kwargs):
    """Plot simple lines."""

    # Plot data.
    ax = putil.axes(ax, polar=polar)
    ax.plot(x, y, **kwargs)

    # Format and save figure.
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title, ytitle)
    putil.save_fig(ffig=ffig)

    return ax


def bars(x, y, ylim=None, xlim=None, xlab=None, ylab=None, title=None,
         ytitle=None, polar=False, ffig=None, ax=None, **kwargs):
    """Plot bar plot."""

    # Plot data.
    ax = putil.axes(ax, polar=polar)
    ax.bar(x, y, **kwargs)

    # Format and save figure.
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title, ytitle)
    putil.save_fig(ffig=ffig)

    return ax


def errorbar(x, y, yerr, ylim=None, xlim=None, xlab=None, ylab=None,
             title=None, ytitle=None, polar=False, ffig=None, ax=None,
             **kwargs):
    """Plot error bars."""

    # Plot data.
    ax = putil.axes(ax, polar=polar)
    ax.errorbar(x, y, **kwargs)

    # Format and save figure.
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title, ytitle)
    putil.save_fig(ffig=ffig)

    return ax


def cat_hist(vals, xlim=None, ylim=None, xlab=None, ylab='n', title=None,
             ytitle=None, polar=False, ffig=None, ax=None, **kwargs):
    """Plot histogram of categorical data."""

    # Plot data.
    ax = putil.axes(ax, polar=polar)
    ax = sns.countplot(vals, ax=ax)

    # Format and save figure.
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title, ytitle)
    putil.save_fig(ffig=ffig)

    return ax


def multi_hist(vals, xlim=None, ylim=None, xlab=None, ylab='n', title=None,
               ytitle=None, polar=False, ffig=None, ax=None, **kwargs):
    """Plot histogram of multiple samples side by side."""

    # Plot data.
    ax = putil.axes(ax, polar=polar)
    ax.hist(vals, **kwargs)

    # Format and save figure.
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title, ytitle)
    putil.save_fig(ffig=ffig)

    return ax


def heatmap(mat, vmin=None, vmax=None, cmap=None, cbar=True, cbar_ax=None,
            annot=None, square=False, xlab=None, ylab=None, title=None,
            ytitle=None, xlim=None, ylim=None, ffig=None, ax=None):
    """Plot rectangular data as heatmap."""

    # Plot data.
    ax = putil.axes(ax)
    sns.heatmap(mat, vmin, vmax, cmap, annot=annot, cbar=cbar, cbar_ax=cbar_ax,
                square=square, ax=ax)

    # Format and save figure.
    putil.format_plot(ax, xlim, ylim, xlab, ylab, title, ytitle)
    putil.save_fig(ffig=ffig)

    return ax
