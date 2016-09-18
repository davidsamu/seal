# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 16:27:05 2016

Collection of plotting function.

@author: David Samu
"""

import numpy as np
from scipy.stats.stats import pearsonr
from quantities import ms

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from matplotlib import collections as mc

from seal.util import util


# %% Matplotlib setup
plt.style.use('classic')

# Some para settings.
mpl.rc('font', size=8)  # default text size
mpl.rc('legend', fontsize='small')
mpl.rc(('xtick', 'ytick'), labelsize='small')
mpl.rc('axes', labelsize='medium', titlesize='x-large')
mpl.rc('figure', autolayout=True)  # prevent object being cut off
mpl.rc('figure', dpi=80, figsize=(6, 4), facecolor='white')
mpl.rc('savefig', dpi=100, facecolor='white')

# Need to set this separately to keep IPython inline plots small size.
savefig_dpi = 150

# To restore matplotlib default settings
# matplotlib.rcdefaults()


# %% Raster and rate plotting functions.

def raster_rate(spikes_list, rates, times, t1, t2, names,
                t_unit=ms, segments=None,
                pvals=None, ylim=None, title=None, xlab='Time (ms)',
                ylab_rate='Firing rate (sp/s)', markersize=1.5,
                legend=True, nlegend=True, ffig=None, fig=None, outer_gs=None):
    """Plot raster and rate plots."""

    if fig is None:
        fig = plt.figure()

    # Create subplots (as nested gridspecs).
    if outer_gs is None:
        outer_gs = gs.GridSpec(2, 1, height_ratios=[1, 1])
    gs_raster = gs.GridSpecFromSubplotSpec(len(spikes_list), 1,
                                           subplot_spec=outer_gs[0],
                                           hspace=.15)
    gs_rate = gs.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_gs[1])
    ylab_posx = -0.05

    # Raster plot(s).
    for i, sp_tr in enumerate(spikes_list):
        ax = fig.add_subplot(gs_raster[i, 0])
        ttl = title if i == 0 else None  # add title to top raster
        raster(sp_tr, t1, t2, t_unit, segments, markersize,
               title=ttl, xlab=None, ylab=names[i], ax=ax)
        ax.set_xticklabels([])
        ax.get_yaxis().set_label_coords(ylab_posx, 0.5)

    # Rate plot.
    ax = fig.add_subplot(gs_rate[0, 0])
    rate(rates, times, t1, t2, names, t_unit, segments, pvals,  ylim,
         title=None, xlab=xlab, ylab=ylab_rate, legend=legend, ax=ax)
    ax.get_yaxis().set_label_coords(ylab_posx, 0.5)

    # Save and return plot.
    save_fig(fig, ffig)
    return fig


def raster(spikes, t1, t2, t_unit=ms, segments=None,
           markersize=1.5, title=None, xlab='Time (ms)', ylab=None,
           ffig=None, ax=None):
    """Plot rasterplot."""

    if ax is None:
        ax = plt.gca()

    # Plot raster.
    plot_segments(segments, t_unit, ax=ax)
    for i, sp_tr in enumerate(spikes):
        t = sp_tr.rescale(t_unit)
        ax.plot(t, (i+1) * np.ones_like(t), 'k.',
                markersize=markersize, alpha=0.33)

    # Format plot.
    xlim = [t1.rescale(t_unit), t2.rescale(t_unit)]
    ylim = [0.5, len(spikes)+0.5]
    set_limits(xlim, ylim, ax=ax)
    ax.locator_params(axis='y', nbins=6)
    show_ticks(xtick_pos='none', ytick_pos='none', ax=ax)
    show_spines(False, False, False, False, ax=ax)
    set_labels(title, xlab, ylab, ax=ax)

    # Order trials from top to bottom (has to happen after setting axis limits)
    ax.invert_yaxis()

    # Save and return plot.
    save_fig(plt.gcf(), ffig)
    return ax


def rate(rates_list, time, t1, t2, names=None, t_unit=ms,
         segments=None, pvals=None, ylim=None, title=None, xlab='Time (ms)',
         ylab='Firing rate (sp/s)', legend=True, ffig=None, ax=None):
    """Plot firing rate."""

    if ax is None:
        ax = plt.gca()

    # Plot rate(s).
    plot_segments(segments, t_unit, ax=ax)
    for name, rts in zip(names, rates_list):

        # Calculate mean, SEM and scale time vector.
        meanr, semr = util.mean_rate(rts)
        time = time.rescale(t_unit)

        # Set line label.
        lbl = '{} (n={} trials)'.format(name, rts.shape[0])

        # Plot mean line and SEM area.
        mean_line = ax.plot(time, meanr, label=lbl)
        ax.fill_between(time, meanr-semr, meanr+semr, alpha=0.2,
                        facecolor=mean_line[0].get_color())

    # Format plot.
    ax.locator_params(axis='y', nbins=6)
    xlim = [t1.rescale(t_unit), t2.rescale(t_unit)]
    set_limits(xlim, ylim, ax=ax)
    show_ticks(xtick_pos='none', ytick_pos='none', ax=ax)
    show_spines(True, False, False, False, ax=ax)
    set_legend(ax, legend)
    set_labels(title, xlab, ylab, ax=ax)

    # Add significance line to top of plot.
    if pvals is not None and len(rates_list) == 2:
        lws = [2.0 * i for i in range(1, len(pvals)+1)]
        cols = len(pvals) * ['c']  # ['m', 'c', 'g', 'y', 'r']
        for pval, lw, col in zip(pvals, lws, cols):
            plot_significant_intervals(rates_list[0], rates_list[1], time,
                                       pval, ypos=ax.get_ylim()[1], color=col,
                                       linewidth=lw, ax=ax)

    # Save and return plot.
    save_fig(plt.gcf(), ffig)
    return ax


# %% Generic plot decorator functions.

def plot_significant_intervals(rates1, rates2, time, pval, ypos=0,
                               color='c', linewidth=4, ax=None):
    """Add significant intervals to axes."""

    if ax is None:
        ax = plt.gca()

    # Get intervals of significant differences between rates.
    sign_periods = util.sign_periods(rates1, rates2, time, pval)

    # Assamble line segments and add them to axes.
    line_segments = [[(t1, ypos), (t2, ypos)] for t1, t2 in sign_periods]
    # 'm', 'w', 'b', 'y', 'k', 'r', 'c' or 'g'
    col = list(mpl.colors.ColorConverter.colors[color[0]])  # RGB
    cols = len(sign_periods) * [col + [1]]  # list of RGBA
    lc = mc.LineCollection(line_segments, colors=cols, linewidth=linewidth)
    ax.add_collection(lc)


def plot_segments(segments, t_unit=ms, alpha=0.2, color='grey', ax=None):
    """Plot all segments of unit."""

    if segments is None:
        return

    if ax is None:
        ax = plt.gca()
    for key, (t_start, t_stop) in segments.items():
        t1 = t_start.rescale(t_unit)
        t2 = t_stop.rescale(t_unit)
        ax.axvspan(t1, t2, alpha=alpha, color=color)


def plot_events(events, t_unit=ms, add_names=True, alpha=1.0,
                color='black', lw=1, ax=None, **kwargs):
    """Plot all events of unit."""

    if ax is None:
        ax = plt.gca()

    # Add each event to plot as a vertical line.
    for key, time in events.items():
        time = time.rescale(t_unit)
        ymax = 0.96 if add_names else 1
        ax.axvline(time, color=color, alpha=alpha, lw=lw, ymax=ymax, **kwargs)

        # Add event label if requested
        if add_names:
            ylim = ax.get_ylim()
            yloc = ylim[0] + 0.98 * (ylim[1] - ylim[0])
            ax.text(time, yloc, key, rotation=90, fontsize='small',
                    verticalalignment='bottom', horizontalalignment='center')


# %% Miscellanous plot setup functions.

def set_limits(xlim=None, ylim=None, ax=None):
    """Generic function to set limits on axes."""

    if ax is None:
        ax = plt.gca()
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def set_labels(title=None, xlab=None, ylab=None, ax=None):
    """Generic function to set title, labels and ticks on axes."""

    if ax is None:
        ax = plt.gca()
    if title is not None:
        ax.set_title(title, y=1.04)
    if xlab is not None:
        ax.set_xlabel(xlab)
    if ylab is not None:
        ax.set_ylabel(ylab)
    ax.tick_params(axis='both', which='major')


def show_spines(bottom=True, left=True, top=False, right=False, ax=None):
    """Remove selected spines (axis lines) from current axes."""

    if ax is None:
        ax = plt.gca()
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['left'].set_visible(left)
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)


def show_ticks(xtick_pos='bottom', ytick_pos='left', ax=None):
    """Remove selected ticks from current axes.
       xtick_pos: [ 'bottom' | 'top' | 'both' | 'default' | 'none' ]
       ytick_pos: [ 'left' | 'right' | 'both' | 'default' | 'none' ]
    """

    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_ticks_position(xtick_pos)
    ax.yaxis.set_ticks_position(ytick_pos)


def colorbar(cbar, fig, cax, cb_title, cb_outline=False):
    """Add colorbar to figure."""

    cb = None
    if cbar:
        cb = fig.colorbar(cax, label=cb_title)
        cb.outline.set_visible(cb_outline)
    return cb


def set_legend(ax, add_legend=True):
    """Add legend to axes."""

    if add_legend:
        legend = ax.legend(loc='upper right', frameon=False)
    return legend


def save_fig(fig, ffig=None, close=True, dpi=savefig_dpi):
    """Save figure to file."""

    if ffig is None:
        return

    util.create_dir(ffig)
    fig.savefig(ffig, dpi=dpi)

    if close:
        plt.close(fig)


def figure(fig=None, **kwargs):
    """Return new figure instance."""

    if fig is None:
        fig = plt.figure(**kwargs)
    return fig


# %% General purpose plotting functions.

def scatter(x, y, xlim=None, ylim=None, xlab=None, ylab=None, title=None,
            add_r=False, ffig=None, ax=None, **kwargs):
    """Plot scatter plot of two vectors."""

    if ax is None:
        ax = plt.gca()

    ax.scatter(x, y, **kwargs)

    # Add correlation test results.
    if add_r:
        r, p = pearsonr(x, y)
        r_text = 'r = {:.2f} ({})'.format(r, util.format_pvalue(p))
        ax.text(0.95, 0.05, r_text, transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='bottom')

    # Format plot.
    set_limits(xlim, ylim, ax)
    show_ticks(xtick_pos='none', ytick_pos='none', ax=ax)
    show_spines(True, True, False, False, ax=ax)
    set_labels(title, xlab, ylab, ax)

    # Save and return plot.
    save_fig(plt.gcf(), ffig)
    return ax


def lines(y, x=None, ylim=None, xlim=None, xlab=None, ylab=None, title=None,
          ffig=None, ax=None, **kwargs):
    """Plot simple lines."""

    if ax is None:
        ax = plt.gca()

    if x is None:
        x = range(len(y))

    ax.plot(x, y, **kwargs)

    # Format plot.
    set_limits(xlim, ylim, ax)
    show_ticks(xtick_pos='none', ytick_pos='none', ax=ax)
    show_spines(True, False, False, False, ax=ax)
    set_labels(title, xlab, ylab, ax=ax)

    # Save and return plot.
    save_fig(plt.gcf(), ffig)
    return ax


def histogram(vals, xlim=None, ylim=None, xlab=None, ylab=None,
              title=None, ffig=None, ax=None, **kwargs):
    """Plot histogram."""

    if ax is None:
        ax = plt.gca()

    vals = vals[~np.isnan(vals)]  # remove NANs
    ax.hist(vals, **kwargs)

    # Format plot.
    set_limits(xlim, ylim, ax)
    show_ticks(xtick_pos='none', ytick_pos='none', ax=ax)
    show_spines(True, False, False, False, ax=ax)
    set_labels(title, xlab, ylab, ax=ax)

    # Save and return plot.
    save_fig(plt.gcf(), ffig)
    return ax


def histogram2D(x, y, nbins=100, hist_type='hexbin', cmap='viridis',
                xlim=None, ylim=None, xlab=None, ylab=None, title=None,
                ax=None, fig=None, ffig=None, **kwargs):
    """Plot 2D histogram."""

    # TODO: refactor figure and axis INIT and FIG SAVE throughout!

    # Plot either as hist2d or hexbin
    if xlim is None:
        xlim = [np.min(x), np.max(x)]
    if ylim is None:
        ylim = [np.min(y), np.max(y)]

    if hist_type == 'hist2d':
        limits = np.array([xlim, ylim])
        ax.hist2d(x, y, bins=nbins, cmap=cmap, range=limits, **kwargs)
    else:
        ax.hexbin(x, y, gridsize=nbins, cmap=cmap, **kwargs)

    # Add colorbar.
    # TODO: should/could be moved to some wrapper function?

    # Format plot.
    set_limits(xlim, ylim, ax)
    show_ticks(xtick_pos='none', ytick_pos='none', ax=ax)
    show_spines(False, False, False, False, ax=ax)
    set_labels(title, xlab, ylab, ax=ax)

    # Save and return plot.
    save_fig(fig, ffig)
    return fig, ax


def heatmap(mat, tvec, t_unit=ms, t1=None, t2=None, vmin=None, vmax=None,
            title=None, xlab='Time (ms)', ylab='Unit number', cmap='viridis',
            cbar=True, cb_title=None, ax=None, fig=None, ffig=None):
    """Plot 2D matrix as heatmap."""

    # Set up params.
    if t1 is None:
        t1 = tvec[0]
    if t2 is None:
        t2 = tvec[-1]

    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Plot raster.
    t_idx = util.indices_in_window(tvec, t1, t2)
    X = tvec[t_idx].rescale(t_unit)
    Y = range(mat.shape[0]+1)
    C = mat[:, t_idx]
    cax = ax.pcolormesh(X, Y, C, cmap=cmap, vmin=vmin, vmax=vmax)

    # Add colorbar.
    cb = colorbar(cbar, fig, cax, cb_title)

    # Format plot.
    set_limits([min(X), max(X)], [min(Y), max(Y)], ax)
    show_ticks(xtick_pos='none', ytick_pos='none', ax=ax)
    show_spines(False, False, False, False, ax=ax)
    set_labels(title, xlab, ylab, ax=ax)

    # Save and return plot.
    save_fig(fig, ffig)
    return fig, ax, cb
