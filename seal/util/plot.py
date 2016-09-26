# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 16:27:05 2016

Collection of plotting function.

@author: David Samu
"""

import warnings
from itertools import cycle
from collections import Counter

import numpy as np
from scipy.stats.stats import pearsonr
from quantities import ms, rad

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from matplotlib import collections as mc
from matplotlib import dates as md

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


# %% Functions to plot group level properties.

def group_params(unit_params, params_to_plot=None, ffig=None):
    """Plot histogram of parameter values across units."""

    # Init params to plot.
    if params_to_plot is None:
        params_to_plot = [  # Session parameters
                          'date', 'channel #', 'unit #', 'sort #',
                          # Quality metrics
                          'MeanWfAmplitude', 'MeanWfDuration (us)', 'SNR',
                          'MeanFiringRate (sp/s)', 'ISIviolation (%)',
                          'TrueSpikes (%)', 'UnitType', 'total # trials',
                          '# rejected trials', '# remaining trials',
                          # Direction selectivity metrics
                          'DSI S1', 'DSI S2 (deg)',
                          'PD S1 (deg)', 'PD S2 (deg)',
                          'PD8 S1 (deg)', 'PD8 S2 (deg)'
                         ]
    categorical = ['date', 'channel #', 'unit #', 'sort #', 'UnitType',
                   'PD8 S1 (deg)', 'PD8 S2 (deg)']

    # Init figure.
    fig, axs = get_subplots(len(params_to_plot))

    # Plot distribution of each parameter.
    colors = get_colors()
    for pp, ax in zip(params_to_plot, axs):
        v = unit_params[pp]
        if pp in categorical:
            v = np.array(v, dtype='object')
        histogram(v, xlab=pp, ylab='n', title=pp, color=next(colors), ax=ax)

    # Add main title
    fig_title = ('Distributions of session parameters, quality metrics ' +
                 'and stimulus response properties across units' +
                 ' (n = {})'.format(unit_params.shape[0]))
    fig.suptitle(fig_title, y=1.06, fontsize='xx-large')

    # Save and return plot.
    save_fig(fig, ffig)
    return fig


# %% Functions to plot basic unit activity (raster, rate, tuning curve, etc).

def raster_rate(spikes_list, rates, times, t1, t2, names,
                t_unit=ms, segments=None,
                pvals=None, ylim=None, title=None, xlab='Time (ms)',
                ylab_rate='Firing rate (sp/s)', markersize=1.5,
                legend=True, nlegend=True, ffig=None, fig=None, outer_gs=None):
    """Plot raster and rate plots."""

    # Create subplots (as nested gridspecs).
    fig = figure(fig)
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

    # Plot raster.
    ax = axes(ax)
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
    show_spines(True, True, True, True, ax=ax)
    set_labels(title, xlab, ylab, ax=ax)

    # Add '1' to tick labels
    # ax.yaxis.set_ticks([1] + ax.yaxis.get_majorticklocs())

    # Order trials from top to bottom (has to happen after setting axis limits)
    ax.invert_yaxis()

    # Save and return plot.
    save_fig(plt.gcf(), ffig)
    return ax


def rate(rates_list, time, t1, t2, names=None, t_unit=ms,
         segments=None, pvals=None, ylim=None, title=None, xlab='Time (ms)',
         ylab='Firing rate (sp/s)', legend=True, ffig=None, ax=None):
    """Plot firing rate."""

    # Plot rate(s).
    ax = axes(ax)
    plot_segments(segments, t_unit, ax=ax)
    for name, rts in zip(names, rates_list):

        # Calculate mean, SEM and scale time vector.
        meanr, semr = util.mean_rate(rts)
        time = time.rescale(t_unit)

        # Set line label.
        lbl = '{} (n={} trials)'.format(name, rts.shape[0])

        # Plot mean line and SEM area.
        line_col = ax.plot(time, meanr, label=lbl)[0].get_color()
        ax.fill_between(time, meanr-semr, meanr+semr, alpha=0.2,
                        facecolor=line_col, edgecolor=line_col)

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


def direction_selectivity(dirs, activity, ds_idx=None, pref_dir=None,
                          color='g', title=None, ffig=None, ax=None):
    """Plot direction selectivity."""

    # Plot response to each directions on polar plot.
    rad_dirs = np.array([d.rescale(rad) for d in dirs])
    ndirs = rad_dirs.size
    left_rad_dirs = rad_dirs - np.pi/ndirs
    w = 2*np.pi / ndirs
    ax = bars(activity, left_rad_dirs, polar=True, width=w, alpha=0.50,
              color=color, edgecolor='w', linewidth=1, title=title,
              ytitle=1.08, ax=ax)

    # Add arrow representing preferred direction and direction selectivity.
    if ds_idx is not None and pref_dir is not None:
        rho = np.max(activity) * ds_idx
        xy = (float(pref_dir.rescale(rad)), rho)
        ax.annotate('', xy, xytext=(0, 0),
                    arrowprops=dict(facecolor=color, edgecolor='k',
                                    shrink=0.0, alpha=0.5))

    # Save and return plot.
    save_fig(plt.gcf(), ffig)
    return ax


# %% Generic plot decorator functions.

def plot_significant_intervals(rates1, rates2, time, pval, ypos=0,
                               color='c', linewidth=4, ax=None):
    """Add significant intervals to axes."""

    ax = axes(ax)

    # Get intervals of significant differences between rates.
    sign_periods = util.sign_periods(rates1, rates2, time, pval)

    # Assamble line segments and add them to axes.
    line_segments = [[(t1, ypos), (t2, ypos)] for t1, t2 in sign_periods]
    # 'm', 'w', 'b', 'y', 'k', 'r', 'c' or 'g'
    col = list(mpl.colors.ColorConverter.colors[color[0]])  # RGB
    cols = len(sign_periods) * [col + [1]]  # list of RGBA
    lc = mc.LineCollection(line_segments, colors=cols, linewidth=linewidth)
    ax.add_collection(lc)


def plot_segments(segments, t_unit=ms, alpha=0.2, color='grey',
                  ax=None, **kwargs):
    """Plot all segments of unit."""

    if segments is None:
        return

    ax = axes(ax)
    for key, (t_start, t_stop) in segments.items():
        t1 = t_start.rescale(t_unit)
        t2 = t_stop.rescale(t_unit)
        ax.axvspan(t1, t2, alpha=alpha, color=color, **kwargs)


def plot_events(events, t_unit=ms, add_names=True, alpha=1.0,
                color='black', lw=1, ax=None, **kwargs):
    """Plot all events of unit."""

    ax = axes(ax)

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


# %% Plot setup functions.

def set_limits(xlim=None, ylim=None, ax=None):
    """Generic function to set limits on axes."""

    ax = axes(ax)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def set_labels(title=None, xlab=None, ylab=None, ytitle=None, ax=None):
    """Generic function to set title, labels and ticks on axes."""

    ax = axes(ax)
    if title is not None:
        ytitle = ytitle if ytitle is not None else 1.04
        ax.set_title(title, y=ytitle)
    if xlab is not None:
        ax.set_xlabel(xlab)
    if ylab is not None:
        ax.set_ylabel(ylab)
    ax.tick_params(axis='both', which='major')


def show_spines(bottom=True, left=True, top=False, right=False, ax=None):
    """Remove selected spines (axis lines) from current axes."""

    ax = axes(ax)
    if 'polar' in ax.spines:  # Polar coordinate
        ax.spines['polar'].set_visible(top)

    else:  # Cartesian coordinate
        ax.spines['bottom'].set_visible(bottom)
        ax.spines['left'].set_visible(left)
        ax.spines['top'].set_visible(top)
        ax.spines['right'].set_visible(right)


def show_ticks(xtick_pos='bottom', ytick_pos='left', ax=None):
    """Remove selected ticks from current axes.
       xtick_pos: [ 'bottom' | 'top' | 'both' | 'default' | 'none' ]
       ytick_pos: [ 'left' | 'right' | 'both' | 'default' | 'none' ]
    """

    ax = axes(ax)
    ax.xaxis.set_ticks_position(xtick_pos)
    ax.yaxis.set_ticks_position(ytick_pos)


def colorbar(cbar, fig, cax, cb_title, cb_outline=False):
    """Add colorbar to figure."""

    cb = None
    if cbar:
        cb = fig.colorbar(cax, label=cb_title)
        cb.outline.set_visible(cb_outline)
    return cb


def set_legend(ax, add_legend=True, loc='upper right',
               frameon=False, **kwargs):
    """Add legend to axes."""

    if add_legend:
        legend = ax.legend(loc=loc, frameon=frameon, **kwargs)

    return legend


def save_fig(fig=None, ffig=None, close=True, bbox_extra_artists=None,
             dpi=savefig_dpi):
    """Save figure to file."""

    if ffig is None:
        return

    if fig is None:
        fig = plt.gcf()

    util.create_dir(ffig)
    fig.savefig(ffig, bbox_extra_artists=bbox_extra_artists,
                dpi=dpi, bbox_inches='tight')

    if close:
        plt.close(fig)


def figure(fig=None, **kwargs):
    """Return new figure instance."""

    if fig is None:
        fig = plt.figure(**kwargs)
    return fig


def axes(ax=None, **kwargs):
    """Return new axes instance."""

    if ax is None:
        ax = plt.gca(**kwargs)
    return ax


def get_subplots(nplots, sp_width=4, sp_height=3):
    """Returns figures with given number of axes initialised."""

    # Create figure with axes arranged in grid pattern.
    nrow = int(np.floor(np.sqrt(nplots)))
    ncol = int(np.ceil(nplots / nrow))
    figsize = (sp_width*ncol, sp_height*nrow)
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)

    # Turn off last axes on grid.
    axsf = axs.flatten()
    [ax.axis('off') for ax in axsf[nplots:]]

    return fig, axsf


# %% Miscellanous plot related functions.

def get_colors(from_mpl_cycle=False, as_cycle=True):
    """Return colour cycle."""

    if from_mpl_cycle:
        col_cylce = mpl.rcParams['axes.prop_cycle']
        cols = [d['color'] for d in col_cylce]
    else:
        cols = ['b', 'g', 'r', 'c', 'm', 'y']

    if as_cycle:
        cols = cycle(cols)

    return cols


def get_proxy_patch(label, color):
    """Return patch proxy artist. Useful for creating custom legends."""

    patch = mpl.patches.Patch(color=color, label=label)
    return patch


# %% General purpose plotting functions.

def base_plot(x, y, xlim=None, ylim=None, xlab=None, ylab=None,
              title=None, ytitle=None, polar=False, figtype='lines',
              ffig=None, ax=None, **kwargs):
    """Generic plotting function."""

    ax = axes(ax, polar=polar)

    if figtype == 'lines':
        ax.plot(x, y, **kwargs)

    elif figtype == 'bars':
        ax.bar(x, y, **kwargs)

    elif figtype == 'hist':

        x = np.array(x)

        # Categorical data.
        if not util.is_numeric_array(x):
            counts = np.array(Counter(x).most_common())
            counts = counts[counts[:, 0].argsort()]  # sort by category
            cats, cnts = counts[:, 0], np.array(counts[:, 1], dtype=int)
            xx = range(len(cats))
            width = 0.7
            ax.bar(xx, cnts, width, **kwargs)
            ax.set_xticks([i+width/2 for i in xx])
            rot_x_labs = np.sum([len(str(cat)) for cat in cats]) > 60
            rot = 45 if rot_x_labs else 0
            ha = 'right' if rot_x_labs else 'center'
            ax.set_xticklabels(cats, rotation=rot, ha=ha)
        else:  # Plot Numeric data.
            x = x[~np.isnan(x)]  # remove NANs
            ax.hist(x, **kwargs)

    elif figtype == 'scatter':
        ax.scatter(x, y, **kwargs)

    else:
        warnings.warn('Unidentified figure type: {}'.format(figtype))

    # Format plot.
    set_limits(xlim, ylim, ax)
    show_ticks(xtick_pos='none', ytick_pos='none', ax=ax)
    show_spines(True, False, False, False, ax=ax)
    set_labels(title, xlab, ylab, ytitle, ax=ax)

    # Save and return plot.
    save_fig(plt.gcf(), ffig)
    return ax


def scatter(x, y, xlim=None, ylim=None, xlab=None, ylab=None,
            title=None, ytitle=None, polar=False, add_r=False,
            ffig=None, ax=None, **kwargs):
    """Plot two vectors on scatter plot."""

    # Plot scatter plot.
    ax = base_plot(x, y, xlim, ylim, xlab, ylab, title, ytitle, polar,
                   'scatter', ffig, ax=ax, **kwargs)

    # Add correlation test results.
    if add_r:
        r, p = pearsonr(x, y)
        r_text = 'r = {:.2f} ({})'.format(r, util.format_pvalue(p))
        ax.text(0.95, 0.05, r_text, transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='bottom')
    return ax


def lines(x, y, ylim=None, xlim=None, xlab=None, ylab=None, title=None,
          ytitle=None, polar=False, ffig=None, ax=None, **kwargs):
    """Plot simple lines."""

    # Plot line plot.
    ax = base_plot(x, y, xlim, ylim, xlab, ylab, title, ytitle, polar,
                   'lines', ffig, ax=ax, **kwargs)
    return ax


def bars(y, x=None, ylim=None, xlim=None, xlab=None, ylab=None, title=None,
         ytitle=None, polar=False, ffig=None, ax=None, **kwargs):
    """Plot simple lines."""

    # Plot bar plot.
    ax = base_plot(x, y, xlim, ylim, xlab, ylab, title, ytitle, polar,
                   'bars', ffig, ax=ax, **kwargs)
    return ax


def histogram(vals, xlim=None, ylim=None, xlab=None, ylab=None, title=None,
              ytitle=None, polar=False, ffig=None, ax=None, **kwargs):
    """Plot histogram."""

    # Plot histogram.
    ax = base_plot(vals, None, xlim, ylim, xlab, ylab, title, ytitle, polar,
                   'hist', ffig, ax=ax, **kwargs)
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

    fig = figure(fig)
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
