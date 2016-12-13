# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 16:27:05 2016

Collection of plotting-related utility functions.

@author: David Samu
"""

import warnings
from itertools import cycle
from collections import OrderedDict as OrdDict

import numpy as np
from quantities import ms

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from matplotlib import collections as mc

import seaborn as sns

from seal.util import util


# %% Matplotlib setup and some plotting constants.

mpl.rc('figure', autolayout=True)  # to prevent object being cut off

ColConv = mpl.colors.ColorConverter()

t_lbl = 'Time since S1 onset (ms)'
FR_lbl = 'Firing rate (sp/s)'

t_ticks = np.arange(-1000, 5000+1, 1000) * ms

my_color_list = ['m', 'g', 'r', 'c', 'b', 'y']


# %% Functions to plot group and unit level properties.

def group_params(unit_params, rec_name, ffig=None):
    """Plot histogram of parameter values across units."""

    # Init parameters to plot.
    to_skip = ['task_idx', 'filepath', 'filename', 'monkey', 'date', 'probe'
               'sampl_prd', 'DS', 'DSI', 'PD', 'TP']
    params = [p for p in unit_params.columns if p not in to_skip]

    # Init figure.
    fig, _, axs = get_gs_subplots(nrow=len(params), subw=4, subh=3,
                                  as_array=False)

#    for param, ax in zip(params, axs):
#        pplots.cat_hist(unit_params[param], ax=ax)

    # Add main title
    fig_title = ('{} session parameters, quality metrics '.format(rec_name) +
                 'and stimulus selectivity across units' +
                 ' (n = {})'.format(unit_params.shape[0]))
    fig.suptitle(fig_title, y=1.06, fontsize='xx-large')

    # Save and return plot.
    save_fig(fig, ffig)
    return fig


def unit_info(u, fs='large', ax=None):
    """Plot unit info as text labels."""

    # Init axes.
    ax = axes(ax)
    hide_axes(ax=ax)

    # Init dict of info labels to plot.
    upars = u.get_unit_params()
    SNR, mWFdur, mFR = [upars[meas] if meas in upars else None
                        for meas in ('SNR', 'MeanSpikeDuration', 'MeanFiringRate')]
    lbl_dict = OrdDict([('SNR', 'N/A' if SNR is None else '{:.2f}'.format(SNR)),
                        ('WfDur', 'N/A' if mWFdur is None else '{:.0f} $\mu$s'.format(mWFdur)),
                        ('FR', 'N/A' if mFR is None else '{:.1f} sp/s'.format(mFR))])

    # Plot each label.
    yloc = .0
    xloc = np.linspace(.10, .85, len(lbl_dict))
    for xloci, (lbl, val) in zip(xloc, lbl_dict.items()):
        lbl_str = '{}: {}'.format(lbl, val)
        ax.text(xloci, yloc, lbl_str, fontsize=fs, va='bottom', ha='center')

    # Set title.
    set_labels(ax, title=upars['task'], ytitle=.75,
               title_kws={'fontsize': 'x-large'})

    return ax


# %% Generic plot decorator functions.

def plot_signif_prds(rates1, rates2, time, pval, test, test_kws, ypos=0,
                     color='c', linewidth=4, ax=None):
    """Add significant intervals to axes."""

    ax = axes(ax)

    # Get intervals of significant differences between rates.
    sign_periods = util.sign_periods(rates1, rates2, time, pval,
                                     test, test_kws)

    # Assamble line segments and add them to axes.
    line_segments = [[(t1, ypos), (t2, ypos)] for t1, t2 in sign_periods]
    lc = mc.LineCollection(line_segments, colors=ColConv.to_rgba(color),
                           linewidth=linewidth)
    lc.sign_prd = True  # add label to find these LC objects for post-adjusting
    plt.gca().add_collection(lc)


def move_signif_lines(ax=None, ypos=None):
    """
    Move significance line segments to top of plot
    (typically after resetting y-limit, e.g. to match across set of axes).
    """

    # Init.
    ax = axes(ax)
    if ypos is None:
        ypos = ax.get_ylim()[1]  # move them to current top

    # Find line segments in axes representing significant periods.
    for c in ax.collections:
        if isinstance(c, mc.LineCollection) and hasattr(c, 'sign_prd'):
            segments = [np.array((seg[:, 0], seg.shape[0]*[ypos])).T
                        for seg in c.get_segments()]
            c.set_segments(segments)


def plot_periods(prds, t_unit=ms, alpha=0.2, color='grey', ax=None, **kwargs):
    """Highlight segments (periods)."""

    if prds is None:
        return

    ax = axes(ax)
    for name, (t_start, t_stop) in prds.periods().iterrows():
        if t_unit is not None:
            t_start = t_start.rescale(t_unit)
            t_stop = t_stop.rescale(t_unit)
        ax.axvspan(t_start, t_stop, alpha=alpha, color=color, **kwargs)


def plot_events(events, t_unit=ms, add_names=True, color='black', alpha=1.0,
                ls='--', lw=1, lbl_rotation=90, lbl_height=0.98,
                lbl_ha='center', ax=None, **kwargs):
    """Plot all events of unit."""

    ax = axes(ax)

    # Add each event to plot as a vertical line.
    for key, time in events.items():
        if t_unit is not None:
            time = time.rescale(t_unit)
        ymax = lbl_height-0.02 if add_names else 1
        ax.axvline(time, color=color, alpha=alpha, lw=lw, ls=ls,
                   ymax=ymax, **kwargs)

        # Add event label if requested
        if add_names:
            ylim = ax.get_ylim()
            yloc = ylim[0] + lbl_height * (ylim[1] - ylim[0])
            ax.text(time, yloc, key, rotation=lbl_rotation, fontsize='small',
                    va='bottom', ha=lbl_ha)


def add_chance_level(ylevel=0.5, color='grey', ls='--', alpha=0.5, ax=None):
    """Add horizontal line denoting chance level for decoder accuracy plot."""

    ax = axes(ax)
    ax.axhline(ylevel, color=color, ls=ls, alpha=alpha)


def add_zero_line(axis='both', color='grey', ls='--', alpha=0.5, ax=None):
    """Add zero line to x and/or y axes."""

    ax = axes(ax)
    if axis in ('x', 'both'):
        ax.axhline(0, color=color, ls=ls, alpha=alpha)
    if axis in ('y', 'both'):
        ax.axvline(0, color=color, ls=ls, alpha=alpha)


def add_identity_line(equal_xy=False, color='grey', ls='--', ax=None):
    """Add identity (x=y) line to axes."""

    ax = axes(ax)

    if equal_xy:   # Safer to use this option, if x and y axes are equalised.
        xymin, xymax = 0, 1
        transform = ax.transAxes

    else:  # Less safe because it breaks if axes limits are changed afterwards.
        [xmin, xmax] = ax.get_xlim()
        [ymin, ymax] = ax.get_ylim()
        xymin = max(xmin, ymin)
        xymax = min(xmax, ymax)
        transform = None

    xy = [xymin, xymax]
    ax.plot(xy, xy, ax=ax, color=color, ls=ls, transform=transform)


# %% Functions to adjust plot limits and aspect.

def set_limits(ax=None, xlim=None, ylim=None):
    """Generic function to set limits on axes."""

    ax = axes(ax)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def match_xy_limits(ax=None):
    """Match aspect (limits) of x and y axes."""

    ax = axes(ax)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    lim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
    ax.set_xlim(lim)
    ax.set_ylim(lim)


def set_aspect(ax=None, aspect=1, adjustable='datalim', anchor=None):
    """Matches aspect ratio of axes."""

    ax = axes(ax)
    ax.set_aspect(aspect, adjustable, anchor)


def sync_axes(axs, sync_x=False, sync_y=False):
    """Synchronize x and/or y axis across list of axes."""

    if not len(axs):
        return

    # Synchronise x-axis limits across plots.
    if sync_x:
        all_xlims = np.array([ax.get_xlim() for ax in axs])
        xlim = (all_xlims[:, 0].min(), all_xlims[:, 1].max())
        [ax.set_xlim(xlim) for ax in axs]

    # Synchronise y-axis limits across plots.
    if sync_y:
        all_ylims = np.array([ax.get_ylim() for ax in axs])
        ylim = (all_ylims[:, 0].min(), all_ylims[:, 1].max())
        [ax.set_ylim(ylim) for ax in axs]

    return axs


# %% Functions to set axes labels, title and legend.

def set_labels(ax=None, xlab=None, ylab=None, title=None, ytitle=None,
               xlab_kws=dict(), ylab_kws=dict(), title_kws=dict()):
    """Generic function to set title, labels and ticks on axes."""

    if ytitle is None:
        ytitle = 1.04

    ax = axes(ax)
    if title is not None:
        ax.set_title(title, y=ytitle, **title_kws)
    if xlab is not None:
        ax.set_xlabel(xlab, **xlab_kws)
    if ylab is not None:
        ax.set_ylabel(ylab, **ylab_kws)
    ax.tick_params(axis='both', which='major')


def set_legend(ax, add_legend=True, loc=0, frameon=False, **kwargs):
    """Add legend to axes."""

    ax = axes(ax)

    legend = None
    if add_legend:
        legend = ax.legend(loc=loc, frameon=frameon, **kwargs)

    return legend


# %% Functions to set/hide ticks and spines.

def set_spines(ax=None, bottom=True, left=True, top=False, right=False):
    """Remove selected spines (axis lines) from axes."""

    ax = axes(ax)
    if is_polar(ax):  # polar plot
        ax.spines['polar'].set_visible(bottom)

    else:  # Cartesian plot
        ax.spines['bottom'].set_visible(bottom)
        ax.spines['left'].set_visible(left)
        ax.spines['top'].set_visible(top)
        ax.spines['right'].set_visible(right)


def hide_spines(ax=None):
    """Hides all spines of axes."""

    set_spines(ax, False, False, False, False)


def set_ticks_side(ax=None, xtick_pos='bottom', ytick_pos='left'):
    """Remove selected tick marks on axes.
       xtick_pos: [ 'bottom' | 'top' | 'both' | 'default' | 'none' ]
       ytick_pos: [ 'left' | 'right' | 'both' | 'default' | 'none' ]
    """

    ax = axes(ax)
    ax.xaxis.set_ticks_position(xtick_pos)
    ax.yaxis.set_ticks_position(ytick_pos)


def hide_ticks(ax=None, show_x_ticks=False, show_y_ticks=False):
    """Hide ticks on either or both axes."""

    ax = axes(ax)
    if not show_x_ticks:
        ax.get_xaxis().set_ticks([])
    if not show_y_ticks:
        ax.get_yaxis().set_ticks([])


def hide_axes(ax=None, show_x=False, show_y=False, show_polar=False):
    """Hide all ticks, labels and spines of either or both axes."""

    # Hide axis ticks and labels.
    ax = axes(ax)
    ax.xaxis.set_visible(show_x)
    ax.yaxis.set_visible(show_y)

    # Hide requested spines of axes. (And don't change the others!)
    to_hide = []
    if is_polar(ax):
        if not show_polar:  # polar plot
            to_hide.append('polar')
    else:
        if not show_x:
            to_hide.extend(['bottom', 'top'])
        if not show_y:
            to_hide.extend(['left', 'right'])

    [ax.spines[side].set_visible(False) for side in to_hide]


def set_xtick_labels(ax=None, pos=None, lbls=None, **kwargs):
    """Set tick labels on x axis."""

    ax = axes(ax)
    ax.set_xticks(pos)
    if lbls is not None:
        ax.set_xticklabels(lbls, **kwargs)


def set_ytick_labels(ax=None, pos=None, lbls=None, **kwargs):
    """Set tick labels on y axis."""

    ax = axes(ax)
    ax.set_yticks(pos)
    if lbls is not None:
        ax.set_yticklabels(lbls, **kwargs)


def rot_xtick_labels(ax=None, rot=45, ha='right'):
    """Rotate labels on x axis."""

    ax = axes(ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rot, ha=ha)


def rot_ytick_labels(ax=None, rot=45, va='top'):
    """Rotate labels on y axis."""

    ax = axes(ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=rot, va=va)


def set_max_n_ticks(ax=None, max_n_ticks=5, axis='both'):
    """Set maximum number of ticks on axes."""

    ax = axes(ax)
    ax.locator_params(axis=axis, nbins=max_n_ticks-1)


# %% Functions to create, access and save axes and figures.

def axes(ax=None, **kwargs):
    """Return new or current axes instance."""

    if ax is None:
        ax = plt.gca(**kwargs)
    return ax


def is_polar(ax=None):
    """Return True is axes is polar type, False otherwise."""

    ax = axes(ax)
    im_polar = 'polar' in ax.spines
    return im_polar


def add_mock_axes(fig, mock_gsp, **kwargs):
    """Add mock (empty) axes to figure."""

    ax = fig.add_subplot(mock_gsp, **kwargs)
    hide_axes(ax=ax)
    return ax


def figure(fig=None, **kwargs):
    """Return new figure instance."""

    if fig is None:
        fig = plt.figure(**kwargs)
    return fig


def gridspec(nrow, ncol, gsp=None, **kwargs):
    """Return new GridSpec instance."""

    if gsp is None:
        gsp = gs.GridSpec(nrow, ncol, **kwargs)
    return gsp


def get_gs_subplots(nrow=None, ncol=None, subw=2, subh=2, ax_kws_list=None,
                    create_axes=True, as_array=True, fig=None, **kwargs):
    """Return list or array of GridSpec subplots."""

    # If ncol not specified: get approx'ly equal number of rows and columns.
    if ncol is None:
        nplots = nrow
        nrow = int(np.floor(np.sqrt(nplots)))
        ncol = int(np.ceil(nplots / nrow))
    else:
        nplots = nrow * ncol

    # Create figure and gridspec object.
    if fig is None:
        fig = figure(figsize=(ncol*subw, nrow*subh))
    gsp = gridspec(nrow, ncol, **kwargs)
    axes = None

    # Create list (or array) of axes.
    if create_axes:

        # Don't pass anything by default.
        if ax_kws_list is None:
            ax_kws_list = nplots * [{}]

        # If single dictionary has been passed, instead of list of
        # plot-specific params in dict, use it for all subplots.
        elif isinstance(ax_kws_list, dict):
            ax_kws_list = nplots * [ax_kws_list]

        # Create axes objects.
        axes = [fig.add_subplot(gs, **ax_kws)
                for gs, ax_kws in zip(gsp, ax_kws_list)]

        # Turn off last (unrequested) axes on grid, if any.
        if nplots < nrow * ncol:
            [ax.axis('off') for ax in axes[nplots:]]

        # Convert from list to array.
        if as_array:
            axes = np.array(axes).reshape(gsp.get_geometry())

    return fig, gsp, axes


def embed_gsp(outer_gsp, nrow, ncol, **kwargs):
    """Return GridSpec embedded into outer SubplotSpec."""

    sub_gsp = gs.GridSpecFromSubplotSpec(nrow, ncol, outer_gsp, **kwargs)
    return sub_gsp


def save_fig(fig=None, ffig=None, dpi=150, bbox_extra_artists=None,
             close=True):
    """Save figure to file."""

    # Init figure and folder to save figure into.
    if ffig is None:
        return
    util.create_dir(ffig)

    if fig is None:
        fig = plt.gcf()

    # Suppress warning about axes being incompatible with tight layout.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)

        # Write figure out into file.
        fig.savefig(ffig, bbox_extra_artists=bbox_extra_artists,
                    dpi=dpi, bbox_inches='tight')

    # Finally, close figure.
    if close:
        plt.close(fig)


def save_gsp_figure(fig, gsp=None, fname=None, title=None, ytitle=0.98,
                    fs_title='xx-large', rect_height=None, pad=1.08,
                    h_pad=None, w_pad=None, **kwargs):
    """Save composite (GridSpec) figure to file."""

    # Add super title to figure.
    if title is not None:
        fig.suptitle(title, y=ytitle, fontsize=fs_title, **kwargs)

    # Adjust gsp's plotting area and set tight layout.
    if gsp is not None:
        if rect_height is None:  # relative height of plotted area
            rect_height = ytitle - 0.03
        rect = [0, 0.0, 1, rect_height]
        gsp.tight_layout(fig, rect=rect, pad=pad, h_pad=h_pad, w_pad=w_pad)

    # Save figure.
    save_fig(fig, fname)


# %% Miscellanous plotting related functions.

def get_cmap(cm_name='jet', **kwargs):
    """Return colormap instance."""

    cmap = plt.get_cmap(cm_name, **kwargs)
    return cmap


def get_colors(mpl_colors=False, as_cycle=True):
    """Return colour cycle."""

    if mpl_colors:
        col_cylce = mpl.rcParams['axes.prop_cycle']  # mpl default color list
        cols = [d['color'] for d in col_cylce]
    else:
        cols = my_color_list  # custom list of colors

    if as_cycle:  # for cyclic indexing
        cols = cycle(cols)

    return cols


def convert_to_rgb(cname):
    """Return RGB tuple of color name (e.g. 'red', 'r', etc)."""

    rgb = ColConv.to_rgb(cname)
    return rgb


def convert_to_rgba(cname):
    """Return RGBA tuple of color name (e.g. 'red', 'r', etc)."""

    rgba = ColConv.to_rgba(cname)
    return rgba


def get_cmat(to_color, fcol='blue', bcol='grey', rgba=False):
    """
    Return foreground/background RGB/RGBA color matrix
    for array of points (True: foreground point, False: background point).
    """

    to_color = np.array(to_color, dtype=bool)

    # Init foreground and background colors.
    if isinstance(fcol, str):
        fcol = convert_to_rgb(fcol)
    if isinstance(bcol, str):
        bcol = convert_to_rgb(bcol)

    # Create color matrix of points.
    col_mat = np.array(len(to_color) * [bcol])
    col_mat[to_color, :] = fcol

    return col_mat


def get_proxy_artist(label, color, artist_type='patch', **kwargs):
    """Return a proxy artist. Useful for creating custom legends."""

    if artist_type == 'patch':
        artist = mpl.patches.Patch(color=color, label=label, **kwargs)

    else:   # line
        artist = mpl.lines.Line2D([], [], color=color, label=label, **kwargs)

    return artist


# %% Plotting related meta-functions.

def format_plot(ax=None, xlim=None, ylim=None, xlab=None, ylab=None,
                title=None, ytitle=None):
    """Generic plotting function."""

    # Format plot.
    set_limits(ax, xlim, ylim)
    set_ticks_side(ax)
    set_spines(ax)
    set_labels(ax, xlab, ylab, title, ytitle)

    return ax


def set_style(context='notebook', style='darkgrid', palette='deep',
              color_codes=True, rc=None):
    """Set Seaborn style, context and other matplotlib style parameters."""

    # 'style': 'darkgrid', 'whitegrid', 'dark', 'white' or 'ticks'.
    # 'context': 'notebook', 'paper', 'poster' or 'talk'.

    sns.set(context=context, style=style, palette=palette,
            color_codes=color_codes, rc=rc)


def inline_on():
    """Turn on inline plotting."""
    plt.ion()


def inline_off():
    """Turn off inline plotting."""
    plt.ioff()
