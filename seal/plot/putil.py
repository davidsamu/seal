# -*- coding: utf-8 -*-
"""
Collection of plotting-related utility functions.

@author: David Samu
"""

import warnings
from itertools import cycle

import numpy as np
import pandas as pd

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from matplotlib import collections as mc

import seaborn as sns

from seal.util import util


# %% Matplotlib setup and some plotting constants.

mpl.rc('figure', autolayout=True)  # to prevent object being cut off

ColConv = mpl.colors.ColorConverter()

t_lbl = 'Time since {} (ms)'
FR_lbl = 'Firing rate (sp/s)'

# Period (frequency) of tick marks and tick labels for time axes. In ms!
t_tick_mrks_prd = 500
t_tick_lbls_prd = 1000  # this has to be multiple of marks period!

my_color_list = ['b', 'r', 'm', 'g', 'c', 'y']

# Stimulus and cue colors.
stim_colors = pd.Series(['m', 'g'], index=['S1', 'S2'])
cue_colors = pd.Series(['grey', 'red', 'blue'],
                       index=['all', 'loc', 'dir'])

# Default Matplotlib RC params.
tick_pad = 3
tick_size = 4
tick_minor_fac = 0.75
seal_rc_params = {'xtick.major.pad': tick_pad,
                  'xtick.minor.pad': tick_minor_fac*tick_pad,
                  'ytick.major.pad': tick_pad,
                  'ytick.minor.pad': tick_minor_fac*tick_pad,
                  'xtick.major.size': tick_size,
                  'xtick.minor.size': tick_minor_fac*tick_size,
                  'ytick.major.size': tick_size,
                  'ytick.minor.size': tick_minor_fac*tick_size}

# Default decorator height levels.
ypos_marker = 0.92
ypos_lbl = 0.96


# %% Info plots.

def get_unit_info_title(u, fullname=False):
    """Plot unit info as text labels."""

    # Init dict of info labels to plot.
    upars = u.get_unit_params()

    # Init formatted parameter values.
    fpars = [('isolation', '{}'),
             ('SNR', 'SNR: {:.2f}'),
             ('ISIvr', 'ISIvr: {:.2f}%'),
             ('TrueSpikes', 'TrSpRt: {:.0f}%'),
             ('BS/NS', '{}'),
             ('mWfDur', 'Wf dur: {:.0f} $\mu$s'),
             ('Fac/Sup', '{}'),
             ('mFR', 'mean rate: {:.1f} sp/s'),
             ('baseline', 'baseline: {:.1f} sp/s')]
    fvals = [(meas, f.format(upars[meas]) if meas in upars else 'N/A')
             for meas, f in fpars]
    fvals = util.series_from_tuple_list(fvals)

    # Create info lines.
    # Header: Unit name.
    header = upars.Name if fullname else upars.task
    info_lines = '\n\n{}\n\n\n\n'.format(header)
    # Unit type.
    info_lines += '{} ({}, {}, {})\n\n'.format(fvals['isolation'],
                                               fvals['SNR'], fvals['ISIvr'],
                                               fvals['TrueSpikes'])
    # Waveform duration.
    info_lines += '{}\n\n'.format(fvals['mWfDur'])
    #info_lines += '{} ({})\n\n'.format(fvals['BS/NS'], fvals['mWfDur'])

    # Firing rate.
    info_lines += '{}, {}, {}\n\n'.format(fvals['Fac/Sup'], fvals['mFR'],
                                          fvals['baseline'])

    # Facilitatory or suppressive?
    #info_lines += '\n'.format()

    return info_lines


# %% Generic plot decorator functions.

def plot_signif_prds(rates1, rates2, pval, test, test_kws, ypos=None,
                     color='c', linewidth=4, ax=None):
    """Add significant intervals to axes."""

    ax = axes(ax)

    if ypos is None:
        ypos = ax.get_ylim()[1]

    # Get intervals of significant differences between rates.
    sign_periods = util.sign_periods(rates1, rates2, pval, test, **test_kws)

    # Assamble line segments and add them to axes.
    line_segments = [[(t1, ypos), (t2, ypos)] for t1, t2 in sign_periods]
    lc = mc.LineCollection(line_segments, colors=ColConv.to_rgba(color),
                           linewidth=linewidth)
    lc.sign_prd = True  # add label to find these artists later
    ax.add_collection(lc)


def highlight_axes(ax=None, color='red', alpha=0.5, **kwargs):
    """Highlight axes."""

    ax = axes(ax)
    rect = mpl.patches.Rectangle(xy=(0, 0), width=1, height=1, color=color,
                                 transform=ax.transAxes, alpha=alpha, **kwargs)
    ax.add_artist(rect)


def plot_periods(prds, alpha=0.10, color='grey', ax=None, **kwargs):
    """Highlight segments (periods)."""

    if prds is None:
        return

    ax = axes(ax)
    xmin, xmax = ax.get_xlim()
    for name, t_start, t_stop in prds:
        if t_start is None:
            t_start = xmin
        if t_stop is None:
            t_stop = xmax
        ax.axvspan(t_start, t_stop, alpha=alpha, color=color, **kwargs)


def plot_events(events, add_names=True, color='black', alpha=1.0,
                ls='--', lw=1, lbl_rotation=90, y_lbl=ypos_lbl,
                lbl_ha='center', ax=None, **kwargs):
    """Plot all events of unit."""

    if events is None:
        return

    ax = axes(ax)

    # Init y extents of lines and positions of labels.
    ylim = ax.get_ylim()
    yloc = ylim[0] + y_lbl * (ylim[1] - ylim[0])
    ymax = y_lbl-0.02 if add_names else 1

    # Add each event to plot as a vertical line.
    for ev_name, (time, label) in events.iterrows():
        ax.axvline(time, color=color, alpha=alpha, lw=lw, ls=ls,
                   ymax=ymax, **kwargs)

        # Add event label if requested
        if add_names:
            txt = ax.text(time, yloc, label, rotation=lbl_rotation,
                          fontsize='small', va='bottom', ha=lbl_ha)
            txt.event_lbl = True  # add label to find these artists later


def plot_event_markers(events, ypos=ypos_marker, marker='o', ms=6, mew=1,
                       mec='orange', mfc='None', ax=None, **kwargs):
    """Add event markers to plot."""

    if events is None:
        return

    ax = axes(ax)

    # Init y position.
    ylim = ax.get_ylim()
    y = ylim[0] + ypos * (ylim[1] - ylim[0])

    for event_data in events:
        ev_time = event_data['time']
        ev_mec = event_data['color'] if 'color' in event_data else mec
        ev_mfc = event_data['color'] if 'color' in event_data else mfc
        marker = ax.plot(ev_time, y, marker, ms=ms, mew=mew, mec=ev_mec,
                         mfc=ev_mfc, **kwargs)[0]
        marker.set_clip_on(False)   # disable clipping
        marker.event_marker = True  # add label to find these artists later


def add_chance_level(ylevel=0.5, color='grey', ls='--', alpha=0.5, ax=None):
    """Add horizontal line denoting chance level for decoder accuracy plot."""

    ax = axes(ax)
    ax.axhline(ylevel, color=color, ls=ls, alpha=alpha)


def add_baseline(baseline=0, color='grey', ls='--', lw=1, ax=None, **kwargs):
    """Add baseline rate to plot."""

    ax = axes(ax)
    if is_polar(ax):  # polar plot
        theta, radius = np.linspace(0, 2*np.pi, 100), baseline*np.ones(100)
        ax.plot(theta, radius, color=color, ls=ls, lw=lw, **kwargs)
    else:
        ax.axhline(baseline, color=color, ls=ls, lw=lw, **kwargs)


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


# %% Functions to adjust position of plot decorators
# (significance lines, event labels and markers, etc.).

def adjust_decorators(ax=None, ypos=None, y_lbl=ypos_lbl, y_mkr=ypos_marker):
    """
    Meta function to adjust position of all plot decorators,
    typically after resetting y-limit, e.g. to match across set of axes.
    """

    move_signif_lines(ax, ypos)
    move_event_lbls(ax, y_lbl)
    move_event_markers(ax, y_mkr)


def move_signif_lines(ax=None, ypos=None):
    """Move significance line segments to top of plot."""

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


def move_event_lbls(ax=None, y_lbl=ypos_lbl):
    """Move event labels to top of plot."""

    # Init.
    ax = axes(ax)

    ylim = ax.get_ylim()
    y = ylim[0] + y_lbl * (ylim[1] - ylim[0])

    # Find line segments in axes representing significant periods.
    for txt in ax.texts:
        if hasattr(txt, 'event_lbl'):
            txt.set_y(y)


def move_event_markers(ax=None, y_mkr=ypos_marker):
    """Move event markers to top of plot."""

    # Init.
    ax = axes(ax)

    ylim = ax.get_ylim()
    y = ylim[0] + y_mkr * (ylim[1] - ylim[0])

    # Find line segments in axes representing significant periods.
    for line in ax.lines:
        if hasattr(line, 'event_marker'):
            line.set_ydata(y)


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


def set_legend(ax, loc=0, frameon=False, **kwargs):
    """Add legend to axes."""

    ax = axes(ax)
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
    """Hide ticks (both marks and labels) on either or both axes."""

    ax = axes(ax)
    if not show_x_ticks:
        ax.get_xaxis().set_ticks([])
    if not show_y_ticks:
        ax.get_yaxis().set_ticks([])


def hide_tick_marks(ax=None, show_x_tick_mrks=False, show_y_tick_mrks=False):
    """Hide ticks marks (but not tick labels) on either or both axes."""

    ax = axes(ax)
    if not show_x_tick_mrks:
        ax.tick_params(axis='x', which='both', length=0)
    if not show_y_tick_mrks:
        ax.tick_params(axis='y', which='both', length=0)


def hide_tick_labels(ax=None, show_x_tick_lbls=False, show_y_tick_lbls=False):
    """Hide tick labels (but not tick marks) on either or both axes."""

    ax = axes(ax)
    if not show_x_tick_lbls:
        ax.tick_params(labelbottom='off')
    if not show_y_tick_lbls:
        ax.tick_params(labelleft='off')

    # Alternatively:
    # lbls = len(ax.get_xticklabels()) * ['']
    # ax.set_xticklabels(lbls)


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


def get_tick_marks_and_labels(t1, t2, mrks_prd=t_tick_mrks_prd,
                              lbls_prd=t_tick_lbls_prd):
    """
    Return tick marks and tick labels for time window.

    t1 and t2 are assumed to be in ms.
    """

    # Calculate tick mark positions.
    t1_limit = np.ceil(t1/mrks_prd) * mrks_prd
    t2_limit = t2 + 1
    tick_mrks = np.arange(t1_limit, t2_limit, mrks_prd)

    # Create tick labels for marks.
    tick_lbls = [str(int(t)) if not t % lbls_prd else '' for t in tick_mrks]

    return tick_mrks, tick_lbls


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
    """Return new (or passed) axes."""

    if ax is None:
        ax = plt.gca(**kwargs)
    return ax


def is_polar(ax=None):
    """Check if axes is polar type."""

    ax = axes(ax)
    im_polar = 'polar' in ax.spines
    return im_polar


def add_mock_axes(fig, sps, **kwargs):
    """Add mock (empty) axes to figure."""

    ax = fig.add_subplot(sps, **kwargs)
    hide_axes(ax=ax)
    return ax


def figure(fig=None, **kwargs):
    """Return new (or passed) figure."""

    if fig is None:
        fig = plt.figure(**kwargs)
    return fig


def gridspec(nrow, ncol, gsp=None, **kwargs):
    """Return new GridSpec instance."""

    if gsp is None:
        gsp = gs.GridSpec(nrow, ncol, **kwargs)
    return gsp


def sps_fig(sps=None, fig=None):
    """Return new (or passed) sps and figure."""

    fig = figure(fig)
    if sps is None:
        sps = gridspec(1, 1)[0]
    return sps, fig


def get_gs_subplots(nrow=None, ncol=None, subw=2, subh=2, ax_kws_list=None,
                    create_axes=False, as_array=True, fig=None, **kwargs):
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


# %% Functions to save figure.

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
                    fs_title='xx-large', rect_height=None, border=0.03,
                    pad=1.08, h_pad=None, w_pad=None, **kwargs):
    """Save composite (GridSpec) figure to file."""

    # Add super title to figure.
    if title is not None:
        fig.suptitle(title, y=ytitle, fontsize=fs_title, **kwargs)

    # Adjust gsp's plotting area and set tight layout.
    if gsp is not None:
        if rect_height is None:  # relative height of plotted area
            rect_height = ytitle - border
        rect = [border, border, 1.0-border, rect_height]
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


def get_artist(label, color, artist_type='patch', **kwargs):
    """Return an artist. Useful for creating custom legends."""

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
              color_codes=True, rc=seal_rc_params):
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
