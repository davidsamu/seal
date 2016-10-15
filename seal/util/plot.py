# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 16:27:05 2016

Collection of plotting function.

@author: David Samu
"""

import warnings
from itertools import cycle
from collections import Counter
from collections import OrderedDict as OrdDict

import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from quantities import ms, rad, deg

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from matplotlib import collections as mc

from seal.analysis import tuning
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


# %% Functions to plot group and unit level properties.

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
                          'DSI S1', 'DSI S2',
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


def unit_info(u, fs='large', ax=None):
    """Plot unit info as text labels."""

    # Init axes.
    ax = axes(ax)
    hide_axes(ax=ax)

    # Init dictionary of info labels to plot.
    uparams = u.get_unit_params()
    lbl_dict = OrdDict([('SNR', '{:.2f}'.format(uparams['SNR'])),
                        ('WfDur', '{:.2f} ms'.format(uparams['MeanWfAmplitude']/1000)),
                        ('FR', '{:.1f} sp/s'.format(uparams['MeanFiringRate (sp/s)']))])

    # Plot each label.
    yloc = .0
    xloc = np.linspace(.10, .85, len(lbl_dict))
    for xloci, (lbl, val) in zip(xloc, lbl_dict.items()):
        lbl_str = '{}: {}'.format(lbl, val)
        ax.text(xloci, yloc, lbl_str, fontsize=fs, va='bottom', ha='center')

    # Set title.
    title = 'ch {} / {}  --  {}'.format(uparams['channel #'],
                                        uparams['unit #'],
                                        uparams['experiment'])
    set_labels(title=title, ytitle=.30, ax=ax)

    return ax


# %% Functions to plot rasters and rates.

def empty_raster_rate(fig, outer_gsp, nraster):
    """Plot empty raster and rate plots."""

    mock_gsp_rasters = embed_gsp(outer_gsp[0], nraster, 1, hspace=.15)
    for i in range(nraster):
        add_mock_axes(fig, mock_gsp_rasters[i, 0])
    mock_gsp_rate = embed_gsp(outer_gsp[1], 1, 1)
    add_mock_axes(fig, mock_gsp_rate[0, 0])


def raster_rate(spikes_list, rates, times, t1, t2, names, t_unit=ms,
                segments=None, pvals=None, ylim=None, title=None,
                xlab='Time (ms)', ylab_rate='Firing rate (sp/s)',
                add_ylab_raster=True,  markersize=1.5, legend=True,
                nlegend=True, fig=None, ffig=None, outer_gsp=None):
    """Plot raster and rate plots."""

    # Create subplots (as nested gridspecs).
    fig = figure(fig)
    if outer_gsp is None:
        outer_gsp = gs.GridSpec(2, 1, height_ratios=[1, 1])
    gsp_raster = embed_gsp(outer_gsp[0], len(spikes_list), 1, hspace=.15)
    gsp_rate = embed_gsp(outer_gsp[1], 1, 1)
    ylab_posx = -0.05

    # Raster plot(s).
    for i, sp_tr in enumerate(spikes_list):
        ax = fig.add_subplot(gsp_raster[i, 0])
        ttl = title if i == 0 else None  # add title to top raster
        ylab = names[i] if add_ylab_raster else None
        raster(sp_tr, t1, t2, t_unit, segments, markersize,
               title=ttl, xlab=None, ylab=ylab, ax=ax)
        ax.set_xticklabels([])
        ax.get_yaxis().set_label_coords(ylab_posx, 0.5)

    # Rate plot.
    ax = fig.add_subplot(gsp_rate[0, 0])
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
    ylim = [0.5, len(spikes)+0.5] if len(spikes) else [0, 1]
    set_limits(xlim, ylim, ax=ax)
    ax.locator_params(axis='y', nbins=6)
    show_ticks(xtick_pos='none', ytick_pos='none', ax=ax)
    show_spines(False, False, False, False, ax=ax)
    set_labels(title, xlab, ylab, ax=ax)

    # Add '1' to tick labels.
    # ax.yaxis.set_ticks([1] + ax.yaxis.get_majorticklocs())

    # Order trials from top to bottom.
    # (Has to happen after setting axis limits!)
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
        lbl = '{} ({} trs)'.format(name, rts.shape[0])

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
    set_labels(title, xlab, ylab, ax=ax)

    # Add legend
    legend_kwargs = dict([('frameon', False), ('framealpha', 0.5),
                          ('loc', 'upper right'), ('borderaxespad', 0),
                          ('handletextpad', 0)])
    set_legend(ax, legend, **legend_kwargs)

    # Add significance line to top of plot.
    if pvals is not None and len(rates_list) == 2:
        lws = [2.0 * i for i in range(1, len(pvals)+1)]
        cols = len(pvals) * ['c']  # ['m', 'c', 'g', 'y', 'r']
        ypos = ax.get_ylim()[1]
        for pval, lw, col in zip(pvals, lws, cols):
            plot_significant_intervals(rates_list[0], rates_list[1], time,
                                       pval, ypos=ypos, color=col,
                                       linewidth=lw, ax=ax)

    # Save and return plot.
    save_fig(plt.gcf(), ffig)
    return ax


# %% Functions to plot tuning curve and direction selectivity.

def empty_direction_selectivity(fig, outer_gsp):
    """Plot empty direction selectivity plots."""

    mock_gsp_polar = embed_gsp(outer_gsp[0], 1, 1)
    add_mock_axes(fig, mock_gsp_polar[0, 0], polar=True)
    mock_gsp_tuning = embed_gsp(outer_gsp[1], 1, 1)
    add_mock_axes(fig, mock_gsp_tuning[0, 0])


# TODO: this function should be refactored!
def direction_selectivity(dir_select_dict, title=None, labels=True,
                          polar_legend=True, tuning_legend=True,
                          ffig=None, fig=None, outer_gsp=None):
    """Plot direction selectivity on polar plot and tuning curve."""

    # Init plots.
    if outer_gsp is None:
        fig, outer_gsp, _ = get_gs_subplots(1, 2, subw=6, subh=5,
                                            create_axes=False, fig=fig)
    ax_polar = fig.add_subplot(outer_gsp[0], polar=True)
    ax_tuning = fig.add_subplot(outer_gsp[1])
    colors = get_colors()
    polar_patches = []
    tuning_patches = []

    for name, values in dir_select_dict.items():

        # Init stimulus plotting.
        dirs, mean_resp, sem_resp, dsi, pref_dir, pref_dir_c = values
        color = next(colors)

        # Plot direction selectivity on polar plot.
        polar_direction_response(dirs, mean_resp, dsi, pref_dir,
                                 color=color, ax=ax_polar)

        # Shift direction - response values to center preferred direction.
        ndirs = len(dirs)
        idx = np.array(range(ndirs))
        dirs_offset = dirs - pref_dir
        to_right = dirs_offset < -180*deg   # indices to move to the right
        to_left = dirs_offset > 180*deg     # indices to move to the left
        center = np.invert(np.logical_or(to_right, to_left))  # indices to keep
        idx = np.hstack((idx[to_left], idx[center], idx[to_right]))
        dirs_shifted = util.deg_mod(dirs_offset[idx])
        idx_to_flip = dirs_shifted > 180*deg
        dirs_shifted[idx_to_flip] = dirs_shifted[idx_to_flip] - 360*deg
        mean_resp_shifted = mean_resp[idx]
        sem_resp_shifted = sem_resp[idx]

        # Calculate and plot direction tuning curve.
        xlab = 'Difference from preferred direction (deg)' if labels else None
        ylab = 'Firing rate (sp/s)' if labels else None
        r = tuning.test_tuning(dirs_shifted, mean_resp_shifted, sem_resp_shifted,
                               stim_min=-180*deg, stim_max=180*deg,
                               xlab=xlab, ylab=ylab, color=color, ax=ax_tuning)
        a, b, x0, sigma = r[0].loc['fit']

        # Collect parameters of polar plot (stimulus - response).
        s_pd = str(float(round(pref_dir, 1)))
        s_pd_c = str(int(pref_dir_c)) if not np.isnan(pref_dir_c.magnitude) else 'nan'
        lgd_lbl = '{}:   {:.3f}'.format(name, dsi)
        lgd_lbl += '     {:>5}$^\circ$ --> {:>3}$^\circ$ '.format(s_pd, s_pd_c)
        polar_patches.append(get_proxy_patch(lgd_lbl, color))

        # Collect parameters of tuning curve fit.
        s_a = str(float(round(a, 1)))
        s_b = str(float(round(b, 1)))
        s_x0 = str(float(round(x0, 1)))
        s_sigma = str(float(round(sigma, 1)))
        lgd_lbl = '{}:{}{:>6}{}{:>6}'.format(name, 5 * ' ', s_a, 5 * ' ', s_b)
        lgd_lbl += '{}{:>6}{}{:>6}'.format(5 * ' ', s_x0, 8 * ' ', s_sigma)
        tuning_patches.append(get_proxy_patch(lgd_lbl, color))

    # Add zero reference line to tuning curve.
    ax_tuning.axvline(0, color='k', ls='--', alpha=0.2)

    # Set limits of tuning curve (after all curves have been plotted).
    xlim = [-180-5, 180+5]
    ylim = [0, None]
    set_limits(xlim, ylim, ax_tuning)

    # Set labels.
    if labels:
        set_labels('Polar plot', ytitle=1.08, ax=ax_polar)
        set_labels('Tuning curve', ytitle=1.08, ax=ax_tuning)
    if title is not None:
        fig.suptitle(title, y=0.98, fontsize='xx-large')

    # Set legends.
    ylegend = -0.30 if labels else -0.20
    fr_on = False if labels else True
    legend_kwargs = dict([('fancybox', True), ('shadow', False),
                          ('frameon', fr_on), ('framealpha', 1.0),
                          ('loc', 'lower center'),
                          ('bbox_to_anchor', [0., ylegend, 1., .0]),
                          ('prop', {'family': 'monospace'})])
    polar_lgn_ttl = 'DSI'.rjust(20) + 'PD'.rjust(14) + 'PD8'.rjust(14)
    tuning_lgd_ttl = ('a (sp/s)'.rjust(35) + 'b (sp/s)'.rjust(15) +
                      'x0 (deg)'.rjust(13) + 'sigma (deg)'.rjust(15))

    lgn_params = [(polar_legend, polar_lgn_ttl, polar_patches, ax_polar),
                  (tuning_legend, tuning_lgd_ttl, tuning_patches, ax_tuning)]
    for (plot_legend, lgn_ttl, patches, ax) in lgn_params:
        if not plot_legend:
            continue
        if not labels:  # customisation for summary plot
            lgd_ttl = None
        lgd = set_legend(ax, handles=patches, title=lgd_ttl, **legend_kwargs)
        lgd.get_title().set_ha('left')
        if legend_kwargs['frameon']:
            lgd.get_frame().set_linewidth(.5)

    # Save figure.
    if hasattr(outer_gsp, 'tight_layout'):
        outer_gsp.tight_layout(fig, rect=[0, 0.0, 1, 0.95])
    save_fig(fig, ffig)


def polar_direction_response(dirs, resp, DSI=None, pref_dir=None,
                             color='g', title=None, ffig=None, ax=None):
    """
    Plot response to each directions on polar plot, with direction selectivity
    vector.
    """

    # Plot response to each directions on polar plot.
    rad_dirs = np.array([d.rescale(rad) for d in dirs])
    ndirs = rad_dirs.size
    left_rad_dirs = rad_dirs - np.pi/ndirs
    w = 2*np.pi / ndirs
    ax = bars(left_rad_dirs, resp, polar=True, width=w, alpha=0.50,
              color=color, edgecolor='w', linewidth=1, title=title,
              ytitle=1.08, ax=ax)

    # Add arrow representing preferred direction and
    # direction selectivity index (DSI).
    if DSI is not None and pref_dir is not None:
        rho = np.max(resp) * DSI
        xy = (float(pref_dir.rescale(rad)), rho)
        ax.annotate('', xy, xytext=(0, 0),
                    arrowprops=dict(facecolor=color, edgecolor='k',
                                    shrink=0.0, alpha=0.5))

    # Save and return plot.
    save_fig(plt.gcf(), ffig)
    return ax


def tuning_curve(val, mean_resp, sem_resp, xfit, yfit, color='b', title=None,
                 xlab=None, ylab=None, ffig=None, ax=None, **kwargs):
    """Plot tuning curve."""

    # Add fitted curve.
    ax = lines(xfit, yfit, color=color, ax=ax)

    # Plot data samples of tuning curve.
    errorbar(val, mean_resp, yerr=sem_resp, fmt='o', color=color,
             title=title, xlab=xlab, ylab=ylab, ax=ax, **kwargs)

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


def move_significance_lines(ax, ypos=None):
    """
    Move significance line segments to top of plot
    (typically after resetting y-limit).
    """

    # Init.
    ax = axes(ax)
    if ypos is None:
        ypos = ax.get_ylim()[1]

    # Find significance line segments.
    for c in ax.collections:
        # This assumes that there is not other LineCollection object plotted!
        if isinstance(c, mc.LineCollection):
            segments = [np.array((seg[:, 0], seg.shape[0]*[ypos])).T
                        for seg in c.get_segments()]
            c.set_segments(segments)


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
                color='black', lw=1, lbl_rotation=90, lbl_height=0.98,
                lbl_ha='center', ax=None, **kwargs):
    """Plot all events of unit."""

    ax = axes(ax)

    # Add each event to plot as a vertical line.
    for key, time in events.items():
        time = time.rescale(t_unit)
        ymax = lbl_height-0.02 if add_names else 1
        ax.axvline(time, color=color, alpha=alpha, lw=lw, ymax=ymax, **kwargs)

        # Add event label if requested
        if add_names:
            ylim = ax.get_ylim()
            yloc = ylim[0] + lbl_height * (ylim[1] - ylim[0])
            ax.text(time, yloc, key, rotation=lbl_rotation, fontsize='small',
                    verticalalignment='bottom', horizontalalignment=lbl_ha)


# %% Plot setup functions.

def set_limits(xlim=None, ylim=None, ax=None):
    """Generic function to set limits on axes."""

    ax = axes(ax)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def sync_axes(axs, sync_x=False, sync_y=False):
    """Synchronize x and/or y axis across list of axes."""

    if sync_x:
        all_xlims = np.array([ax.get_xlim() for ax in axs])
        xlims = (all_xlims[:, 0].min(), all_xlims[:, 1].max())
        [ax.set_xlim(xlims) for ax in axs]

    if sync_y:
        all_ylims = np.array([ax.get_ylim() for ax in axs])
        ylims = (all_ylims[:, 0].min(), all_ylims[:, 1].max())
        [ax.set_ylim(ylims) for ax in axs]

    return axs


def set_labels(title=None, xlab=None, ylab=None, ytitle=None, ax=None,
               title_kwargs=dict(), xlab_kwargs=dict(), ylab_kwargs=dict()):
    """Generic function to set title, labels and ticks on axes."""

    ax = axes(ax)
    if title is not None:
        ytitle = ytitle if ytitle is not None else 1.04
        ax.set_title(title, y=ytitle, **title_kwargs)
    if xlab is not None:
        ax.set_xlabel(xlab, **xlab_kwargs)
    if ylab is not None:
        ax.set_ylabel(ylab, **ylab_kwargs)
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


def hide_axes(show_x=False, show_y=False, ax=None):
    """Hide all ticks and spines of either or both axes."""

    ax = axes(ax)

    if not show_x:
        ax.get_xaxis().set_ticks([])
    if not show_y:
        ax.get_yaxis().set_ticks([])

    show_spines(show_x, show_y, show_x, show_y, ax)

    xtick_pos = 'default' if show_x else 'none'
    ytick_pos = 'default' if show_y else 'none'
    show_ticks(xtick_pos, ytick_pos, ax)


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

    if not add_legend:
        return None

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


# TODO: replace this with get_gs_subplots below!
def get_subplots(nplots, sp_width=4, sp_height=3, **kwargs):
    """Returns figures with given number of axes initialised."""

    # Create figure with axes arranged in grid pattern.
    nrow = int(np.floor(np.sqrt(nplots)))
    ncol = int(np.ceil(nplots / nrow))
    figsize = (sp_width*ncol, sp_height*nrow)
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize, **kwargs)

    # Turn off last axes on grid.
    axsf = axs.flatten()
    [ax.axis('off') for ax in axsf[nplots:]]

    return fig, axsf


def get_gs_subplots(nrow=None, ncol=None, subw=2, subh=2,
                    create_axes=True, as_array=True, fig=None, **kwargs):
    """Return list or array of GridSpec subplots."""

    # Create figure, gridspec.
    if fig is None:
        fig = figure(figsize=(ncol*subw, nrow*subh))
    gsp = gs.GridSpec(nrow, ncol, **kwargs)
    axes = None

    # Create list (or array) of axes.
    if create_axes:
        axes = [fig.add_subplot(gs) for gs in gsp]
        if as_array:
            axes = np.array(axes).reshape(gsp.get_geometry())

    return fig, gsp, axes


def embed_gsp(outer_gsp, nrow, ncol, **kwargs):
    """Return GridSpec embedded into outer SubplotSpec."""

    sub_gsp = gs.GridSpecFromSubplotSpec(nrow, ncol, outer_gsp, **kwargs)
    return sub_gsp


def add_mock_axes(fig, mock_gsp, **kwargs):
    """Add mock (empty) axes to figure."""

    ax = fig.add_subplot(mock_gsp, **kwargs)
    hide_axes(ax=ax)
    return ax


def get_colormap(cm_name='jet', **kwargs):
    """Return colormap instance."""

    cmap = plt.get_cmap(cm_name, **kwargs)
    return cmap


def inline_on():
    """Turn on inline plotting."""
    plt.ion()


def inline_off():
    """Turn off inline plotting."""
    plt.ioff()


# %% Miscellanous plot related functions.

def get_colors(from_mpl_cycle=False, as_cycle=True):
    """Return colour cycle."""

    if from_mpl_cycle:
        col_cylce = mpl.rcParams['axes.prop_cycle']  # MPL default color list
        cols = [d['color'] for d in col_cylce]
    else:
        cols = ['m', 'g', 'r', 'c', 'b', 'y']  # custom list of colors

    if as_cycle:  # for cyclic indexing
        cols = cycle(cols)

    return cols


def get_proxy_patch(label, color):
    """Return patch proxy artist. Useful for creating custom legends."""

    patch = mpl.patches.Patch(color=color, label=label)
    return patch


# %% General purpose plotting functions.

def base_plot(x, y=None, xlim=None, ylim=None, xlab=None, ylab=None,
              title=None, ytitle=None, polar=False, figtype='lines',
              ffig=None, ax=None, **kwargs):
    """Generic plotting function."""

    ax = axes(ax, polar=polar)

    if figtype == 'lines':
        ax.plot(x, y, **kwargs)

    elif figtype == 'bars':
        ax.bar(x, y, **kwargs)

    elif figtype == 'errorbar':
        ax.errorbar(x, y, **kwargs)

    elif figtype == 'hist':

        x = np.array(x)

        # Categorical data.
        if not util.is_numeric_array(x):

            # Sort by category value (with missing values at the end).
            counts = np.array(Counter(x).most_common())
            idx = pd.notnull(counts[:, 0])
            non_null_order = counts[idx, 0].argsort()
            null_order = np.where(np.logical_not(idx))[0]
            counts = counts[np.hstack((non_null_order, null_order)), :]

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


def bars(x, y, ylim=None, xlim=None, xlab=None, ylab=None, title=None,
         ytitle=None, polar=False, ffig=None, ax=None, **kwargs):
    """Plot simple lines."""

    # Plot bar plot.
    ax = base_plot(x, y, xlim, ylim, xlab, ylab, title, ytitle, polar,
                   'bars', ffig, ax=ax, **kwargs)
    return ax


def errorbar(x, y, ylim=None, xlim=None, xlab=None, ylab=None,
             title=None, ytitle=None, polar=False, ffig=None, ax=None,
             **kwargs):
    """Plot error bars."""

    # Plot line plot.
    ax = base_plot(x, y, xlim, ylim, xlab, ylab, title, ytitle, polar,
                   'errorbar', ffig, ax=ax, **kwargs)
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
