# -*- coding: utf-8 -*-

"""
Function to plot unit information plot.

@author: David Samu
"""

from seal.util import util
from seal.plot import putil


def plot_unit_info(u, fs='large', ax=None):
    """Plot unit info as text labels."""

    # Init axes.
    ax = putil.axes(ax)
    putil.hide_axes(ax=ax)

    # Init dict of info labels to plot.
    upars = u.get_unit_params()

    # Init formatted parameter values.
    fpars = [('isolation', '{}'),
             ('SNR', 'SNR: {:.2f}'),
             ('ISIvr', 'ISIvr: {:.2f}%'),
             ('TrueSpikes', 'TrSpRt: {:.0f}%'),
             ('BS/NS', '{}'),
             ('mWfDur', 'WfDur: {:.0f} $\mu$s'),
             ('Fac/Sup', '{}'),
             ('mFR', 'mean rate: {:.1f} sp/s'),
             ('baseline', 'baseline: {:.1f} sp/s'),
             ('TaskRelated', 'task-related? {}')]
    fvals = [(meas, f.format(upars[meas]) if meas in upars else 'N/A')
             for meas, f in fpars]
    fvals = util.series_from_tuple_list(fvals)

    # Create info lines.
    # Unit  name.
    info_lines = '\n\n{}\n\n'.format(upars.task)  # upars.Name ?
    # Unit type.
    info_lines += '{} ({}, {}, {})\n\n'.format(fvals['isolation'],
                                               fvals['SNR'], fvals['ISIvr'],
                                               fvals['TrueSpikes'])
    # Waveform duration.
    info_lines += '{} ({})\n\n'.format(fvals['BS/NS'], fvals['mWfDur'])

    # Firing rate.
    info_lines += '{}, {}, {},\n{}\n\n'.format(fvals['Fac/Sup'], fvals['mFR'],
                                               fvals['baseline'],
                                               fvals['TaskRelated'])

    # Facilitatory or suppressive?
    info_lines += '\n'.format()

    # Plot info as axes title.
    putil.set_labels(ax, title=info_lines, ytitle=0,
                     title_kws={'fontsize': 'large'})

    # Highlight excluded unit.
    if u.is_excluded():
        putil.highlight_axes(ax)

    return ax
