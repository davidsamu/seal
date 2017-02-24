# -*- coding: utf-8 -*-

"""
Functions to plot decoding results.

@author: David Samu
"""

import numpy as np
import pandas as pd
import seaborn as sns

from seal.plot import putil


def plot_scores(Scores, ax=None, time='time', value='score', unit='fold',
                color='b'):
    """Plot decoding scores over time as time series."""

    # Prepare data
    lScores = pd.DataFrame(Scores.unstack(), columns=[value])
    lScores[time] = lScores.index.get_level_values(0)
    lScores[unit] = lScores.index.get_level_values(1)
    lScores.index = np.arange(len(lScores.index))

    # Plot as time series.
    sns.tsplot(lScores, time=time, value=value, unit=unit, color=color, ax=ax)


def plot_decoding_res(ax, Scores, ShfldScores, nvals=None, prds=None,
                      xlim=None, ylim=[0, 1], xlab='time',
                      ylab='decoding accuracy', title='', ytitle=1.04):
    """Plot decoding accuracy results."""

    # Plot scores.
    plot_scores(Scores, ax, color='b')

    # Plot shuffled scores.
    if not ShfldScores.isnull().all().all():
        plot_scores(ShfldScores, ax, color='g')

        # Add legend.
        lgn_patches = [putil.get_artist('synchronous', 'b'),
                       putil.get_artist('pseudo-population', 'g')]
        putil.set_legend(ax, handles=lgn_patches)

    # Add chance level line.
    # This currently plots all nvals combined across stimulus period!
    if nvals is not None:
        chance_lvl = 1.0 / nvals
        putil.add_chance_level(ax=ax, ylevel=chance_lvl)

    # Add stimulus periods.
    if prds is not None:
        putil.plot_periods(prds, ax=ax)

    # Set axis limits.
    putil.set_limits(ax, xlim, ylim)

    # Format plot.
    putil.set_labels(ax, xlab, ylab, title, ytitle)


def plot_weights(ax, Coefs, prds=None, xlim=None, xlab='time',
                 ylab='unit coefficient', title='', ytitle=1.04):
    """Plot decoding weights."""

    # Unstack dataframe with results.
    lCoefs = pd.DataFrame(Coefs.unstack().unstack(), columns=['coef'])
    lCoefs['time'] = lCoefs.index.get_level_values(0)
    lCoefs['value'] = lCoefs.index.get_level_values(1)
    lCoefs['uid'] = lCoefs.index.get_level_values(2)
    lCoefs.index = np.arange(len(lCoefs.index))

    # Plot time series.
    sns.tsplot(lCoefs, time='time', value='coef', unit='value',
               condition='uid', ax=ax)

    # Add chance level line and stimulus periods.
    putil.add_chance_level(ax=ax, ylevel=0)
    putil.plot_periods(prds, ax=ax)

    # Set axis limits.
    putil.set_limits(ax, xlim)

    # Format plot.
    putil.set_labels(ax, xlab, ylab, title, ytitle)
    putil.hide_legend(ax)
