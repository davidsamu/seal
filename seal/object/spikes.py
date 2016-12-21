#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:56:46 2016

Class for array of spike trains.

@author: David Samu
"""

import numpy as np
import pandas as pd
from quantities import s
from neo import SpikeTrain
from elephant import statistics

from seal.util import util


class Spikes:
    """Class for storing spike trains per trial and associated properties."""

    # %% Constructor
    def __init__(self, spk_trains, t_starts=None, t_stops=None):
        """Create a Spikes instance."""

        # Create empty instance.
        self.spikes = None
        self.t_starts = None
        self.t_stops = None

        # Init t_starts and t_stops.
        # Below deals with single values, including None.
        n_trs = len(spk_trains)
        if not util.is_iterable(t_starts):
            t_starts = n_trs * [t_starts]
        if not util.is_iterable(t_stops):
            t_stops = n_trs * [t_stops]

        # Convert into Pandas Series for speed and more functionality.
        self.t_starts = pd.Series(t_starts)
        self.t_stops = pd.Series(t_stops)

        # Create list of Neo SpikeTrain objects.
        self.spk_trains = pd.Series(index=np.arange(n_trs), dtype=object)
        for i in self.spk_trains.index:

            # This also removes spikes outside of time window.
            t_start, t_stop = self.t_starts[i], self.t_stops[i]
            spk_tr = util.values_in_window(spk_trains[i], t_start, t_stop)
            self.spk_trains[i] = SpikeTrain(spk_tr, t_start=t_start,
                                            t_stop=t_stop)

    # %% Utility methods.

    def init_time_limits(self, t1s=None, t2s=None):
        """Set time limits to default values if not specified."""

        # Default time limits.
        if t1s is None:
            t1s = self.t_starts
        if t2s is None:
            t2s = self.t_stops
        return t1s, t2s

    def n_trials(self):
        """Return number of trials."""

        n_trs = len(self.spk_trains.index)
        return n_trs

    def get_spikes(self, trs=None, t1s=None, t2s=None):
        """Return spike times of given trials within time windows."""

        # Default trial list.
        if trs is None:
            trs = np.ones(self.n_trials(), dtype=bool)

        # Default time limits.
        t1s, t2s = self.init_time_limits(t1s, t2s)

        # Assamble time-windowed spike trains.
        itrs = np.where(trs)[0]
        spk_trains = pd.Series(index=itrs, dtype=object)
        for itr in itrs:

            # Select spikes between t1 and t2 during selected trials, and
            # Convert them into new SpikeTrain list, with time limits set.
            t1, t2 = t1s[itr], t2s[itr]
            spk_tr = util.values_in_window(self.spk_trains[itr], t1, t2)
            spk_trains[itr] = SpikeTrain(spk_tr, t_start=t1, t_stop=t2)

        return spk_trains

    # %% Methods for summary statistics over spikes.

    def n_spikes(self, trs=None, t1s=None, t2s=None):
        """Return spike count of given trials in time windows."""

        # Default time limits.
        t1s, t2s = self.init_time_limits(t1s, t2s)

        # Select spikes within windows.
        spk_trains = self.get_spikes(trs, t1s, t2s)

        # Count spikes during each selected trial.
        n_spikes = spk_trains.apply(np.size)

        return n_spikes

    def rates(self, trs=None, t1s=None, t2s=None):
        """Return rates of given trials in time windows."""

        t1s, t2s = self.init_time_limits(t1s, t2s)

        # Get number of spikes.
        n_spikes = self.n_spikes(trs, t1s, t2s)

        # Rescale time limits.
        t1s = util.rescale_series(t1s, s)
        t2s = util.rescale_series(t2s, s)

        # Calculate rates for each selected trial.
        rates = n_spikes / (t2s - t1s)

        return rates

    def spike_rate_stats(self, trs=None, t1s=None, t2s=None):
        """Return spike rate statistics across selected trials."""

        # Get rates in given trials in time windows.
        rates = np.array(self.rates(trs, t1s, t2s))

        # Calculate statistics.
        if rates.size:
            mean_rate = np.mean(rates)
            std_rate = np.std(rates)
            sem_rate = std_rate / np.sqrt(rates.size)
        else:  # in case there are no spike in interval
            mean_rate = np.nan * 1/s
            std_rate = np.nan * 1/s
            sem_rate = np.nan * 1/s

        # Put them into a Series.
        spk_stats = pd.Series([mean_rate, std_rate, sem_rate],
                              index=['mean', 'std', 'sem'])

        return spk_stats

    def isi(self, trs=None, t1s=None, t2s=None):
        """Return interspike intervals per trial."""

        spks = self.get_spikes(trs, t1s, t2s)
        isi = [statistics.isi(spk) for spk in spks]

        return isi
