#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:56:46 2016

Class for array of spike trains.

@author: David Samu
"""


from itertools import compress

import numpy as np
import pandas as pd
from quantities import s
from neo import SpikeTrain
from elephant import statistics

from seal.util import util
from seal.object.trials import Trials


class Spikes:
    """Class for spike trains per trial and associated properties."""

    # %% Constructor
    def __init__(self, spike_train_list, t_start=0*s, t_stop=0*s):
        """Create a Spikes instance."""

        # Create empty instance.
        self.spikes = None
        self.t_start = t_start.rescale(s)
        self.t_stop = t_stop.rescale(s)

        # Remove spikes outside of time window [t_start, t_stop].
        tr_spk = [util.values_in_window(spk, t_start, t_stop)
                  for spk in spike_train_list]

        # Create list of Neo SpikeTrain objects.
        self.spikes = [SpikeTrain(spk, t_start=t_start, t_stop=t_stop)
                       for spk in tr_spk]

    # %% Utility methods.

    def init_time_limits(self, t1, t2):
        """Set time limits to default values if not specified."""

        # Default time limits.
        if t1 is None:
            t1 = self.t_start
        if t2 is None:
            t2 = self.t_stop
        return t1, t2

    def n_trials(self):
        """Return number of trials."""

        n_trials = len(self.spikes)
        return n_trials

    def get_spikes(self, trs=None, t1=None, t2=None):
        """Return spike times of given trials within time window."""

        # Default trials.
        if trs is None:
            all_trs = np.ones(self.n_trials(), dtype=bool)
            trs = Trials(all_trs, 'all trials')  # all trials

        # Default time limits.
        t1, t2 = self.init_time_limits(t1, t2)

        # Select spikes between t1 and t2 during selected trials.
        spikes = [util.values_in_window(spk, t1, t2)
                  for spk in compress(self.spikes, trs.trials)]

        # Convert them into new SpikeTrain list, with time limits set.
        spikes = [SpikeTrain(spk, t_start=t1, t_stop=t2) for spk in spikes]

        return spikes

    # %% Methods for summary statistics over spikes.

    def spike_stats_in_prd(self, trs=None, t1=None, t2=None):
        """Return spike count and rate of given trials within time window."""

        # Default time limits.
        t1, t2 = self.init_time_limits(t1, t2)

        # Rescale time limits.
        t1 = t1.rescale(s)
        t2 = t2.rescale(s)

        # Select spikes.
        spikes = self.get_spikes(trs, t1, t2)

        # Calculate rate.
        n_spikes = np.array([spk.size for spk in spikes])
        f_rate = n_spikes / (t2 - t1)

        return n_spikes, f_rate

    def spike_count_stats(self, trs=None, t1=None, t2=None):
        """Return spike count statistics across selected trials."""

        # Get rates in specified trials within specified time window.
        f_rate = self.spike_stats_in_prd(trs, t1, t2)[1]

        # Calculate statistics.
        mean_rate = np.mean(f_rate)
        std_rate = np.std(f_rate)
        sem_rate = std_rate / np.sqrt(f_rate.size)

        # Put them into a Series.
        spike_stats = pd.Series([mean_rate, std_rate, sem_rate],
                                index=['mean', 'std', 'sem'])

        return spike_stats

    def isi(self, trs=None, t1=None, t2=None):
        """Return interspike intervals per trial."""

        spikes = self.get_spikes(trs, t1, t2)
        isi = [statistics.isi(spk) for spk in spikes]

        return isi
