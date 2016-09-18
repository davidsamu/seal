#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:56:46 2016

Class for array of spike trains.

@author: David Samu
"""

import numpy as np
from itertools import compress
from neo import SpikeTrain
from elephant import statistics

from seal.util import util
from seal.object.trials import Trials


class Spikes:
    """Class for array of spike trains and associated properties."""

    # %% Constructor
    def __init__(self, spike_train_list, t_start=None, t_stop=None):
        """Create a Spikes instance."""

        # Create empty instance.
        self.spikes = None

        # Remove spikes outside of time window [t_start, t_stop].
        tr_sp = [util.values_in_window(sp, t_start, t_stop)
                 for sp in spike_train_list]
        # Create list of neo SpikeTrain objects.
        self.spikes = [SpikeTrain(sp, t_start=t_start, t_stop=t_stop)
                       for sp in tr_sp]

    # %% Utility methods.

    def n_trials(self):
        """Return number of trials."""

        n_trials = len(self.spikes)
        return n_trials

    def get_spikes(self, trials=None, t1=None, t2=None):
        """Return spike times of given trials within time window."""

        # Set default values.
        if isinstance(trials, list):
            trials = trials[0]
        if trials is None:
            all_trials = np.ones(self.n_trials(), dtype=bool)
            trials = Trials(all_trials, 'all trials')  # all trials
        if t1 is None:
            t1 = -np.inf  # no lower limit
        if t2 is None:
            t2 = np.inf   # no upper limit

        # Select spikes between t1 and t2 during selected trials.
        spikes = [util.values_in_window(sp, t1, t2) for sp
                  in compress(self.spikes, trials.trials)]
        return spikes

    # %% Methods for summary statistics over spikes.

    def n_spikes(self, trials=None, t1=None, t2=None):
        """Return number of spikes of given trials within time window."""

        spikes = self.get_spikes(trials, t1, t2)
        n_spikes = np.array([sp.size for sp in spikes])
        return n_spikes

    def avg_n_spikes(self, trials=None, t1=None, t2=None):
        """Return average number of spikes across selected trials."""

        n_spikes = self.n_spikes(trials, t1, t2)
        avg_n_spikes = np.mean(n_spikes)
        return avg_n_spikes

    def isi(self, trials=None, t1=None, t2=None):
        """Return interspike intervals per trial."""

        spikes = self.get_spikes(trials, t1, t2)
        isi = [statistics.isi(sp) for sp in spikes]
        return isi
