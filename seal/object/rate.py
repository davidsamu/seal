# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 17:05:15 2016

Class for storing firing rates and associated properties.

@author: David Samu
"""

import numpy as np
from quantities import s, ms, Hz

from elephant.statistics import instantaneous_rate

from seal.util import util


class Rate:
    """
    Class for storing firing rates and associated properties.
    """

    # %% Constructor
    def __init__(self, kernel, spikes, step=10*ms, min_rate=0.01):
        """Create a Rate instance."""

        # Create empty instance.
        self.name = ''
        self.rates = None
        self.times = None
        self.kernel = None

        # Calculate, scale and trim rate.
        rates = [instantaneous_rate(sp, step, kernel)
                 for sp in spikes]
        rhz = [r[:, 0].rescale(Hz) for r in rates]  # rescale to all to Hz
        rhz = np.maximum(np.array(rhz), 0) * Hz     # zero out negative values
        if min_rate is not None:
            rhz[rhz < min_rate*Hz] = 0 * Hz              # zero out tiny values

        # Stor rate values and sample times.
        self.rates = rhz
        self.times = rates[0].times.rescale(s)
        self.kernel = kernel

    # %% Methods to get times and rates for given time windows and trials.

    def _get_indices(self, t1=None, t2=None):
        """Returns indices within time window."""

        # Set default values
        if t1 is None:
            t1 = self.times[0]
        if t2 is None:
            t2 = self.times[-1]

        t_idx = util.indices_in_window(self.times, t1, t2)
        return t_idx

    def get_times(self, t1=None, t2=None):
        """Return times of rates between t1 and t2."""

        # Select indices and get corresponsing times.
        t_idx = self._get_indices(t1, t2)
        rate_times = self.times[t_idx]

        return rate_times

    def get_rates(self, trials=None, t1=None, t2=None):
        """Return firing rates of some trials within time window."""

        # Set default values.
        if trials is None:
            trials = np.ones(len(self.rates), dtype=bool)

        # Select requested trials and keep only rates between t1 and t2.
        t_idx = self._get_indices(t1, t2)
        rates = self.rates[np.ix_(trials, t_idx)]

        return rates

    def get_rates_and_times(self, trials=None, t1=None, t2=None):
        """Return firing rates and times of some trials within time window."""

        rates = self.get_rates(trials, t1, t2)
        times = self.get_times(t1, t2)

        return rates, times
