# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 17:05:15 2016

Class for storing firing rates and associated properties.

@author: David Samu
"""

import warnings
import numpy as np
from quantities import s, ms, Hz

from elephant.statistics import instantaneous_rate

from seal.util import util


# TODO: combine rates and times into DataFrame. Consider similar refactoring in other Seal objects.
# TODO: add query methods to kernel (width, type, step, etc)

class Rate:
    """Class for storing firing rates per trial and associated properties."""

    # %% Constructor.
    def __init__(self, name, kernel, spikes, step=10*ms, min_rate=0.01):
        """Create a Rate instance."""

        # Create empty instance.
        self.name = ''
        self.rates = None
        self.times = None
        self.kernel = None

        # Calculate, scale and trim rate.
        with warnings.catch_warnings():
            # Let's ignore warnings about negative firing rate values,
            # they are corrected below.
            warnings.simplefilter("ignore")
            rates = [instantaneous_rate(sp, step, kernel)
                     for sp in spikes]

        rhz = [r[:, 0].rescale(Hz) for r in rates]  # rescale all to Hz
        rhz = np.maximum(np.array(rhz), 0) * Hz     # zero out negative values
        if min_rate is not None:
            rhz[rhz < min_rate*Hz] = 0 * Hz         # zero out tiny values

        # Store rate values and sample times.
        self.name = name
        self.rates = rhz
        self.times = rates[0].times.rescale(s)
        self.kernel = kernel

    # %% Methods to get times and rates for given time windows and trials.

    def get_times(self, t1=None, t2=None):
        """Return times of rates between t1 and t2."""

        rate_times = util.values_in_window(self.times, t1, t2)
        return rate_times

    def get_rates(self, trials=None, t1=None, t2=None):
        """Return firing rates of some trials within time window."""

        # Set default values.
        if trials is None:
            trials = np.ones(len(self.rates), dtype=bool)

        # Select requested trials and keep only rates between t1 and t2.
        t_idx = util.indices_in_window(self.times, t1, t2)
        rates = self.rates[np.ix_(trials, t_idx)]

        return rates

    # TODO: see above!!
    def get_rates_and_times(self, trials=None, t1=None, t2=None):
        """Return firing rates and times of some trials within time window."""

        rates = self.get_rates(trials, t1, t2)
        times = self.get_times(t1, t2)

        return rates, times
