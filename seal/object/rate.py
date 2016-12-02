# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 17:05:15 2016

Class for storing firing rates and associated properties.

@author: David Samu
"""

import warnings
import numpy as np
import pandas as pd
from quantities import ms, Hz

from elephant.statistics import instantaneous_rate
from elephant.kernels import RectangularKernel

from seal.util import util


class Rate:
    """Class for storing firing rates per trial and associated properties."""

    # %% Constructor.
    def __init__(self, name, kernel, spikes, step=10*ms, min_rate=0.01*Hz,
                 rdim=Hz, tdim=ms):
        """Create a Rate instance."""

        # Create empty instance.
        self.name = ''
        self.rates = None
        self.kernel = None
        self.step = None
        self.tdim = tdim
        self.rdim = rdim

        # Calculate firing rates and extract sampling times.
        with warnings.catch_warnings():
            # Let's ignore warnings about negative firing rate values,
            # they are fixed below.
            warnings.simplefilter("ignore")
            rate_list = [instantaneous_rate(sp, step, kernel) for sp in spikes]
            tvec = rate_list[0].times.rescale(tdim)

        # Extract rates and rescale them to Hz.
        rates = [r[:, 0].rescale(rdim) for r in rate_list]

        # Convert to DataFrame.
        rates = pd.DataFrame(np.array(rates), columns=tvec)

        # Zero out negative and tiny positive values.
        if min_rate is not None:
            rates[rates < float(min_rate.rescale(rdim))] = 0

        # Store rate values and kernel.
        self.name = name
        self.rates = rates
        self.kernel = kernel
        self.step = step

    # %% Kernel query methods.

    def kernel_type(self):
        """Return type of kernel."""

        ktype = type(self.kernel)
        return ktype

    def kernel_sigma(self):
        """Return sigma of Gaussian kernel or width of Rectangular kernel."""

        sigma = self.kernel.sigma

        # Rectangular kernel.
        if isinstance(self.kernel, RectangularKernel):
            width = util.rect_width_from_sigma(sigma)
            return width

        # Gaussian kernel.
        else:
            return sigma

    # %% Methods to get times and rates for given time windows and trials.

    def get_sample_times(self, t1=None, t2=None):
        """Return times of rates between t1 and t2."""

        if t1 is not None:
            t1 = t1.rescale(self.tdim)
        if t2 is not None:
            t2 = t2.rescale(self.tdim)

        sample_times = util.values_in_window(self.rates.columns, t1, t2)

        return sample_times

    def get_rates(self, trs=None, t1=None, t2=None):
        """Return firing rates of some trials within time window."""

        # Set default values.
        if trs is None:
            trs = np.ones(len(self.rates), dtype=bool)

        # Select rates from requested trials
        # and between t1 and t2 (or whole time period).
        tvec = self.get_sample_times(t1, t2)
        rates = self.rates.loc[trs, tvec]

        return rates
