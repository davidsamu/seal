# -*- coding: utf-8 -*-
"""
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
    def __init__(self, name, kernel, spikes, step=10*ms, min_rate=0.01*Hz):
        """Create a Rate instance."""

        # Create empty instance.
        self.name = name
        self.rates = None
        self.tvec = None
        self.kernel = kernel
        self.step = step

        # Calculate firing rates.
        with warnings.catch_warnings():
            # Let's ignore warnings about negative firing rate values,
            # they are fixed below.
            warnings.simplefilter('ignore', UserWarning)

            rates = len(spikes) * [[]]
            for i, sp in enumerate(spikes):

                # Estimate firing rates.
                rts = instantaneous_rate(sp, step, kernel)
                rates[i] = pd.Series(np.array(rts)[:, 0],
                                     index=rts.times.rescale(ms))

        # Stack rate vectors into dataframe, adding NaNs to samples missing
        # from any trials.
        rates = pd.concat(rates, axis=1).T

        # Zero out negative and tiny positive values.
        if min_rate is not None:
            rates[rates < float(min_rate)] = 0

        # Store rate and time sample values.
        self.rates = rates
        self.tvec = np.array(rates.columns) * ms

    # %% Kernel query methods.

    def kernel_type(self):
        """Return kernel type."""

        ktype = type(self.kernel)
        return ktype

    def kernel_sigma(self):
        """
        Return kernel width (Rectangular kernel) or sigma (Gaussian kernel).
        """

        sigma = self.kernel.sigma

        # Rectangular kernel.
        if isinstance(self.kernel, RectangularKernel):
            width = util.rect_width_from_sigma(sigma)
            return width

        # Any other, e.g. Gaussian, kernel.
        return sigma

    # %% Methods to get sample times for given time periods.

    def get_sampled_t_limits(self, t1=None, t2=None):
        """Return sampled time limits."""

        ts1 = float(util.find_nearest(self.tvec, t1))
        ts2 = float(util.find_nearest(self.tvec, t2))
        return ts1, ts2

    def get_sample_times(self, t1=None, t2=None):
        """Return sample times between t1 and t2."""

        if t1 is not None:
            t1 = t1.rescale(self.tvec.units)
        if t2 is not None:
            t2 = t2.rescale(self.tvec.units)

        sample_times = util.values_in_window(self.tvec, t1, t2)

        return sample_times

    def get_sample_times_list(self, t1s=None, t2s=None):
        """Return list of sample times for each pair of t1s and t2s."""

        stvec = [self.get_sample_times(t1, t2) for t1, t2 in zip(t1s, t2s)]
        return stvec

    # %% Methods to get rates for given trials and time periods.

    def get_rates(self, trs, t1s, t2s, ref_ts=None, tstep=None,
                  rem_all_nan_ts=True):
        """
        Return firing rates of some trials within trial-specific time windows.

        trs:      List with indices of trials to select.
        t1s, t2s: Time window per trial. They must contain all trials!
        ref_ts:   Array of reference times to align rate vectors by.
        """

        # Set default trials.
        if trs is None:
            print('No trial set has been passed. Returning all trials.')
            trs = np.arange(len(self.rates))

        # Select times corresponding to selected trials.
        t1s, t2s = t1s[trs], t2s[trs]
        if ref_ts is not None:
            ref_ts = ref_ts[trs]

        # Select rates from some trials between trial-specific time limits.
        rates = len(trs) * [[]]
        for i, itr in enumerate(trs):
            ts1, ts2 = self.get_sampled_t_limits(t1s.iloc[i], t2s.iloc[i])
            istep = int(tstep/self.step)
            rates[i] = self.rates.loc[itr, ts1:ts2:istep]

        # Align rates relative to reference times.
        if ref_ts is None:  # default: align to start of each time window
            ref_ts = [r.index[0] for r in rates]
        else:
            ref_ts = [float(util.find_nearest(self.tvec, t)) for t in ref_ts]

        for i, r in enumerate(rates):
            r.index = r.index - ref_ts[i]

        # Stack rate vectors into dataframe, adding NaNs to samples missing
        # from some trials.
        rates = pd.concat(rates, axis=1).T if len(rates) else pd.DataFrame()

        # Remove time points with all NaN rates (no measurement?)
        if rem_all_nan_ts:
            rates = rates.loc[:, ~rates.isnull().all()]

        return rates
