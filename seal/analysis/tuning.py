#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:52:52 2016

Collection of functions related to tuning and
other stimulus response properties of units.

@author: David Samu
"""

import numpy as np
import pandas as pd
from scipy import exp
from scipy.optimize import curve_fit

from seal.util import util, plot


# %% Core analysis methods.

def gaus(x, a=0, b=1, x0=0, sigma=1):
    """Define gaussian function."""

    g = a + b * exp(-(x-x0)**2 / (2*sigma**2))

    return g


def fit_gaus_curve(x, y, y_err=None):
    """
    Fit Gaussian curve to stimulus - response values. Returns best estimate on
    parameter values and their std of error in Pandas data frame.
    """

    # Init
    x_dim, y_dim = x.units, y.units
    x, y = np.array(x), np.array(y)
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    # Every element of y_err has to be positive.
    if np.all(y_err <= 0):
        y_err = None
    else:
        y_err[y_err <= 0] = np.min(y_err[y_err > 0])  # default value: minimum

    # Set initial values.
    a_init = ymin                    # baseline (vertical shift)
    b_init = ymax - ymin             # height (vertical stretch)
    x0_init = np.sum(x*y)/np.sum(y)  # mean (horizontal shift)
    # spread (horizontal stretch)
    sigma_init = np.sqrt(np.sum(y*(x-x0_init)**2)/np.sum(y))
    p_init = [a_init, b_init, x0_init, sigma_init]

    # Lower and upper bounds of variables.
    bounds = ([0.8*ymin,   0,               xmin, 0.],
              [np.mean(y), 1.2*(ymax-ymin), xmax, xmax-xmin])

    # Find optimal Gaussian curve fit,
    # using ‘trf’ (Trust Region Reflective) method.
    p_opt, p_cov = curve_fit(gaus, x, y, p0=p_init, bounds=bounds,
                             sigma=y_err, absolute_sigma=True)

    # Errors on (standard deviation of) parameter estimates.
    p_err = np.sqrt(np.diag(p_cov))

    # Convert results into Pandas dataframe and add back units.
    fit_res = pd.DataFrame([p_opt, p_err], index=['fit', 'std err'],
                           columns=['a', 'b', 'x0', 'sigma'])
    fit_res.a = util.add_dim_to_df_col(fit_res.a, y_dim)
    fit_res.b = util.add_dim_to_df_col(fit_res.b, y_dim)
    fit_res.x0 = util.add_dim_to_df_col(fit_res.x0, x_dim)
    fit_res.sigma = util.add_dim_to_df_col(fit_res.sigma, x_dim)

    return fit_res


def test_tuning(stim, mean_resp, sem_resp, do_plot=True,
                stim_min=None, stim_max=None, **kwargs):
    """Test tuning of stimulus - response data."""

    # Fit tuning curve to stimulus - response.
    # Currently only fits gaussian tunining curve.
    fit_res = fit_gaus_curve(stim, mean_resp, sem_resp)

    # Plot stimulus - response pairs and fitted tuning curve.
    ax = None
    if do_plot:
        # Generate data points for plotting fitted tuning curve.
        n = 500  # number of data points to sample from fitted function
        a, b, x0, sigma = [v.magnitude for v in fit_res.loc['fit']]
        if stim_min is None:
            stim_min = np.min(stim)
        if stim_max is None:
            stim_max = np.max(stim)
        xfit = util.quantity_linspace(stim_min, stim_max, stim.units, n)
        yfit = np.array([gaus(xi, a, b, x0, sigma) for xi in np.array(xfit)])
        ax = plot.tuning_curve(stim, mean_resp, sem_resp, xfit, yfit, **kwargs)
    return fit_res, ax


def compare_tuning_curves(stim_resp_dict, do_plot=True,
                          stim_min=None, stim_max=None, **kwargs):
    """Compare tuning curves across list of stimulus - responses pairs."""

    colors = plot.get_colors()
    tuning_res = pd.DataFrame(columns=['a', 'b', 'x0', 'sigma'])
    for name, values in stim_resp_dict.items():
        stim, mean_resp, sem_resp = values[:3]
        fit_res, ax = test_tuning(stim, mean_resp, sem_resp,
                                  stim_min=stim_min, stim_max=stim_max,
                                  do_plot=do_plot, color=next(colors), **kwargs)
        tuning_res.loc[name] = fit_res.loc['fit']

    return tuning_res
