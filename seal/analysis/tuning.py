#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of functions related to tuning and
other stimulus response properties of units.

@author: David Samu
"""

import numpy as np
import pandas as pd
from scipy import exp
from scipy.optimize import curve_fit


# %% Core analysis methods.

def gaus(x, a=0, b=1, x0=0, sigma=1):
    """Define gaussian function."""

    g = a + b * exp(-(x-x0)**2 / (2*sigma**2))

    return g


def fit_gaus_curve(x, y, y_err=None, lower_bounds=None, upper_bounds=None,
                   a_init=None, b_init=None, x_init=None, sigma_init=None):
    """
    Fit Gaussian curve to stimulus - response values. Return best estimate on
    parameter values and their std of error in Pandas data frame.
    """

    # Init fit results.
    fit_params = pd.DataFrame(index=['fit', 'std err'],
                              columns=['a', 'b', 'x0', 'sigma'])
    fit_res = pd.Series(index=['FWHM', 'R2', 'RMSE'], dtype=object)

    # Remove NaN values from input.
    idx = (~pd.isnull(x) & ~pd.isnull(y)).index
    if y_err is not None:
        idx = (idx & ~pd.isnull(y_err).index)
        y_err = y_err[idx]
    x, y = x[idx], y[idx]

    # Check that there is non-NaN data
    # and that Y values are not all 0 (i.e. no response).
    if x.size > 1 and not np.all(y == 0):

        # Init input params.
        x, y = np.array(x), np.array(y)
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)

        # Every element of y_err has to be positive.
        if y_err is not None:
            if np.all(y_err <= 0):
                y_err = None
            else:
                y_err[y_err <= 0] = np.min(y_err[y_err > 0])  # default: min.

        # Set initial values.
        if a_init is None:
            a_init = ymin                    # baseline (vertical shift)
        if b_init is None:
            b_init = ymax - ymin             # height (vertical stretch)
        if x_init is None:
            x0_init = np.sum(x*y)/np.sum(y)  # mean (horizontal shift)
        if sigma_init is None:               # spread (horizontal stretch)
            sigma_init = np.sqrt(np.sum(y*(x-x0_init)**2)/np.sum(y))
        p_init = [a_init, b_init, x0_init, sigma_init]

        # Lower and upper bounds of variables.
        bounds = ([0.8*ymin,   0,               xmin, 0],
                  [np.mean(y), 1.2*(ymax-ymin), xmax, (xmax-xmin)/2])
        for ibnds, bnds in enumerate([lower_bounds, upper_bounds]):
            if bnds is not None:
                for i, b in enumerate(bnds):
                    if b is not None:
                        bounds[ibnds][i] = b

        # Find optimal Gaussian curve fit,
        # using ‘trf’ (Trust Region Reflective) method.
        p_opt, p_cov = curve_fit(gaus, x, y, p0=p_init, bounds=bounds,
                                 sigma=y_err, absolute_sigma=True)

        # Errors on (standard deviation of) parameter estimates.
        p_err = np.sqrt(np.diag(p_cov))

        # Put fit parameters into Pandas dataframe.
        fit_params.loc['fit'] = p_opt
        fit_params.loc['std err'] = p_err

        # Calculate additional metrics of fit.
        a, b, x0, sigma = p_opt
        FWHM = 2 * np.sqrt(2*np.log(2)) * sigma
        R2, RMSE = calc_R2_RMSE(x, y, gaus, a=a, b=b, x0=x0, sigma=sigma)

        # Put them into a Series.
        fit_res.loc['FWHM'] = FWHM
        fit_res.loc['R2'] = R2
        fit_res.loc['RMSE'] = RMSE

    return fit_params, fit_res


# %% Miscullaneous functions.

def gen_fit_curve(f, stim_min, stim_max, n=100, **f_kwargs):
    """Generate data points for plotting fitted tuning curve."""

    # Sample stimulus values uniformaly within interval.
    xfit = np.linspace(stim_min, stim_max, n)

    # Generate response values using tuning curve fit.
    yfit = f(xfit, **f_kwargs)

    return xfit, yfit


def calc_R2_RMSE(x, y, f, **f_kwargs):
    """Calculate root mean squared error and R-squared value of fit."""

    # Init.
    n, nparams = x.size, len(f_kwargs)

    # Not enough data samples.
    if nparams >= n:
        return np.nan, np.nan

    # Calculate
    ymean = np.mean(y)
    yfit = f(x, **f_kwargs)
    resid = y - yfit

    # R-squared.
    SS_res = np.sum(resid**2)
    SS_tot = np.sum((y-ymean)**2)
    R2 = 1 - SS_res / SS_tot

    # RMSE.
    MSE = SS_res / (n - nparams)
    RMSE = MSE**0.5

    return R2, RMSE
