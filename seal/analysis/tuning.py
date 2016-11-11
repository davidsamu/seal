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
from quantities import deg

from seal.util import util


# %% Core analysis methods.

def gaus(x, a=0, b=1, x0=0, sigma=1):
    """Define gaussian function."""

    g = a + b * exp(-(x-x0)**2 / (2*sigma**2))

    return g


# TODO: optionally, provide bounds and init values for fit? 
# TODO: check if bounds and init values are consistent, change them otherwise!
def fit_gaus_curve(x, y, y_err=None):
    """
    Fit Gaussian curve to stimulus - response values. Returns best estimate on
    parameter values and their std of error in Pandas data frame.
    """

    # Init fit results.
    columns = ['a', 'b', 'x0', 'sigma', 'FWHM', 'R2', 'RMSE']
    fit_res = pd.DataFrame(index=['fit', 'std err'], columns=columns)

    # Remove NaN values from input.
    idx = np.logical_not(np.logical_or(np.isnan(x), np.isnan(y)))
    if y_err is not None:
        idx = np.logical_and(idx, np.logical_not(np.isnan(y_err)))
        y_err = y_err[idx]
    x, y = x[idx], y[idx]

    # Save input dimensions
    x_dim, y_dim = x.units, y.units

    # Check thet there is non-NaN data
    # and that Y values are not all 0 (eg. no response).
    if x.size > 1 or not np.all(y == 0):

        # Init input params.
        x, y = np.array(x), np.array(y)
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)

        # Every element of y_err has to be positive.
        if np.all(y_err <= 0):
            y_err = None
        else:
            y_err[y_err <= 0] = np.min(y_err[y_err > 0])  # def. value: minimum

        # Set initial values.
        a_init = ymin                    # baseline (vertical shift)
        b_init = ymax - ymin             # height (vertical stretch)
        x0_init = np.sum(x*y)/np.sum(y)  # mean (horizontal shift)
        # spread (horizontal stretch)
        sigma_init = np.sqrt(np.sum(y*(x-x0_init)**2)/np.sum(y))
        p_init = [a_init, b_init, x0_init, sigma_init]

        # Lower and upper bounds of variables.
        bounds = ([0.8*ymin,   0,               xmin, 0.],
                  [np.mean(y), 1.2*(ymax-ymin), xmax, (xmax-xmin)/2])
        
        # Find optimal Gaussian curve fit,
        # using ‘trf’ (Trust Region Reflective) method.
        p_opt, p_cov = curve_fit(gaus, x, y, p0=p_init, bounds=bounds,
                                 sigma=y_err, absolute_sigma=True)

        # Errors on (standard deviation of) parameter estimates.
        p_err = np.sqrt(np.diag(p_cov))
        
        # Convert results into Pandas dataframe.
        fit_res.loc['fit', columns[0:4]] = p_opt
        fit_res.loc['std err', columns[0:4]] = p_err

        # Add additional metrics of fit.
        a, b, x0, sigma = p_opt
        fit_res.loc['fit', 'FWHM'] = 2 * np.sqrt(2*np.log(2)) * sigma * x_dim

        R2, RMSE = calc_R2_RMSE(x, y, gaus, a=a, b=b, x0=x0, sigma=sigma)
        fit_res.loc['fit', 'R2'] = R2
        fit_res.loc['fit', 'RMSE'] = RMSE

    fit_res.a = util.add_dim_to_df_col(fit_res.a, y_dim)
    fit_res.b = util.add_dim_to_df_col(fit_res.b, y_dim)
    fit_res.x0 = util.add_dim_to_df_col(fit_res.x0, x_dim)
    fit_res.sigma = util.add_dim_to_df_col(fit_res.sigma, x_dim)    

    return fit_res
    

# %% Miscullaneous functions.

def center_pref_dir(dirs, PD, meanFR=None, semFR=None):
    """Center preferred direction by shift direction - response values."""

    # Init.
    ndirs = len(dirs)
    idx = np.array(range(ndirs))

    # Center preferred direction.
    dirs_offset = dirs - PD

    # Reorganise direction and response arrays to even number of values
    # to left and right (e.g. 4 - 4 for 8 directions).
    to_right = dirs_offset < -180*deg   # indices to move to the right
    to_left = dirs_offset > 180*deg     # indices to move to the left
    center = np.invert(np.logical_or(to_right, to_left))  # indices to keep
    idx = np.hstack((idx[to_left], idx[center], idx[to_right]))

    # Shift directions.
    dirs_ctrd = util.deg_mod(dirs_offset[idx])
    idx_to_flip = dirs_ctrd > 180*deg
    dirs_ctrd[idx_to_flip] = dirs_ctrd[idx_to_flip] - 360*deg

    # Shift responses.
    meanFR_ctrd = None
    if meanFR is not None:
        meanFR_ctrd = meanFR[idx]
    semFR_ctrd = None
    if semFR is not None:
        semFR_ctrd = semFR[idx]

    return dirs_ctrd, meanFR_ctrd, semFR_ctrd


# TODO: this could be sped up by doing the fit on the whole vector and then adding quantity dimension!
def gen_fit_curve(f, stim_units, stim_min, stim_max, n=100, **f_kwargs):
    """Generate data points for plotting fitted tuning curve."""
    
    # Generate synthetic stimulus-response data points.
    if stim_units is not None:        
        xfit = util.quantity_linspace(stim_min, stim_max, stim_units, n)
    else:
        xfit = np.linspace(stim_min, stim_max, n)
    yfit = np.array([f(xi, **f_kwargs) for xi in xfit])

    return xfit, yfit

    
# TODO: calculate R2 and RMSE on all samples (rather than means to direction?)
def calc_R2_RMSE(x, y, f, **f_kwargs):
    """Calculate root mean squared error and R-squared value of fit."""
    
    # Init.
    n, nparams = x.size, len(f_kwargs)
    
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
    