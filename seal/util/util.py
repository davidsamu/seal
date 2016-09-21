# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 10:33:15 2016

Collection of utility functions.

@author: David Samu
"""

import os
import warnings
import pickle
import datetime

import string
import numpy as np
import scipy as sp
import multiprocessing as mp
from collections import Iterable

from scipy import stats

from quantities import Quantity, deg


# %% Input / output functions.

def read_matlab_object(f, obj_name=None):
    """Return Matlab structure or object from it."""

    mat_struct = sp.io.loadmat(f, struct_as_record=False, squeeze_me=True)
    if obj_name is not None:
        mat_struct = mat_struct[obj_name]
    return mat_struct


def read_objects(f, obj_names):
    """Read in objects from pickled data file."""

    data = pickle.load(open(f, 'rb'))

    # Unload objects from dictionary.
    if isinstance(obj_names, str):
        objects = data[obj_names]   # single object
    else:
        objects = [data[oname] for oname in obj_names]  # multiple objects

    return objects


def write_objects(obj_dict, f):
    """Write out dictionary object into pickled data file."""

    create_dir(f)
    pickle.dump(obj_dict, open(f, 'wb'))


def write_table(dataframe, excel_writer, **kwargs):
    """Write out Pandas dataframe as Excel table."""

    if excel_writer is not None:
        dataframe.to_excel(excel_writer, **kwargs)
        excel_writer.save()


def get_latest_file(dir_name, ext='.data'):
    """Return name of latest file from folder."""

    fname = ''
    if os.path.isdir(dir_name):  # if folder exists
        fnames = [f for f in os.listdir(dir_name) if f.endswith(ext)]
        if len(fnames) > 0:
            fname = max(fnames)  # max is the latest (most recently created)

    return fname


# %% String formatting functions.

def format_to_fname(s):
    """Format string to file name compatible string."""

    valid_chars = "_ %s%s" % (string.ascii_letters, string.digits)
    fname = ''.join(c for c in s if c in valid_chars)
    fname = fname.replace(' ', '_')
    return fname


def format_pvalue(pval):
    """Format a p-value into readable string."""

    if pval < 10**-4:
        pstr = 'p < 0.0001'
    elif pval < 10**-3:
        pstr = 'p < 0.001'
    else:
        pstr = '{:.3f}'.format(pval)

    return pstr


# %% System-related functions.

def run_in_pool(f, params, nCPU=None):
    """Run a function in parallel pool."""

    if nCPU is None:  # set number of cores
        nCPU = mp.cpu_count() - 1

    with mp.Pool(nCPU) as p:
        res = p.starmap(f, params)

    return res


def create_dir(f):
    """Create directory if it does not already exist."""

    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


def timestamp():
    """Returns time stamp."""

    now = datetime.datetime.now()
    timestamp = '{:%Y%m%d_%H%M%S}'.format(now)
    return timestamp


# %% Basic data manipulation functions.

def is_iterable(obj):
    """Check if object is iterable (and not a string)."""

    iterable = isinstance(obj, Iterable)
    string = isinstance(obj, str)
    np_scalar = isinstance(obj, (np.ndarray, Quantity)) and len(obj.shape) == 0

    return iterable and not string and not np_scalar


def indices(vec, val):
    """Return all indices of value in vector."""

    idxs = np.where(vec == val)[0]
    return idxs


def index(vec, val):
    """Return (first) index of value in vector."""

    # Get all indices.
    idxs = indices(vec, val)

    # Check if val is in vec.
    if len(idxs) > 0:
        idx = idxs[0]
    else:
        idx = None

    return idx


def indices_in_window(v, vmin=None, vmax=None):
    """Return indices of values between min and max values."""
    if vmin is None:
        vmin = -np.inf
    if vmax is None:
        vmax = np.inf
    return np.logical_and(v >= vmin, v <= vmax)


def values_in_window(v, vmin=None, vmax=None):
    """Return values between min and max values."""
    return v[indices_in_window(v, vmin, vmax)]


def quantity_linspace(q1, q2, unit, n, endpoint=True, retstep=False):
    """Implement numpy.linspace on phyisical quantities."""

    v1 = np.array(q1.rescale(unit))
    v2 = np.array(q2.rescale(unit))
    vec = np.linspace(v1, v2, n, endpoint, retstep) * unit
    return vec


def zscore_timeseries(timeseries):
    """Z-score set of time series at each time point (per column)."""

    zscored_ts = stats.zscore(timeseries)
    zscored_ts[np.isnan(zscored_ts)] = 0  # remove NaN values

    return zscored_ts


# %% Function for analysing directions.

def cart2pol(x, y):
    """Perform convertion from Cartesian to polar coordinates."""

    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    """Perform convertion from polar to Cartesian coordinates."""

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def deg_mod(d, max_d=360*deg):
    """Converts cirulcar value (degree) into modulo interval."""

    d = d.rescale(deg)
    d_mod = (d.magnitude % max_d.magnitude) * d.units

    return d_mod


def deg_diff(d1, d2):
    """Return difference between two angles in degrees."""

    d1 = deg_mod(d1)
    d2 = deg_mod(d2)
    d = abs(d1-d2)
    d = d if d < 180*deg else 360*deg - d
    return d


# %% General statistics functions.

def SNR(v):
    """Returns signal to noise ratio of values."""

    snr = np.mean(v) / np.std(v)
    return snr


def mean_rate(rates):
    """Return mean and SEM of firing rates."""

    # Calculate mean and SEM.
    rate_mean = np.mean(rates, 0)
    rate_sem = np.std(rates, 0) / np.sqrt(rates.shape[0])

    return rate_mean, rate_sem


def sign_diff(ts1, ts2, p):
    """
    Return times of significant difference
    between two sets of time series.
    """

    lr1 = ts1.shape[1]
    lr2 = ts2.shape[1]

    if lr1 != lr2:
        warnings.warn('Unequal lengths ({} and {}).'.format(lr1, lr2))

    # Calculate p-values and times of significant difference.
    pvals = np.array([stats.ttest_ind(ts1[:, i], ts2[:, i], equal_var=False)[1]
                      for i in range(min(lr1, lr2))])
    tsign = pvals < p

    return pvals, tsign


def periods(t_on, time=None, min_len=None):
    """Return list of periods where t_on is True and with minimum length."""

    if time is None:
        time = np.array(range(t_on))

    if len(t_on) != len(time):
        warnings.warn('Lengths of t_on ({}) and time differ ({})!'
                      .format(len(t_on), len(time)))

    # Starts of periods.
    tstarts = np.insert(t_on, 0, False)
    istarts = np.logical_and(tstarts[:-1] == False, tstarts[1:] == True)

    # Ends of periods.
    tends = np.insert(t_on, -1, False)
    iends = np.logical_and(tends[:-1] == True, tends[1:] == False)

    # Zip (start, end) pairs of periods.
    pers = [(t1, t2) for t1, t2 in zip(time[istarts], time[iends])]

    # Drop too short periods
    if min_len is not None:
        pers = [(t1, t2) for t1, t2 in pers if t2-t1 >= min_len]

    return pers


def sign_periods(ts1, ts2, time, p):
    """
    Return list of periods of significantly difference
    between sets of time series.
    """

    # Indices of significant difference.
    tsign = sign_diff(ts1, ts2, p)[1]
    # Periods of significant difference.
    sign_periods = periods(tsign, time)

    return sign_periods
