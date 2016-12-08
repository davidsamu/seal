# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 10:33:15 2016

Collection of utility functions.

@author: David Samu
"""

import os
import copy
import warnings
import pickle
import datetime

import string
import numpy as np
import scipy as sp
import pandas as pd
import multiprocessing as mp
from collections import Iterable
from collections import OrderedDict as OrdDict

from quantities import Quantity, deg, rad, ms
from elephant.kernels import GaussianKernel, RectangularKernel

# TODO: to be split into submodules!


# %% Input / output and object manipulation functions.

def read_matlab_object(f, obj_names=None):
    """Return Matlab structure or object from it."""

    mat_struct = sp.io.loadmat(f, struct_as_record=False, squeeze_me=True)

    # If not specified, return all objects.
    if obj_names is None:
        obj_names = mat_struct.keys()

    # Return a single object.
    if not is_iterable(obj_names):
        mat_struct = mat_struct[obj_names]

    # Return dictionary of objects.
    else:
        mat_struct = {k: mat_struct[k] for k in obj_names if k in mat_struct}

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
    return


def write_table(dataframe, excel_writer, **kwargs):
    """Write out Pandas dataframe as Excel table."""

    if excel_writer is not None:
        create_dir(excel_writer.path)
        dataframe.to_excel(excel_writer, **kwargs)
        excel_writer.save()
    return


def get_latest_file(dir_name, ext='.data'):
    """Return name of latest file from folder."""

    fname = None
    if os.path.isdir(dir_name):  # if folder exists
        fnames = [f for f in os.listdir(dir_name) if f.endswith(ext)]
        if len(fnames) > 0:
            fname = max(fnames)  # max is the latest (most recently created)

    return fname


def get_copy(obj, deep=True):
    """Returns (deep) copy of object."""

    if deep:
        copy_obj = copy.deepcopy(obj)
    else:
        copy_obj = copy.copy(obj)

    return copy_obj


# %% String formatting functions.

def params_from_fname(fname, nchar_date=6, n_ext=4):
    """Extract recording parameters from file name."""

    # Remove extension and split into parts by '_' underscore character.
    [monkey, dateprobe, task, sortno] = fname[:-n_ext].split('_')
    [date, probe] = [dateprobe[:nchar_date], dateprobe[nchar_date:].upper()]
    return monkey, date, probe, task, sortno


def format_to_fname(s):
    """Format string to file name compatible string."""

    valid_chars = "_ %s%s" % (string.ascii_letters, string.digits)
    fname = ''.join(c for c in s if c in valid_chars)
    fname = fname.replace(' ', '_')
    return fname


def date_to_str(datetime):
    """Convert and return datetime object to string."""

    date_str = datetime.strftime('%m%d%y')
    return date_str


def format_pvalue(pval, max_digit=4):
    """Format a p-value into readable string."""

    if (pval > 1 or pval < 0):
        warnings.warn('Invalid p-value passed: {:.4f}'.format(pval))

    if pval < 10**-4 and max_digit >= 4:
        pstr = 'p < 0.0001'
    elif pval < 10**-3 and max_digit >= 3:
        pstr = 'p < 0.001'
    elif pval < 0.01 and max_digit >= 3:
        pstr = 'p = {:.3f}'.format(pval)
    else:
        pstr = 'p = {:.2f}'.format(pval)

    return pstr


def star_pvalue(pval, n_max_stars=4):
    """Format a p-value into a number of *** stars."""

    if pval < 10**-4 and n_max_stars >= 4:
        pstr = '****'
    elif pval < 10**-3 and n_max_stars >= 3:
        pstr = '***'
    elif pval < 0.01 and n_max_stars >= 2:
        pstr = '**'
    elif pval < 0.05:
        pstr = '*'
    else:
        pstr = '='

    return pstr


def format_offsets(offsets):
    """Return list of degree offset values string formatted."""

    offsets_str = ', '.join([str(int(off))+' deg' for off in offsets])

    return offsets_str


def format_rec_ch_idx(rec_ch_idx):
    """Format (recording, channel, unit idx) triple as string."""

    rec, ch, idx = rec_ch_idx
    rec_ch_idx_str = '{}_Ch{}_{}'.format(rec, ch, idx)

    return rec_ch_idx_str


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
    if d and not os.path.exists(d):
        os.makedirs(d)
    return


def timestamp():
    """Return time stamp."""

    now = datetime.datetime.now()
    timestamp = '{:%Y%m%d_%H%M%S}'.format(now)
    return timestamp


def datestamp():
    """Return date stamp."""

    now = datetime.datetime.now()
    datestamp = '{:%Y%m%d}'.format(now)
    return datestamp


# %% Basic data manipulation functions.

def is_iterable(obj):
    """Check if object is iterable (and not a string)."""

    iterable = isinstance(obj, Iterable)
    string = isinstance(obj, str)
    np_scalar = isinstance(obj, (np.ndarray, Quantity)) and len(obj.shape) == 0

    is_itrbl = iterable and not string and not np_scalar

    return is_itrbl


def is_numpy_array(obj):
    """Check if object is Numpy array."""

    is_np_arr = type(obj).__module__ == np.__name__
    return is_np_arr


def is_numeric_array(obj):
    """Check if object is numeric Numpy array."""

    is_num_arr = is_numpy_array(obj) and np.issubdtype(obj.dtype, np.number)
    return is_num_arr


def is_quantity(obj):
    """Check if object is Quantity array."""

    is_quant = isinstance(obj, Quantity)
    return is_quant


def is_date(obj):
    """Check if object is datetime object or array of datetimes."""

    is_date = (isinstance(obj, datetime.date) or
               (is_iterable(obj) and isinstance(obj[0], datetime.date)))
    return is_date


def fill_dim(v):
    """Add dimension (fill shape) to scalar 'array'."""

    if not v.shape:
        v.shape = (1)
    return v


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


def indices_in_window(v, vmin=None, vmax=None, l_inc=True, r_inc=True):
    """Return indices of values between min and max values."""

    # Default limits.
    if vmin is None:
        vmin = -np.inf
    if vmax is None:
        vmax = np.inf

    # Get indices within limits (inclusive or exclusive).
    idx_small = v >= vmin if l_inc else v > vmin
    idx_large = v <= vmax if r_inc else v < vmax
    idxs = np.array(np.logical_and(idx_small, idx_large))
    idxs = fill_dim(idxs)  # special case of scalar value

    return idxs


def values_in_window(v, vmin=None, vmax=None):
    """Return values between min and max values."""

    v = fill_dim(v)  # special case of scalar value
    v_idxs = v[indices_in_window(v, vmin, vmax)]

    return v_idxs


def zscore_timeseries(timeseries, axis=0):
    """
    Z-score set of time series at each time point (per column).

    axis: Axis along which to operate. If None, compute over the whole array.
    """

    zscored_ts = sp.stats.zscore(timeseries, axis=axis)
    zscored_ts[np.isnan(zscored_ts)] = 0  # remove NaN values

    return zscored_ts


def make_df(row_list, col_names=None):
    """Return Pandas dataframes made of data passed."""

    od_rows = OrdDict(row_list)
    df = pd.DataFrame(od_rows, index=col_names).T
    return df


def series_from_tuple_list(tuple_list):
    """Create Pandas series from list of (name, value) tuples."""

    sp_names, sp_vals = zip(*tuple_list)
    series = pd.Series(sp_vals, index=sp_names)

    return series


def get_scalar_vals(series, remove_dimensions=False):
    """
    Function to extract all non-iterator type value from Series.
    Optionally, remove dimension from quantity values.
    """

    # Select non-iterator type values.
    idxs = [not is_iterable(val) for val in series.values]
    sub_series = series[idxs]

    # Remove dimension from quantity values.
    if remove_dimensions:
        [sub_series.replace(v, float(v), inplace=True)
         for v in sub_series.values if isinstance(v, Quantity)]

    return sub_series


# %% Functions to handle Numpy and Pandas objects with Quantities as elements.

def quantity_linspace(q1, q2, n, dim=None, endpoint=True, retstep=False):
    """Implement numpy.linspace on phyisical quantities."""

    if dim is None:
        dim = q1.units

    v1 = np.array(q1.rescale(dim))
    v2 = np.array(q2.rescale(dim))
    vec = np.linspace(v1, v2, n, endpoint, retstep) * dim
    return vec


def quantity_arange(q1, q2, step):
    """Implement numpy.arange on phyisical quantities."""

    dim = step.units
    v1 = np.array(q1.rescale(dim))
    v2 = np.array(q2.rescale(dim))
    vec = np.arange(v1, v2, float(step)) * dim
    return vec


def list_to_quantity(lvec, dim=None):
    """Convert list or Pandas Series of quantity values to Quantity array."""

    dim = lvec[0].units if dim is None else dim
    np_vec = np.array([v for v in lvec]) * dim

    return np_vec


def add_dim_to_series(ser, dim):
    """Add physical dimension to Pandas Series."""

    qarr = list_to_quantity(ser, dim)
    qser = pd.DataFrame([qarr], columns=ser.index).T

    return qser


def remove_dim_from_array(qvec, dtype=float):
    """Remove dimension from Quantity array and return Numpy array."""

    np_vec = np.array([float(qv) for qv in qvec], dtype=dtype)

    return np_vec


def remove_dim_from_series(qser):
    """Remove physical dimension from Pandas Series."""

    arr = remove_dim_from_array(qser)
    series = pd.Series(arr, index=qser.index, name=qser.name)

    return series


def dim_series_to_array(qser):
    """Convert quantity Series to quantity array."""

    qarr = list_to_quantity(list(qser))

    return qarr


def add_quant_elem(df, row, col, qel):
    """Add quantity element to DataFrame object."""

    df.loc[row, col] = qel  # this creates the column if not in DF yet
    df[col] = df[col].astype(object)
    df.loc[row, col] = qel


def add_quant_col(df, col, colname):
    """Add quantity column to DataFrame object."""

    df[colname] = col
    df[colname] = add_dim_to_series(df[colname], col.units)


# %% Function for analysing directions.

def deg2rad(v_deg):
    """Convert degrees to radians."""

    v_rad = np.pi * v_deg / 180
    return v_rad


def rad2deg(v_rad):
    """Convert radians to degrees."""

    v_deg = 180 * v_rad / np.pi
    return v_deg


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
    d_mod = np.mod(d, max_d)
    return d_mod


def deg_diff(d1, d2):
    """Return difference between two angles in degrees."""

    d1 = deg_mod(d1)
    d2 = deg_mod(d2)
    d = abs(d1-d2)
    d = d if d < 180*deg else 360*deg - d
    return d


def coarse_dir(origd, dirs):
    """Return direction from list that is closest to provided one."""

    deg_diffs = np.array([deg_diff(d, origd) for d in dirs])
    cd = dirs[np.argmin(deg_diffs)]

    return cd


def deg_w_mean(dirs, weights=None, cdirs=None):
    """
    Taking a vector of directions (2D unit vectors) and their weights, returns
        - index of unidirectionality of weights
            + inverse of spread, "direction selectivity", DSI
        - weighted mean of directions
            + "preferred direction", PD
        - coarsed weighted mean of directions
            + "preferred one of the original (or passed) directions", PDc.
    """

    if weights is None:
        weights = np.ones(len(dirs))

    if cdirs is None:
        cdirs = np.unique(dirs)

    # Remove values correspinding to NaN weights.
    idxs = np.logical_not(np.isnan(weights))
    dirs, weights = dirs[idxs], weights[idxs]

    # Uniform zero weights (eg. no response).
    if dirs.size == 0 or np.all(weights == 0):
        return 0, np.nan*deg, np.nan*deg

    # Convert directions to Cartesian unit vectors.
    dirs_xy = np.array([pol2cart(1, d.rescale(rad)) for d in dirs])

    # Calculate mean along x and y dimensions.
    x_mean = np.average(dirs_xy[:, 0], weights=weights)
    y_mean = np.average(dirs_xy[:, 1], weights=weights)

    # Re-convert into angle in degrees.
    rho, phi = cart2pol(x_mean, y_mean)
    phi_deg = deg_mod(phi*rad)

    # Coarse to one of the original directions.
    phi_deg_c = coarse_dir(phi_deg, cdirs)

    return rho, phi_deg, phi_deg_c


def select_period_around_max(ts, twidth, t_start=None, t_stop=None):
    """Returns values in timeseries v within given period around maximum."""

    if t_start is None:
        t_start = ts.index[0]

    if t_stop is None:
        t_stop = ts.index[-1]

    tmax = ts.argmax()
    t1, t2 = tmax - twidth, tmax + twidth

    return t1, t2


# %% General statistics and analysis functions.

def SNR(v):
    """Returns signal to noise ratio of values."""

    snr = np.mean(v) / np.std(v)
    return snr


def sem(v, axis=0):
    """Return standard error of the mean."""

    # Keeping dimension of Quantity arrays (as opposed to scipy.stats.sem).
    std = np.nanstd(v, axis)      # standard deviation
    n = (~np.isnan(v)).sum(axis)  # number of (non-None) samples
    sem = std / np.sqrt(n)

    return sem


def mean_sem(v, axis=0):
    """Return mean and SEM of array."""

    v_mean = np.nanmean(v, axis=axis)
    v_sem = sem(v, axis=axis)

    return v_mean, v_sem


def modulation_index(v1, v2):
    """Calculate modulation index between pair(s) of values."""

    if v1 == 0 and v2 == 0:
        mi = 0
    elif v1 == -v2:
        warnings.warn('+ve / -ve opposites encountered, returning 0.')
        mi = 0
    else:
        mi = (v1 - v2) / (v1 + v2)

    return mi


def fano_factor(v):
    """Calculate Fano factor of vector of spike counts."""

    varv, meanv = np.var(v),  np.mean(v)

    if meanv == 0:
        return np.nan

    fanofac = varv / meanv
    return fanofac


def pearson_r(v1, v2):
    """Calculate Pearson's r and p value."""

    r, p = sp.stats.pearsonr(v1, v2)
    return r, p


def lin_regress(x, y):
    """Returns linear regression results."""

    return sp.stats.linregress(x, y)


# TODO: add non-parametric AND non-paired test!
def t_test(x, y, paired=False, equal_var=False, nan_policy='propagate'):
    """
    Run t-test between two related (paired) or independent (unpaired) samples.
    """

    if paired:
        stat, pval = sp.stats.ttest_rel(x, y, nan_policy=nan_policy)
    else:
        stat, pval = sp.stats.ttest_ind(x, y, equal_var=equal_var)

    return stat, pval


def wilcoxon_test(x, y, zero_method='wilcox', correction=False):
    """
    Run Wilcoxon test, testing the null-hypothesis that
    two related paired samples come from the same distribution. In particular,
    it tests whether the distribution of the differences x-y is symmetric
    about zero. It is a non-parametric version of the paired T-test.

    Note: Because the normal approximation is used for the calculation,
          the samples used should be large. A typical rule is to require
          that n > 20.
    """

    stat, pval = sp.stats.wilcoxon(x, y, zero_method=zero_method,
                                   correction=correction)
    return stat, pval


def sign_diff(ts1, ts2, p, test, test_kwargs):
    """
    Return times of significant difference between two sets of time series.
    """

    # To make sure we can use standard Numpy indexing.
    ts1 = np.array(ts1)
    ts2 = np.array(ts2)

    lr1 = ts1.shape[1]
    lr2 = ts2.shape[1]

    if lr1 != lr2:
        warnings.warn('Unequal lengths ({} and {}).'.format(lr1, lr2))

    # Select test.
    if test == 't-test':
        test_func = t_test
    elif test == 'wilcoxon':
        test_func = wilcoxon_test
    else:
        print('Unrecognised test name: ' + str(test) + ', running t-test.')
        test_func = t_test
        return None, None

    # Calculate p-values and times of significant difference.
    pvals = np.array([test_func(ts1[:, i], ts2[:, i], **test_kwargs)[1]
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


def sign_periods(ts1, ts2, time, p, test, test_kwargs):
    """
    Return list of periods of significantly difference
    between sets of time series.
    """

    # Indices of significant difference.
    tsign = sign_diff(ts1, ts2, p, test, test_kwargs)[1]
    # Periods of significant difference.
    sign_periods = periods(tsign, time)

    return sign_periods


# %% Functions to create kernels for firing rate estimation.

def rect_width_from_sigma(sigma):
    """Return rectangular kernel width from sigma."""

    width = 2 * np.sqrt(3) * sigma.rescale(ms)
    return width


def sigma_from_rect_width(width):
    """Return sigma from rectangular kernel width."""

    sigma = width.rescale(ms) / 2 / np.sqrt(3)
    return sigma


def rect_kernel(width):
    """Create rectangular kernel with given width."""

    sigma = sigma_from_rect_width(width)
    rk = RectangularKernel(sigma=sigma)
    return rk


def gaus_kernel(sigma):
    """Create Gaussian kernel with given sigma."""

    gk = GaussianKernel(sigma=sigma)
    return gk


def create_kernel(kerneltype, width):
    """Create kernel of given type with given width."""

    if kerneltype in ('G', 'Gaussian'):
        fkernel = gaus_kernel

    elif kerneltype in ('R', 'Rectangular'):
        fkernel = rect_kernel

    else:
        warnings.warn('Unrecognised kernel type %s'.format(kerneltype) +
                      ', returning Rectangular kernel.')
        fkernel = rect_kernel

    krnl = fkernel(width)

    return krnl


def kernel(kname):
    """
    Return kernel created with parameters encoded in name (type and width).
    """

    kerneltype = kname[0]
    width = int(kname[1:]) * ms
    kern = create_kernel(kerneltype, width)

    return kern


def kernel_set(knames):
    """Return set of kernels specified in list of knames."""

    kset = OrdDict((kname, kernel(kname)) for kname in knames)
    return kset
