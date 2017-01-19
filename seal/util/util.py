# -*- coding: utf-8 -*-
"""
Collection of utility functions.

@author: David Samu
"""

import os
import copy
import warnings
import pickle
import datetime
import functools

import string
import numpy as np
import scipy as sp
import pandas as pd
import multiprocessing as mp
from collections import Iterable

from quantities import Quantity, s


# Constants.
min_sample_size = 10


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
    elif pval < 0.10:
        pstr = 'p = {:.2f}'.format(pval)
    else:
        pstr = 'p > 0.10'

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


def format_uid(uid):
    """Format unit ID (recording, channel, unit idx) triple as string."""

    rec, ch, idx = uid
    uid_str = '{}_Ch{}_{}'.format(rec, ch, idx)

    return uid_str


# %% System-related functions.

def get_n_cores():
    """Get number of cores."""

    ncores = mp.cpu_count()
    return ncores


def run_in_pool(f, params, nCPU=None):
    """Run a function in parallel pool."""

    if nCPU is None:  # set number of cores
        nCPU = get_n_cores() - 1

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


def find_nearest(arr, v):
    """Return nearest value in array to v."""

    nearest = arr[(np.abs(arr-v)).argmin()]
    return nearest


def series_from_tuple_list(tuple_list):
    """Create Pandas series from list of (name, value) tuples."""

    if not len(tuple_list):
        return pd.Series()

    sp_names, sp_vals = zip(*tuple_list)
    series = pd.Series(sp_vals, index=sp_names)

    return series


def get_scalar_vals(series, remove_dimensions=False):
    """
    Function to extract all non-iterator, non-class type value from Series.
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


# %% Functions to create different types of combinations of lists (trials).

def union_lists(val_lists, name='or'):
    """Union (OR) of list of lists of values into single list."""

    agg_list = functools.reduce(np.union1d, val_lists)
    agg_ser = pd.Series([agg_list], index=[name])

    return agg_ser


def intersect_lists(val_lists, name='and'):
    """Intersection (AND) of list of lists of values into single list."""

    com_list = functools.reduce(np.intersect1d, val_lists)
    com_ser = pd.Series([com_list], index=[name])

    return com_ser


def diff_lists(val_lists, name='diff'):
    """Difference of list of lists of values into single list."""

    diff_list = functools.reduce(np.setdiff1d, val_lists)
    diff_ser = pd.Series([diff_list], index=[name])

    return diff_ser


def combine_lists(val_lists, comb_type, **kwargs):
    """Combine list of lists by pass type ('and', 'or', or 'diff')."""

    comb_ser = None
    if comb_type == 'and':
        comb_ser = intersect_lists(val_lists, **kwargs)
    elif comb_type == 'or':
        comb_ser = union_lists(val_lists, **kwargs)
    elif comb_type == 'diff':
        comb_ser = diff_lists(val_lists, **kwargs)

    return comb_ser


def filter_lists(val_list_ser, filter_list):
    """Filter Series of lists by an other list."""

    ftrd_ser = val_list_ser.apply(np.intersect1d, args=(filter_list,))
    ftrd_ser = ftrd_ser[[len(ftrs) > 0 for ftrs in ftrd_ser]]

    return ftrd_ser


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


def rescale_series(ser, dim):
    """Rescale dimension of Pandas Series."""

    ser2 = ser.copy()
    for i in ser2.index:
        ser2[i] = ser2[i].rescale(dim)
    return ser2


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


# %% General statistics and analysis functions.

def zscore_timeseries(timeseries, axis=0):
    """
    Z-score set of time series at each time point (per column).

    axis: Axis along which to operate. If None, compute over the whole array.
    """

    zscored_ts = sp.stats.zscore(timeseries, axis=axis)
    zscored_ts[np.isnan(zscored_ts)] = 0  # remove NaN values

    return zscored_ts


def select_period_around_max(ts, twidth, t_start=None, t_stop=None):
    """Returns values in timeseries within given period around maximum."""

    if t_start is None:
        t_start = ts.index[0]

    if t_stop is None:
        t_stop = ts.index[-1]

    tmax = ts.argmax()
    t1, t2 = tmax - twidth, tmax + twidth

    return t1, t2


def SNR(v):
    """Returns signal to noise ratio of values."""

    snr = np.mean(v) / np.std(v)
    return snr


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


def t_test(x, y, paired=False, equal_var=False, nan_policy='propagate'):
    """
    Run t-test between two related (paired) or independent (unpaired) samples.
    """

    # Remove any NaN values.
    if paired:
        idx = np.logical_and(~np.isnan(x), ~np.isnan(y))
        x, y = x[idx], y[idx]

    # Insufficient sample size.
    xvalid, yvalid = [v[~np.isnan(v)] for v in (x, y)]
    if min(len(xvalid), len(yvalid)) < min_sample_size:
        return np.nan, np.nan

    if paired:
        stat, pval = sp.stats.ttest_rel(x, y, nan_policy=nan_policy)
    else:
        stat, pval = sp.stats.ttest_ind(xvalid, yvalid, equal_var=equal_var)

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

    # Remove any NaN values. Test is always paired!
    idx = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x, y = x[idx], y[idx]

    # Insufficient sample size.
    if min(len(x), len(y)) < min_sample_size:
        return np.nan, np.nan

    stat, pval = sp.stats.wilcoxon(x, y, zero_method=zero_method,
                                   correction=correction)
    return stat, pval


def mann_whithney_u_test(x, y, use_continuity=True, alternative='two-sided'):
    """Run Mann-Whitney (aka unpaired Wilcoxon) rank test on samples."""

    # Insufficient sample size.
    xvalid, yvalid = [v[~np.isnan(v)] for v in (x, y)]
    if min(len(xvalid), len(yvalid)) < min_sample_size:
        return np.nan, np.nan

    stat, pval = sp.stats.mannwhitneyu(x, y, use_continuity, alternative)

    return stat, pval


def sign_diff(ts1, ts2, p, test, **kwargs):
    """
    Return times of significant difference between two sets of time series.

    ts1, ts2: time series stored in DataFrames, columns are time samples.
    """

    # Get intersection of time vector.
    tvec = np.intersect1d(ts1.columns, ts2.columns)

    # Select test.
    if test == 't-test':
        test_func = t_test
    elif test == 'wilcoxon':
        test_func = wilcoxon_test
    elif test == 'mann_whitney_u':
        test_func = mann_whithney_u_test
    else:
        print('Unrecognised test name: ' + str(test) + ', running t-test.')
        test_func = t_test

    # Calculate p-values and times of significant difference.
    pvals = pd.Series([test_func(ts1[t], ts2[t], **kwargs)[1]
                       for t in tvec], index=tvec)
    tsign = pvals < p

    return pvals, tsign


def periods(t_on_ser, min_len=None):
    """Return list of periods where t_on is True and with minimum length."""

    if not len(t_on_ser.index):
        return []

    # Init data.
    tvec = np.array(t_on_ser.index)
    t_on = np.array(t_on_ser)

    # Starts of periods.
    tstarts = np.insert(t_on, 0, False)
    istarts = np.logical_and(tstarts[:-1] == False, tstarts[1:] == True)

    # Ends of periods.
    tends = np.insert(t_on, -1, False)
    iends = np.logical_and(tends[:-1] == True, tends[1:] == False)

    # Zip (start, end) pairs of periods.
    pers = [(t1, t2) for t1, t2 in zip(tvec[istarts], tvec[iends])]

    # Drop periods shorter than minimum length.
    if min_len is not None:
        pers = [(t1, t2) for t1, t2 in pers if t2-t1 >= min_len]

    return pers


def sign_periods(ts1, ts2, pval, test, min_len=None, **kwargs):
    """
    Return list of periods of significantly difference
    between two sets of time series (row: samples, columns: time points).
    """

    # Indices of significant difference.
    tsign = sign_diff(ts1, ts2, pval, test, **kwargs)[1]

    # Periods of significant difference.
    sign_periods = periods(tsign, min_len)

    return sign_periods


def calc_stim_resp_stats(stim_resp):
    """Calculate stimulus response statistics from raw response values."""

    null_resp = np.nan * 1/s
    resp_stats = pd.DataFrame(columns=['mean', 'std', 'sem'])
    for v, v_grp in stim_resp.groupby(['vals']):
        resp = np.array(v_grp['resp'])
        # Calculate statistics.
        if resp.size:
            mean_resp = np.mean(resp)
            std_resp = np.std(resp)
            sem_resp = std_resp / np.sqrt(resp.size)
        else:  # in case of no response
            mean_resp = null_resp
            std_resp = null_resp
            sem_resp = null_resp

        resp_stats.loc[v] = (mean_resp, std_resp, sem_resp)

    return resp_stats
