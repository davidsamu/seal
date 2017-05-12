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
import multiprocessing as mp
from collections import Iterable

import numpy as np
import scipy as sp
import pandas as pd
from quantities import Quantity

from seal.util import constants


# %% Input / output and object manipulation functions.

def read_matlab_object(fname, obj_names=None):
    """Return Matlab structure or object from it."""

    mat_struct = sp.io.loadmat(fname, struct_as_record=False, squeeze_me=True)

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


def write_matlab_object(fname, obj_dict):
    """Write out dictionary as Matlab object."""

    create_dir(fname)
    sp.io.savemat(fname, obj_dict)


def read_objects(fname, obj_names=None):
    """Read in objects from pickled data file."""

    data = pickle.load(open(fname, 'rb'))

    # Unload objects from dictionary.
    if obj_names is None:
        objects = data  # all objects
    elif isinstance(obj_names, str):
        objects = data[obj_names]   # single object
    else:
        objects = [data[oname] for oname in obj_names]  # multiple objects

    return objects


def write_objects(obj_dict, fname):
    """Write out dictionary object into pickled data file."""

    create_dir(fname)
    pickle.dump(obj_dict, open(fname, 'wb'))


def write_table(df, writer, **kwargs):
    """Write out Pandas dataframe as Excel table."""

    if writer is None:
        return

    create_dir(writer.path)
    df.to_excel(writer, **kwargs)
    writer.save()


def save_sheets(list_dfs, sheet_names, fname):
    """Save each DataFrame from list into separate sheet of Excel document."""

    # Init writer.
    writer = pd.ExcelWriter(fname)
    create_dir(writer.path)

    # Save sheets.
    for i, df in enumerate(list_dfs):
        df.to_excel(writer, sheet_names[i])

    writer.save()


def write_excel(df, sheet_name, fname):
    """Export DataFrame into Excel document."""

    save_sheets([df], [sheet_name], fname)


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

def params_from_fname(fname, nchar_date=6):
    """Extract recording parameters from file name."""

    # Remove extension and split into parts by '_' underscore character.
    froot = os.path.splitext(fname)[0]
    [subj, date_elec, taskidx, sortno] = froot.split('_')
    date, elec = [date_elec[:nchar_date], date_elec[nchar_date:].upper()]
    task, idx = taskidx[:-1], int(taskidx[-1])

    # Return in Series.
    index = ['subj', 'date', 'elec', 'taskidx', 'task', 'idx', 'sortno']
    name_fields = pd.Series([subj, date, elec, taskidx, task, idx, sortno],
                            index=index)

    return name_fields


def format_to_fname(s):
    """Format string to file name compatible string."""

    valid_chars = "_ %s%s" % (string.ascii_letters, string.digits)
    fname = ''.join(c for c in s if c in valid_chars)
    fname = fname.replace(' ', '_')
    return fname


def format_feat_name(feat, to_fname=False):
    """Format feature name to string."""

    feat_str = ', '.join(feat) if is_iterable(feat) else feat
    if to_fname:
        feat_str = format_to_fname(feat_str)
    return feat_str


def join(str_list):
    """Join list of strings as file path."""

    path = os.path.join(*str_list)
    return path


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


def star_pvalue(pval, n_max_stars=4, equal=''):
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
        pstr = equal

    return pstr


def format_offsets(offsets):
    """Return list of degree offset values string formatted."""

    offsets_str = ', '.join([str(int(off))+' deg' for off in offsets])

    return offsets_str


def format_uid(uid):
    """Format unit ID (recording, channel, unit idx) triple as string."""

    subj, date, elec, ch, idx = uid
    uid_str = '{}_{}{}_Ch{}_{}'.format(subj, date, elec, ch, idx)

    return uid_str


def is_number(s):
    """Test if string can be converted into numeric value."""

    try:
        float(s)
        return True
    except ValueError:
        return False


# %% System-related functions.

def get_n_cores():
    """Get number of cores."""

    ncores = mp.cpu_count()
    return ncores


def run_in_pool(f, params, nCPU=None):
    """Run a function parallel with a list of parameters on local processor."""

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


def is_null(obj):
    """Check if object is null (i.e. None or np.nan)."""

    isnull = (obj is None) or (isinstance(obj, float) and np.isnan(obj))
    return isnull


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


def series_from_tuple_list(tuple_list, mi_names=None):
    """Create Pandas series from list of (name, value) tuples."""

    if not len(tuple_list):
        return pd.Series()

    # Create Series from list of tuples.
    sp_names, sp_vals = zip(*tuple_list)
    series = pd.Series(sp_vals, index=sp_names)

    # Format index to MultiIndex.
    if mi_names is not None:
        series.index = pd.MultiIndex.from_tuples(series.index,
                                                 names=mi_names)

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

    if not len(qvec):
        return qvec
    if not isinstance(qvec[0], Quantity):
        return qvec

    np_vec = np.array([float(qv) for qv in qvec], dtype=dtype)

    return np_vec


def remove_dim_from_series(qser):
    """Remove physical dimension from Pandas Series."""

    arr = remove_dim_from_array(np.array(qser))
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


# %% Functions to handle Pandas objects.

def get_mi_level_combs(data, level_names):
    """Return combination of level values from MultiIndex-ed data."""

    lvl_vals = data.reset_index().set_index(level_names).index

    return lvl_vals


def get_subj_date_pairs(data):
    """
    Return unique (subject, date) pairs from MultiIndex index of DF or Series.
    """

    sd_pairs = get_mi_level_combs(data, ['subj', 'date']).unique()

    return sd_pairs


def melt_table(df, colnames, add_cols=[], reindex=True, unstack=True):
    """Melt DataFrame, typically before for plotting it with Seaborn."""

    # Melt DataFrame.
    ldf = df.unstack() if unstack else df.stack()
    ldf = pd.DataFrame(ldf, columns=colnames)

    # Add levels of MultiIndex as columns.
    for name, lvl in add_cols:
        if is_iterable(lvl):
            col = pd.Index(get_mi_level_combs(ldf, lvl))
        else:
            col = ldf.index.get_level_values(lvl)

        ldf[name] = col

    # Replace original MultiIndex with integer indexing.
    if reindex:
        ldf.index = np.arange(len(ldf.index))

    return ldf


# %% Functions to init and handle analysis of multiple stimulus periods.

def add_stim_to_feat(feat, stims):
    """Add stimulus to feature name for stimulus period info table."""

    stim_feat = [(stim, feat) if feat in constants.stim_feats else feat
                 for stim in stims]
    return stim_feat


def init_stim_prds(stims, feat, sep_by=None, zscore_by=None, even_by=None,
                   PDD_offset=None, tr_prds=None, prds=None, ref_ev=None):
    """Initialize stimulus periods to be analyzed."""

    if tr_prds is None:
        tr_prds = constants.fixed_tr_prds

    pars = pd.DataFrame(index=stims)

    # Stimulus feature parameters.
    pars['feat'] = add_stim_to_feat(feat, stims)
    pars['sep_by'] = add_stim_to_feat(sep_by, stims)
    pars['zscore_by'] = add_stim_to_feat(zscore_by, stims)
    pars['even_by'] = add_stim_to_feat(even_by, stims)
    pars['PDD_offset'] = [PDD_offset for stim in stims]

    # Time period parameters.
    pars['prd'] = [stim + ' half' for stim in stims] if prds is None else prds
    pars['ref_ev'] = ([stim + ' on' for stim in stims]
                      if ref_ev is None else ref_ev)

    pars['stim_start'] = [tr_prds.loc[stim].start for stim in stims]
    pars['stim_stop'] = [tr_prds.loc[stim].stop for stim in stims]

    pars['prd_start'] = [tr_prds.loc[prd].start for prd in pars['prd']]
    pars['prd_stop'] = [tr_prds.loc[prd].stop for prd in pars['prd']]

    return pars


def concat_stim_prd_res(res_list, tshifts=None, truncate_prds=None,
                        remove_all_nan_units=True, remove_any_nan_times=True):
    """Concatenate stimulus period results."""

    # Make a copy of input data.
    res_list = [res.copy() for res in res_list if res is not None]

    # Is there any data to concatenate?
    if not len(res_list):
        return

    # Convert Series to DataFrame.
    if isinstance(res_list[0], pd.Series):
        res_list = [pd.DataFrame(res).T for res in res_list]

    # Offset time points (columns).
    if tshifts is not None and len(tshifts):
        for i, tshift in enumerate(tshifts):
            res_list[i].columns = res_list[i].columns + tshift

    # Truncate to provided time period.
    if truncate_prds is not None and len(truncate_prds):
        for i, (tstart, tstop) in enumerate(truncate_prds):
            cols = res_list[i].columns
            prdcols = cols[(cols >= tstart) & (cols <= tstop)]
            res_list[i] = res_list[i][prdcols]

    # Concatenate stimulus-period specific results.
    res = pd.concat(res_list, axis=1)
    res.columns = res.columns.astype(int)

    # Remove units (rows) with all NaN values
    # (e.g. because of not enough trials to do unit-specific analysis).
    if remove_all_nan_units:
        to_keep = ~res.isnull().all(1)
        res = res.loc[to_keep, :]

    # Remove time points (columns) with any NaN value (S2 happened earlier).
    if remove_any_nan_times:
        to_keep = ~res.isnull().any(0)
        res = res.loc[:, to_keep]

    # Remove duplicated time points (overlaps of periods).
    res = res.loc[:, ~res.columns.duplicated()]

    # Convert single row DataFrame to Series.
    # if len(res.index) == 1:
    #    res = res.T.squeeze()

    return res


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


def normalize(v, nmin=0, nmax=1):
    """Normalize array into specified range."""

    # Empty array.
    if not v.size:
        return v
    if nmin >= nmax:
        warnings.warn('min >= max')
        return v

    vmin, vmax = v.min(), v.max()

    # All values are the same.
    if vmin == vmax:
        return np.ones(v.shape)

    # Normalize into [0, 1]
    vnorm = (v-vmin) / (vmax-vmin)

    # Scale and shift into specified range.
    vnorm = (nmax - nmin) * vnorm + nmin

    return vnorm


def pearson_r(v1, v2):
    """Calculate Pearson's r and p value."""

    r, p = sp.stats.pearsonr(v1, v2)
    return r, p


def lin_regress(x, y):
    """Returns linear regression results."""

    return sp.stats.linregress(x, y)


def calc_stim_resp_stats(stim_resp, all_stim_vals=None):
    """Calculate stimulus response statistics from raw response values."""

    # Init all stimulus values.
    if all_stim_vals is None:
        all_stim_vals = np.sort(stim_resp.vals.unique())

    # Init response matrix.
    resp_stats = pd.DataFrame(columns=['mean', 'std', 'sem'],
                              index=all_stim_vals, dtype=float)

    for v in all_stim_vals:
        v = float(v)
        resp = stim_resp.resp[stim_resp.vals == v]
        # Calculate statistics.
        if resp.size:
            mean_resp = resp.mean()
            std_resp = resp.std()
            sem_resp = std_resp / np.sqrt(resp.size)
            resp_stats.loc[v] = (mean_resp, std_resp, sem_resp)

    return resp_stats


def long_periods(vec, min_len):
    """Return indices of periods with length at least as specified minimum."""

    if vec.dtype != 'bool':
        warnings.warn('vec is not boolean Series.')

    # Entire period is 'off'.
    if not vec.any():
        return pd.Series(name='index', dtype=object)

    # Chop vector into blocks of consequtive time points with same value.
    blocks = (vec.shift(1) != vec).astype(int).cumsum()
    df = pd.DataFrame.from_items([('vec', vec), ('blocks', blocks)])
    prds = df.reset_index().groupby(['vec', 'blocks'])['index'].apply(np.array)

    # Select only 'on' periods.
    prds = prds[True]

    # Keep only ones having at least minimum length.
    idxs = [(prd[-1] - prd[0]) > min_len for idx, prd in prds.iteritems()]
    prds = prds[idxs]

    return prds
