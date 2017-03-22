# -*- coding: utf-8 -*-
"""
Utility functions for getting file names, and import / export data for
ROC analyses.

@author: David Samu
"""

from seal.util import util


# %% Utility functions to get file names, and import / export data.

def aroc_res_fname(res_dir, nrate, tstep, n_perm, offsets):
    """Return full path to AROC result with given parameters."""

    offset_str = '_'.join([str(int(d)) for d in offsets])
    fname = '{}_tstep{}_nperm{}_offs{}.data'.format(nrate, tstep, n_perm,
                                                    offset_str)
    fpath = util.join([res_dir+'res', fname])

    return fpath


def aroc_fig_fname(res_dir, prefix, offsets, cmap, sort_prd=None):
    """Return full path to AROC result with given parameters."""

    sort_prd_str = ('sorted_by_' + util.format_to_fname(sort_prd)
                    if sort_prd is not None else 'unsorted')
    ostr = 'offset_' + '_'.join([str(int(d)) for d in offsets])
    fname = 'AROC_{}_{}.png'.format(prefix, sort_prd_str)
    ffig = util.join([res_dir+'heatmap', ostr, cmap, fname])

    return ffig


def aroc_fig_title(between_str, task, nunits, offsets, sort_prd=None):
    """Return title to AROC results figure with given parameters."""

    prd_str = 'sorted by: ' + sort_prd if sort_prd is not None else 'unsorted'
    ostr = ', '.join([str(int(d)) for d in offsets])
    title = ('AROC between {}, {}\n'.format(between_str, prd_str) +
             '{}, # units: {}, offsets: {} deg'.format(task, nunits, ostr))
    return title


def aroc_table_fname(res_dir, task, nrate, tstep, n_perm, offsets,
                     sort_prd, min_len, pth, vth_hi, vth_lo):
    """Return full path to AROC results table with given parameters."""

    ostr = '_'.join([str(int(d)) for d in offsets])
    ftable = ('{}_{}_tstep{}_nperm{}_offs{}'.format(task, nrate, tstep,
                                                    n_perm, ostr) +
              '_prd{}_minlen{}_pth{}'.format(sort_prd, int(min_len), pth) +
              '_vthhi{}_vthlo{}'.format(vth_hi, vth_lo))
    ftable = util.join([res_dir+'tables',
                        util.format_to_fname(ftable)+'.xlsx'])
    return ftable


def load_aroc_res(res_dir, nrate, tstep, n_perm, offsets):
    """Load AROC results."""

    fres = aroc_res_fname(res_dir, nrate, tstep, n_perm, offsets)
    aroc_res = util.read_objects(fres)

    return aroc_res
