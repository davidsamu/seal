# -*- coding: utf-8 -*-
"""
Utility functions for getting file names, and import / export data for
ROC analyses.

@author: David Samu
"""

from seal.util import util


# %% Utility functions to get file names, and import / export data.

def aroc_res_fname(res_dir, nrate, n_perm, offsets):
    """Return full path to AROC result with given parameters."""

    offset_str = '_'.join([str(int(d)) for d in offsets])
    fname = '{}_nperm_{}_offs_{}.data'.format(nrate, n_perm, offset_str)
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


def aroc_fig_title(between_str, monkey, task, nrec, offsets, sort_prd=None):
    """Return title to AROC results figure with given parameters."""

    prd_str = 'sorted by: ' + sort_prd if sort_prd is not None else 'unsorted'
    ostr = ', '.join([str(int(d)) for d in offsets])
    title = ('AROC between {}, {}\n'.format(between_str, prd_str) +
             'monkey: {}, task: {}'.format(monkey, task) +
             ', # recordings: {}, offsets: {} deg'.format(nrec, ostr))
    return title


def aroc_table_fname(res_dir, monkey, task, nrate, n_perm, offsets,
                     sort_prd, min_len, pth, vth_hi, vth_lo):
    """Return full path to AROC results table with given parameters."""

    ostr = '_'.join([str(int(d)) for d in offsets])
    ftable = ('{}_nperm_{}_offs_{}'.format(nrate, n_perm, ostr) +
              '_prd_{}_min_len_{}_pth_{}'.format(sort_prd, int(min_len), pth) +
              '_vth_hi_{}_vth_lo_{}'.format(vth_hi, vth_lo))
    ftable = util.join([res_dir+'tables',
                        util.format_to_fname(ftable)+'.xlsx'])
    return ftable


def load_aroc_res(res_dir, nrate, n_perm, offsets):
    """Load AROC results."""

    fres = aroc_res_fname(res_dir, nrate, n_perm, offsets)
    aroc_res = util.read_objects(fres, ['aroc', 'pval'])

    return aroc_res
