#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for getting file names, and import / export data for
decoding analyses.

@author: David Samu
"""


import os

from seal.util import util


def res_fname(res_dir, subdir, tasks, feat, nrate, ncv, Cs, n_pshfl,
              sep_err_trs, sep_by, zscore_by, even_by, PDD_offset, n_most_DS,
              tstep):
    """Return full path to decoding result with given parameters."""

    tasks_str = '_'.join(tasks)
    feat_str = util.format_to_fname(str(feat))
    ncv_str = 'ncv{}'.format(ncv)
    Cs_str = 'nregul{}'.format(len(Cs)) if Cs != [1] else 'reguloff'
    pshfl_str = 'npshfl{}'.format(n_pshfl)
    err_str = 'w{}err'.format('o' if sep_err_trs else '')
    zscore_str = ('_zscoredby'+util.format_feat_name(zscore_by, True)
                  if not util.is_null(zscore_by) else '')
    even_str = ('_evenby'+util.format_feat_name(even_by, True)
                if not util.is_null(even_by) else '')
    PDD_str = ('_PDDoff{}'.format(int(PDD_offset))
               if not util.is_null(PDD_offset) else '')
    nDS_str = ('top{}u'.format(n_most_DS)
               if n_most_DS != 0 else 'allu')
    tstep_str = 'tstep{}ms'.format(int(tstep))
    dir_name = '{}{}/{}{}{}{}'.format(res_dir, tasks_str, feat_str,
                                      zscore_str, even_str, PDD_str)
    fname = '{}_{}_{}_{}_{}_{}_{}.data'.format(nrate, ncv_str, Cs_str,
                                               pshfl_str, err_str, nDS_str,
                                               tstep_str)
    fres = util.join([dir_name, subdir, fname])

    return fres


def fig_fname(res_dir, subdir, ext='pdf', **par_kws):
    """Return full path to decoding result with given parameters."""

    fres = res_fname(res_dir, subdir, **par_kws)
    ffig = os.path.splitext(fres)[0] + '.' + ext
    return ffig


def fig_title(res_dir, tasks, feat, nrate, ncv, Cs, n_pshfl, sep_err_trs,
              sep_by, zscore_by, even_by, PDD_offset, n_most_DS, tstep):
    """Return title for decoding result figure with given parameters."""

    feat_str = util.format_to_fname(str(feat))
    tasks_str = ', '.join(tasks)
    cv_str = 'Logistic regression with {}-fold CV'.format(ncv)
    Cs_str = 'regularization: ' + (str(Cs) if Cs != [1] else 'off')
    err_str = 'error trials ' + ('excl.' if sep_err_trs else 'incl.')
    pshfl_str = '# population shuffles: {}'.format(n_pshfl)
    zscore_str = ('raw non-z-scored rates' if util.is_null(zscore_by) else
                  'z-scored by ' + util.format_feat_name(zscore_by, True))
    even_str = ('uneven trial numbers' if util.is_null(even_by) else
                'trial numbers evened across ' +
                util.format_feat_name(even_by, True))
    PDD_str = ('using all 8 directions' if util.is_null(PDD_offset) else
               'using only PDD/PAD +- {}'.format(PDD_offset))
    nDS_str = ('using {} most DS units'.format(n_most_DS)
               if n_most_DS != 0 else 'all units')
    tstep_str = 'time step: {} ms'.format(int(tstep))
    title = ('Decoding {} in {}'.format(feat_str, tasks_str) +
             '\n{}, {}'.format(cv_str, Cs_str) +
             '\nFR kernel: {}, {}, {}'.format(nrate, tstep_str, err_str) +
             '\n{}, {}'.format(pshfl_str, nDS_str) +
             '\n{}, {}, {}'.format(zscore_str, even_str, PDD_str))
    return title


def load_res(res_dir, list_n_most_DS=None, **par_kws):
    """Load decoding results."""

    if list_n_most_DS is None:
        list_n_most_DS = [par_kws['n_most_DS']]

    # Load results for each number of units included.
    all_rt_res = {}
    for n_most_DS in list_n_most_DS:
        par_kws['n_most_DS'] = n_most_DS
        fres = res_fname(res_dir, 'results', **par_kws)
        rt_res = util.read_objects(fres, 'rt_res')
        all_rt_res[n_most_DS] = rt_res

    if len(list_n_most_DS) == 1:
        all_rt_res = list(all_rt_res.values())[0]

    return all_rt_res
