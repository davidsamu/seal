#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for getting file names, and import / export data for
decoding analyses.

@author: David Samu
"""


import os

from seal.util import util


def res_fname(res_dir, subdir, feat, nrate, ncv, n_pshfl, sep_err_trs,
              sep_by, zscore_by, n_most_DS, tstep):
    """Return full path to decoding result with given parameters."""

    feat_str = util.format_to_fname(str(feat))
    ncv_str = 'ncv{}'.format(ncv)
    pshfl_str = 'npshfl{}'.format(n_pshfl)
    err_str = 'w{}err'.format('o' if sep_err_trs else '')
    zscore_str = ('_zscoredby_'+util.format_feat_name(zscore_by, True)
                  if zscore_by is not None else '')
    nDS_str = ('top{}u'.format(n_most_DS)
               if n_most_DS != 0 else 'allu')
    tstep_str = 'tstep{}ms'.format(int(tstep))
    dir_name = '{}{}{}'.format(res_dir, feat_str, zscore_str)
    fname = '{}_{}_{}_{}_{}_{}.data'.format(nrate, ncv_str, pshfl_str,
                                            err_str, nDS_str, tstep_str)
    fres = util.join([dir_name, subdir, fname])

    return fres


def fig_fname(res_dir, subdir, ext='png', **par_kws):
    """Return full path to decoding result with given parameters."""

    fres = res_fname(res_dir, subdir, **par_kws)
    ffig = os.path.splitext(fres)[0] + '.' + ext
    return ffig


def fig_title(res_dir, feat, nrate, ncv, n_pshfl, sep_err_trs,
              sep_by, zscore_by, n_most_DS, tstep):
    """Return title for decoding result figure with given parameters."""

    feat_str = util.format_to_fname(str(feat))
    cv_str = 'Logistic regression with {}-fold CV'.format(ncv)
    err_str = 'error trials ' + ('excl.' if sep_err_trs else 'incl.')
    pshfl_str = '# population shuffles: {}'.format(n_pshfl)
    zscore_str = ('' if zscore_by is None else
                  'z-scored by ' + util.format_feat_name(zscore_by, True))
    nDS_str = ('using {} most DS units'.format(n_most_DS)
               if n_most_DS != 0 else 'all units')
    tstep_str = 'time step: {} ms'.format(int(tstep))
    title = ('Decoding {}\n{}'.format(feat_str, cv_str) +
             '\nFR kernel: {}, {}, {}'.format(nrate, tstep_str, err_str) +
             '\n{}, {}, {}'.format(pshfl_str, nDS_str, zscore_str))
    return title


def load_res(res_dir, **par_kws):
    """Load decoding results."""

    fres = res_fname(res_dir, **par_kws)
    dec_res = util.read_objects(fres, ['Scores', 'Coefs', 'C'])

    return dec_res
