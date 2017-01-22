# -*- coding: utf-8 -*-
"""
Functions related to exporting data.

@author: David Samu
"""


import numpy as np
import scipy as sp
import pandas as pd

from seal.util import util


def export_unit_list(UA, fname):
    """Export unit list and parameters into Excel table."""

    unit_params = UA.unit_params()
    writer = pd.ExcelWriter(fname)
    util.write_table(unit_params, writer)


def export_unit_trial_selection(UA, fname):
    """Export unit and trial selection as Excel table."""

    # Gather selection dataframe.
    columns = ['recording', 'channel', 'unit index', 'task', 'unit included',
               'first included trial', 'last included trial']
    SelectDF = pd.DataFrame(columns=columns)

    for i, u in enumerate(UA.iter_thru(excl=True)):
        rec, ch, un, task = u.get_utid()
        inc = int(not u.is_excluded())
        inc_trs = u.inc_trials()
        ftr, ltr = 0, 0
        if len(inc_trs):
            ftr, ltr = inc_trs.min()+1, inc_trs.max()+1
        SelectDF.loc[i] = [rec, ch, un, task, inc, ftr, ltr]

    # Sort table to facilitate reading by recording.
    SelectDF.sort_values(['recording', 'channel', 'unit index', 'task'],
                         inplace=True)
    SelectDF.index = range(1, len(SelectDF.index)+1)

    # Write out selection dataframe.
    writer = pd.ExcelWriter(fname)
    util.write_table(SelectDF, writer)


def export_decoding_data(UA, fname, rec, task, trs, prd, nrate=None):
    """Export decoding data into .mat file."""

    # Init.
    uids = UA.get_uids_of_rec(rec)

    # Below inits rely on these params being the same across units, which is
    # only appropriate when exporting a single task of a single recording!
    u = UA.get_unit(task, uids[0])
    t1s, t2s = u.pr_times(prd, add_latency=False, concat=False)
    if nrate is None:
        nrate = u.init_nrate()

    # Trial params.
    trparams = u.TrialParams
    trpars = np.array([util.remove_dim_from_series(trparams[par])
                       for par in trparams]).T
    trpar_names = trparams.columns.tolist()

    # Trial events.
    trevents = u.Events
    trevns = np.array([util.remove_dim_from_series(trevents[evn])
                       for evn in trevents]).T
    trevn_names = trevents.columns.tolist()

    # Rates.
    rates = np.array([u._Rates[nrate].get_rates(trs, t1s, t2s)
                      for u in UA.iter_thru([task], uids)])

    # Sampling times.
    times = np.array(u._Rates[nrate].get_rates(trs, t1s, t2s).columns)

    # Create dictionary to export.
    export_dict = dict(recording=rec, task=task,
                       period=prd, nrate=nrate,
                       trial_parameter_names=trpar_names,
                       trial_parameters=trpars,
                       trial_event_names=trevn_names,
                       trial_events=trevns,
                       times=times, rates=rates)

    # Export data.
    sp.io.savemat(fname, export_dict)
