# -*- coding: utf-8 -*-
"""
Functions related to exporting data.

@author: David Samu
"""

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
    columns = ['task', 'session', 'channel', 'unit index', 'unit included',
               'first included trial', 'last included trial']
    SelectDF = pd.DataFrame(columns=columns)

    for i, u in enumerate(UA.iter_thru(excl=True)):
        rec, ch, un, task = u.get_utid()
        inc = int(not u.is_excluded())
        inc_trs = u.inc_trials()
        ftr, ltr = 0, 0
        if len(inc_trs):
            ftr, ltr = inc_trs.min()+1, inc_trs.max()+1
        SelectDF.loc[i] = [task, rec, ch, un, inc, ftr, ltr]

    # Write out selection dataframe.
    writer = pd.ExcelWriter(fname)
    util.write_table(SelectDF, writer)


def export_decoding_data(UA, fname):
    """Export decoding data into .mat file."""

    # TODO
    pass
