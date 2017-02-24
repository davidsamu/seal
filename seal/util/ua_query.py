# -*- coding: utf-8 -*-
"""
Utility functions to query information of units in UnitArray objects.

@author: David Samu
"""

import warnings

import numpy as np
import pandas as pd

from seal.util import kernels


# %% Query methods.

def get_DSInfo_table(UA, utids=None):
    """Return data frame with direction selectivity information."""

    # Init.
    if utids is None:
        utids = UA.utids()

    # Test direction selectivity.
    test_DS(UA)

    DSInfo = []
    for utid in utids:
        u = UA.get_unit(utid[:3], utid[3])

        # Get DS info.
        PD = u.DS.PD.cPD[('S1', 'max')]
        DSI = u.DS.DSI.mDS.S1

        DSInfo.append((utid, (PD, DSI)))

    DSInfo = pd.DataFrame.from_items(DSInfo, columns=['PD', 'DSI'],
                                     orient='index')

    return DSInfo


def get_a_unit(UA, rec, task):
    """Query units for recording and task in UA."""

    # Check if there's any unit in task of recording.
    utids = UA.utids(tasks=[task], recs=[rec])
    if not len(utids):
        warnings.warn('No unit found for given task and recording.')
        return

    # Extract first unit.
    utid = utids[0]
    u = UA.get_unit(utid[:3], utid[3])

    return u


def get_trial_params(UA, rec, task):
    """Return trial param table for a given recording-task pair."""

    u = get_a_unit(UA, rec, task)
    TrParams = u.TrData

    return TrParams


def get_prd_times(UA, rec, task, prd, ref_ev, trs=None):
    """Return timing of period across given trials."""

    if trs is None:
        TrParams = get_trial_params(UA, rec, task)
        trs = np.arange(len(TrParams.index))

    u = get_a_unit(UA, rec, task)
    t1s, t2s = u.pr_times(prd, trs, add_latency=False, concat=False)
    ref_ts = u.ev_times(ref_ev)

    return t1s, t2s, ref_ts


def get_rate_matrix(UA, rec, task, uids, prd, ref_ev, nrate, trs=None):
    """Return rate matrix across ."""

    t1s, t2s, ref_ts = get_prd_times(UA, rec, task, prd, ref_ev, trs)
    rates = {u.Name: u._Rates[nrate].get_rates(trs, t1s, t2s, ref_ts)
             for u in UA.iter_thru([task], uids)}
    rates = pd.concat(rates)

    return rates


def get_spike_times(UA, rec, task, uids, prd, ref_ev, trs=None):
    """Return spike times of units during given recording, task and trials."""

    # Get timings of interval requested.
    t1s, t2s, ref_ts = get_prd_times(UA, rec, task, prd, ref_ev, trs)

    # Query spike events across units and trails.
    spike_dict = {tuple(u.get_utid()): u._Spikes.get_spikes(trs, t1s, t2s,
                                                            ref_ts)
                  for u in UA.iter_thru([task], uids)}
    Spikes = pd.concat(spike_dict, axis=1).T

    return Spikes


# %% Manipulation methods.

def add_rate(UA, name):
    """Add rate to units in UnitArray."""

    kernel, step = kernels.kernel_set([(name, kernels.kstep)]).loc[name]
    [u.add_rate(name, kernel, step) for u in UA.iter_thru()
     if name not in u._Rates]


def test_DS(UA):
    """Test DS if it has not been tested yet."""

    [u.test_DS() for u in UA.iter_thru() if not len(u.DS)]
