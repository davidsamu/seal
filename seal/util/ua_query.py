# -*- coding: utf-8 -*-
"""
Utility functions to query information of units in UnitArray objects.

@author: David Samu
"""

import warnings

import numpy as np
import pandas as pd

from seal.util import kernels, util
from seal.object import unitarray


# %% Init methods.

def create_UA_from_recs(frecs, ua_name='UA'):
    """Create combined UnitArray from multiple recordings."""

    UA = unitarray.UnitArray(ua_name)
    for frec in frecs:
        rUA = util.read_objects(frec, 'UnitArr')
        UA.add_recording(rUA)

    return UA


# %% Query methods.

def get_DSInfo_table(UA, utids=None, stim='S1'):
    """Return data frame with direction selectivity information."""

    # Init.
    if utids is None:
        utids = UA.utids()

    # Test direction selectivity.
    test_DS(UA)

    DSInfo = []
    for u in UA.iter_thru():

        utid = tuple(u.get_utid())
        if utid not in utids:
            continue

        # Get DS info.
        PD = u.DS.PD.cPD[(stim, 'max')]
        DSI = u.DS.DSI.mDS[stim]

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
    """Return rate matrix across units and periods."""

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
    Spikes.index.set_names(unitarray.utid_names, inplace=True)

    return Spikes


def get_prd_mean_rates(UA, tasks, prd, nrate, max_len=None):
    """Return mean rates per unit during period in DataFrame."""

    mrates = {}
    for u in UA.iter_thru(tasks):
        t1s, t2s = u.pr_times(prd, concat=False)
        trs = u.inc_trials()
        mrates[u.Name] = u._Rates[nrate].get_rates(trs, t1s, t2s, t1s).mean()
    rates = pd.concat(mrates, axis=1).T

    if max_len is not None:
        rates = rates.loc[:, rates.columns <= max_len]

    return rates


# %% Manipulation methods.

def add_rate(UA, name):
    """Add rate to units in UnitArray."""

    kernel, step = kernels.kernel_set([(name, kernels.kstep)]).loc[name]
    [u.add_rate(name, kernel, step) for u in UA.iter_thru()
     if name not in u._Rates]


def test_DS_frecs(frecs):
    """Test DS of list of recordings."""

    print('Testing DS...')
    for frec in frecs:
        print(frec)
        UA = util.read_objects(frec, 'UnitArr')
        test_DS(UA)
        util.write_objects({'UnitArr': UA}, frec)


def test_DS(UA):
    """Test DS if it has not been tested yet."""

    [u.test_DS() for u in UA.iter_thru() if not len(u.DS)]


def exclude_low_DS(UA, dsi_th=0.3, stims=None):
    """Exclude units with low DS."""

    if stims is None:
        stims = ['S1', 'S2']
    nstart = len(UA.utids())

    # Get DS results.
    DSs = {stim: get_DSInfo_table(UA, None, stim) for stim in stims}
    DSs = pd.concat(DSs, axis=1)

    # Threshold DSI across stimuli.
    th_dsi = [DSs[(stim, 'DSI')] > dsi_th for stim in stims]
    to_inc = pd.concat(th_dsi, axis=1).any(1)

    # Exclude below threshold DSI units.
    for u in UA.iter_thru():
        to_exclude = not to_inc[tuple(u.get_utid())]
        u.set_excluded(to_exclude)

    # Report exclusion stats.
    nexc = nstart - len(UA.utids())
    pexc = int(100*nexc/nstart)
    print('Excluded {}/{} ({}%) of all units'.format(nexc, nstart, pexc))

    return UA
