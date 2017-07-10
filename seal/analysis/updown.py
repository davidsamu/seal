# -*- coding: utf-8 -*-
"""
Functions to perform up-down state analysis.

@author: David Samu
"""

import numpy as np
import pandas as pd

import neo
from quantities import s, ms
from elephant.statistics import time_histogram

from seal.util import util, ua_query, constants


def combine_units(spk_trs):
    """Combine spikes across units into a single spike train."""

    t_start = spk_trs[0].t_start
    t_stop = spk_trs[0].t_stop
    comb_spks = np.sort(np.concatenate([np.array(spk_tr)
                        for spk_tr in spk_trs]))

    comb_spk_tr = neo.core.SpikeTrain(comb_spks*s, t_start=t_start,
                                      t_stop=t_stop)

    return comb_spk_tr


def get_spike_times(UA, recs, task, tr_prd='whole trial', ref_ev='S1 on'):
    """Return spike times ."""

    # Init.
    if recs is None:
        recs = UA.recordings()

    # Get spikes times for period for each recording.
    lSpikes = []
    for rec in recs:
        print(rec)
        uids = UA.utids(tasks=[task], recs=[rec]).droplevel('task')
        spks = ua_query.get_spike_times(UA, rec, task, uids, tr_prd, ref_ev)
        lSpikes.append(spks)

    Spikes = pd.concat(lSpikes)

    return Spikes


def get_binned_spk_cnts(comb_spk_tr, prd, binsize):
    """
    Return binned spike counts during period for spike train
    (tyically combined across units).
    """

    tstart, tstop = constants.fixed_tr_prds.loc[prd]

    # Calculate binned spike counts.
    lspk_cnt = [np.array(time_histogram([spk_tr], binsize, tstart,
                                        min(tstop, spk_tr.t_stop)))[:, 0]
                for spk_tr in comb_spk_tr]
    tvec = util.quantity_arange(tstart, tstop, binsize).rescale(ms)

    # Deal with varying number of bins.
    idxs = range(np.array([len(sc) for sc in lspk_cnt]).min())
    lspk_cnt = [sc[idxs] for sc in lspk_cnt]
    tvec = tvec[idxs]

    # Create trial x time bin spike count DF.
    spk_cnt = pd.DataFrame(np.array(lspk_cnt), columns=np.array(tvec))

    return spk_cnt
