# -*- coding: utf-8 -*-
"""
Functions related to receptive field cleanness and coverage.

@author: David Samu
"""

import warnings

import numpy as np
import scipy as sp
import pandas as pd


# Path to RF mapping tables.
fbase = '/home/upf/Research/data/RF mapping/'
fRF = pd.Series([fbase + '202/202_RF_all_sessions.xlsx'],
                index=['202'])


# %% Import RF results.

def get_RF_mapping_results(recs, best_rec=True):
    """Return RF mapping results for given set of recordings."""

    # Import RF mapping result table.
    allRFres = {monkey: pd.read_excel(fname)
                for monkey, fname in fRF.iteritems()}
    allRFres = pd.concat(allRFres)

    # Select best RF mapping for each recording.
    RFrecs = []
    for rec in recs:

        RF_rec_names = allRFres.RF_rec_name[allRFres.recording == rec].unique()

        # If no RF mapping results are available.
        if not len(RF_rec_names):
            warnings.warn('Could not find RF mapping results for '+rec)

        # Select single best RF recording.
        if best_rec:
            # Remove the ones containing the string 'Ipsi' or 'ipsi'.
            RF_rec_names = [fRF for fRF in RF_rec_names
                            if 'ipsi' not in fRF.lower()]

            # Select the RF recording with highest number of trials.
            ntrs = [allRFres.ntrials[allRFres.RF_rec_name == fRF].mean()
                    for fRF in RF_rec_names]
            RF_rec_names = [RF_rec_names[np.argmax(ntrs)]]

        RFrecs.extend(RF_rec_names)

    # Select results of selected recordings.
    RFres = allRFres.loc[allRFres.RF_rec_name.isin(RFrecs)]

    return RFres


# %% Exclude units with low RF coverage.

def exclude_uncovered_units(UA):
    """Exclude units from UnitArray with low RF coverage."""

    # Get RF mapping results.
    RFres = get_RF_mapping_results(UA.recordings())

    # For each unit (channel) test distance of each stimulus location
    # from RF center.
    nstart = len(UA.utids())
    for u in UA.iter_thru():

        # Get results of unit.
        rec, ch, idx, task = u.get_utid()
        uidx = (RFres.recording == rec) & (RFres.channel == ch)
        rfres = RFres.loc[uidx].squeeze()
        x, y, FWHM, R2 = rfres[['cntr_x', 'cntr_y', 'FWHM', 'R2']]

        # Get stimulus center locations.
        stims = ('S1', 'S2')
        s1locs, s2locs = [list(u.TrData[(stim, 'Loc')].unique())
                          for stim in stims]

        # Calculate distance of RF center from each stimulus.
        dists = np.array([sp.spatial.distance.euclidean([x, y], stimloc)
                          for stimloc in s1locs + s2locs])

        # Exclude unit if
        # 1) it has a strong RF away from all stimulus presented, and
        # 2) it has low DS during task and
        # 3) task is not Remote task
        if (R2 > 0.5 and np.all(dists > FWHM) and u.DS.DSI.mDS.max() < 0.3 and
           'Rem' not in task):
            u.set_excluded(True)

    # Report some stats on unit exclusion.
    nexc = nstart - len(UA.utids())
    pexc = int(100*nexc/nstart)
    print('Excluded {}/{} ({}%) of all units'.format(nexc, nstart, pexc))

    return UA
