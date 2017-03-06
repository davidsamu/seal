# -*- coding: utf-8 -*-
"""
Functions related to receptive field cleanness and coverage.

@author: David Samu
"""

import warnings

import numpy as np
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































