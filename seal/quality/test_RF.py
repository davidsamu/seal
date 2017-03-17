# -*- coding: utf-8 -*-
"""
Functions related to receptive field cleanness and coverage.

@author: David Samu
"""

import warnings

import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns

from seal.util import util, ua_query
from seal.plot import putil



# Path to RF mapping tables.
fbase = '/home/upf/Research/data/RF mapping/'
fRF = pd.Series([fbase + '202/202_RF_all_sessions.xlsx'],
                index=['202'])


# %% Misc functions.

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
            continue

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


def get_unit_results(u, RFres):
    """Return RF mapping results of given unit."""

    rec, ch, idx, task = u.get_utid()
    uidx = (RFres.recording == rec) & (RFres.channel == ch)
    rfres = RFres.loc[uidx].squeeze()

    uRFres = rfres[['cntr_x', 'cntr_y', 'FWHM', 'R2']]
    uRFres.name = (rec, ch, idx, task)

    return uRFres


def plot_RF_DS_relation(UA, fdir):
    """
    Plot relation between cleanness of mapping (fit R2) and
    direction selectivity of units per task.
    """

    # Init.
    stims = ('S1', 'S2')
    RFres = get_RF_mapping_results(UA.recordings())
    if RFres.empty:  # no RF mapping was done
        return
    # Test DS in case it hasn't been tested yet.
    ua_query.test_DS(UA)

    # Prepare RF mapping and DS results.
    RF_DS_res = []
    for u in UA.iter_thru():
        # Get RF mapping results.
        uRF_DSres = get_unit_results(u, RFres)
        x, y = uRF_DSres[['cntr_x', 'cntr_y']]
        for stim in stims:
            # Get DS.
            uRF_DSres[stim+'_mDSI'] = u.DS.DSI.mDS[stim]
            # Calculate distance of RF center from each stimulus.
            stim_locs = list(u.TrData[(stim, 'Loc')].unique())
            for i, stimloc in enumerate(stim_locs):
                dist = sp.spatial.distance.euclidean([x, y], stimloc)
                name = stim + '_dist' + ('_'+str(i+1) if i > 0 else '')
                uRF_DSres[name] = dist
        RF_DS_res.append(uRF_DSres)
    RF_DS_res = pd.concat(RF_DS_res, axis=1).T
    RF_DS_res = RF_DS_res.astype(float)

    # Plot RF coverage and DS on regression plot.
    tasks = RF_DS_res.index.get_level_values(3).unique()
    fig, gs, axes = putil.get_gs_subplots(nrow=len(stims), ncol=len(tasks),
                                          subw=4, subh=4, ax_kws_list=None,
                                          create_axes=True)
    colors = sns.color_palette('muted', len(tasks))
    for istim, stim in enumerate(stims):
        for itask, task in enumerate(tasks):
            # Plot regression plot.
            ax = axes[istim, itask]
            sDSI = stim+'_mDSI'
            df = RF_DS_res.xs(task, level=3)
            sns.regplot('R2', sDSI, df, color=colors[itask], ax=ax)
            # Set title.
            r, p = sp.stats.pearsonr(df.R2, df[sDSI])
            pstr = util.format_pvalue(p)
            title = '{} {}: r = {:.2f} ({})'.format(task, stim, r, pstr)
            xlab = 'RF mapping R2'
            ylab = stim + ' DSI'
            putil.set_labels(ax, xlab, ylab, title)

    # Save plot.
    title = UA.Name
    ffig = fdir + '.png'
    putil.save_fig(ffig, fig, title, ytitle=1.1)


# %% Exclude units with low RF coverage.

def exclude_uncovered_units(UA, min_RF_R2=0.5, exc_unmapped=False):
    """Exclude units from UnitArray with low RF coverage."""

    # Get RF mapping results.
    RFres = get_RF_mapping_results(UA.recordings())

    # For each unit (channel) test distance of each stimulus location
    # from RF center.
    nstart = len(UA.utids())
    nnoRF = 0
    for u in UA.iter_thru():

        # Get results of unit.
        uRFres = get_unit_results(u, RFres)

        # No RF mapping data found for unit.
        if uRFres.empty:
            u.set_excluded(exc_unmapped)
            nnoRF += 1
            continue

        x, y, FWHM, R2 = uRFres

        # Get stimulus center locations.
        stims = ('S1', 'S2')
        s1locs, s2locs = [list(u.TrData[(stim, 'Loc')].unique())
                          for stim in stims]

        # Calculate distance of RF center from each stimulus.
        dists = np.array([sp.spatial.distance.euclidean([x, y], stimloc)
                          for stimloc in s1locs + s2locs])

        # Exclude unit if
        # 1) it does not have a strong RF, or
        # 2) all stimulus presented are away from RF.
        if R2 > min_RF_R2 or np.all(dists > FWHM):
            u.set_excluded(True)

    # Report some stats on unit exclusion.
    nexc = nstart - len(UA.utids())
    pexc = int(100*nexc/nstart)
    print('Excluded {}/{} ({}%) of units'.format(nexc, nstart, pexc))
    if nnoRF != 0:
        pnoRF = int(100*nnoRF/nstart)
        snorf = 'and were ' if exc_unmapped else 'but were not '
        print('{}/{} ({}%) of units '.format(nnoRF, nstart, pnoRF) +
              'had no RF mapping data {} excluded.'.format(snorf))

    return UA
