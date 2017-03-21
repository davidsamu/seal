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
monkeys = ['201', '202']
fbase = '/home/upf/Research/data/RF mapping/'
fRF = pd.Series([fbase + mky + '_RF_all_sessions.xlsx' for mky in monkeys],
                index=monkeys)


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


def intersect_area(d, r1, r2):
    """
    Return the area of intersection of two circles with given distance from
    each other (d) and radii (r1 and r2).
    """

    if d <= abs(r1-r2):
        # One circle is entirely enclosed in the other.
        overlap = np.pi * min(r1, r2)**2
    elif d >= r1 + r2:
        # The circles don't overlap at all.
        overlap = 0
    else:
        # There's some non-complete overlap.
        r12, r22, d2 = r1**2, r2**2, d**2
        alpha = np.arccos((d2 + r12 - r22) / (2*d*r1))
        beta = np.arccos((d2 + r22 - r12) / (2*d*r2))
        gamma = (r12 * np.sin(2*alpha) + r22 * np.sin(2*beta))
        overlap = r12 * alpha + r22 * beta - 0.5 * gamma

    return overlap


def RF_coverage_analysis(UA, stims, fdir):
    """Relate unit activity to RF coverage."""

    # Init.
    RFres = get_RF_mapping_results(UA.recordings())
    if RFres.empty:  # no RF mapping was done
        return
    # Test DS in case it hasn't been tested yet.
    ua_query.test_DS(UA)

    # Prepare RF mapping and DS results.
    RF_res = []
    for u in UA.iter_thru():
        # Get RF mapping results.
        uRF_res = get_unit_results(u, RFres)
        x, y, FWHM = uRF_res[['cntr_x', 'cntr_y', 'FWHM']]
        RF_rad = FWHM / 2  # approximate circular RF radius by half of FWHM.
        for stim in stims:
            # Get DS.
            uRF_res[stim+'_mDSI'] = u.DS.DSI.mDS[stim]
            # Get stim size.
            stim_sizes = u.TrData[(stim, 'Size')].value_counts()
            if len(stim_sizes) > 1:
                warnings.warn(('More than one simulus sizes found in unit: ' +
                               u.Name + ', using the one with most trials.'))
            stim_rad = stim_sizes.index[0] / 2  # diameter --> radius
            uRF_res[stim+'_rad'] = stim_rad
            # Calculate distance and overlap between RF and each stimulus.
            stim_locs = list(u.TrData[(stim, 'Loc')].unique())
            for i, stimloc in enumerate(stim_locs):
                # Set names.
                postfix = '_'+str(i+1) if i > 0 else ''
                dname, cname = ['{}_{}{}'.format(stim, nm, postfix) for nm in
                                ('dist', 'cover')]
                # Calc distance.
                dist = sp.spatial.distance.euclidean([x, y], stimloc)
                uRF_res[dname] = dist
                # Calc coverage.
                isa = intersect_area(dist, RF_rad, stim_rad)
                RF_area = np.pi * RF_rad**2
                uRF_res[cname] = isa/RF_area
            # Get average rate during each stimulus.
            stim_rates = u.get_prd_rates(stim, add_latency=True)
            # Mean across trials.
            uRF_res[stim+'_mean_rate'] = float(stim_rates.mean())
            # Mean to best direction.
            dir_trs = u.trials_by_param((stim, 'Dir'))
            max_rate = max([float(stim_rates[trs].mean()) for trs in dir_trs])
            uRF_res[stim+'_max_rate'] = max_rate
        RF_res.append(uRF_res)
    RF_res = pd.concat(RF_res, axis=1).T
    RF_res = RF_res.astype(float)

    # Add to UA for later access.
    UA.RF_res = RF_res

    return RF_res


def plot_RF_results(RF_res, stims, fdir, title):
    """Plot receptive field results."""

    # Plot RF coverage and rate during S1 on regression plot for each
    # recording and task.
    tasks = RF_res.index.get_level_values(3).unique()
    for vname, ylim in [('mean_rate', [0, None]), ('max_rate', [0, None]),
                        ('mDSI', [0, 1])]:
        fig, gs, axes = putil.get_gs_subplots(nrow=len(stims), ncol=len(tasks),
                                              subw=4, subh=4, ax_kws_list=None,
                                              create_axes=True)
        colors = sns.color_palette('muted', len(tasks))
        for istim, stim in enumerate(stims):
            for itask, task in enumerate(tasks):
                # Plot regression plot.
                ax = axes[istim, itask]
                scov, sval = [stim + '_' + name for name in ('cover', vname)]
                df = RF_res.xs(task, level=3)
                sns.regplot(scov, sval, df, color=colors[itask], ax=ax)
                # Add stats.
                r, p = sp.stats.pearsonr(df[sval], df[scov])
                pstr = util.format_pvalue(p)
                txt = 'r = {:.2f}, {}'.format(r, pstr)
                ax.text(0.02, 0.98, txt, va='top', ha='left',
                        transform=ax.transAxes)
                # Set labels.
                title = '{} {}'.format(task, stim)
                xlab, ylab = [sn.replace('_', ' ') for sn in (scov, sval)]
                putil.set_labels(ax, xlab, ylab, title)
                # Set limits.
                xlim = [0, 1]
                putil.set_limits(ax, xlim, ylim)

        # Save plot.
        fname = '{}_cover_{}.png'.format(util.format_to_fname(title), vname)
        ffig = util.join([fdir, vname, fname])
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
