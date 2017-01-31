#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of functions for plotting unit activity and direction selectivity
for different sets of trials or trial periods.

@author: David Samu
"""

import os
import warnings

import numpy as np
import pandas as pd

from seal.io import export
from seal.util import util
from seal.object import unitarray
from seal.quality import test_sorting
from seal.plot import putil, pquality
from seal.quality import test_stability

# Figure size constants
subw = 7
w_pad = 5


# %% Quality tests across tasks.

def get_selection_params(u, UnTrSel=None):
    """Return unit and trial selection parameters of unit provided by user."""

    if UnTrSel is None or u.is_empty():
        return None, None, None

    # Init.
    rec, ch, idx, task = u.get_utid()
    ntrials = len(u.TrialParams.index)

    # Find unit in selection table.
    row = UnTrSel.ix[((UnTrSel.recording == rec) & (UnTrSel.channel == ch) &
                      (UnTrSel['unit index'] == idx) & (UnTrSel.task == task))]
    uname = 'rec {}, ch {}, idx {}, task {}'.format(rec, ch, idx, task)

    # Unit not in table.
    if not len(row.index):
        warnings.warn(uname + ': not found in selection table.')
        return None, None, None

    # If there's more than one match.
    if len(row.index) > 1:
        warnings.warn(uname + ': multiple rows found for unit ' +
                      'in selection table, using first match.')
        row = row.iloc[0:1]

    # Get index of first and last trials to include.
    include = bool(int(row['unit included']))
    first_tr = int(row['first included trial']) - 1  # indexing starts with 0
    last_tr = int(row['last included trial'])   # inclusive --> exclusive

    # Set values outside of range to limits.
    first_tr = max(first_tr, 0)
    last_tr = min(last_tr, ntrials)
    if last_tr == -1:  # -1: include all the way to the end
        last_tr = ntrials

    # Check some simple cases of data inconsistency.
    if include and first_tr >= last_tr:
        warnings.warn(uname + ': index of first included trial is larger or' +
                      ' equal to last in selection table! Excluding unit.')
        include = False

    return include, first_tr, last_tr


def quality_test(UA, ftempl=None, plot_qm=False, fselection=None):
    """Test and plot quality metrics of recording and spike sorting """

    # Init plotting theme.
    putil.set_style('notebook', 'ticks')

    # Import unit&trial selection file.
    UnTrSel = pd.read_excel(fselection) if (fselection is not None) else None

    # For each unit over all tasks.
    for uid in UA.uids():

        # Init figure.
        if plot_qm:
            fig, gsp, _ = putil.get_gs_subplots(nrow=1, ncol=len(UA.tasks()),
                                                subw=subw, subh=1.6*subw)
            wf_axs, amp_axs, dur_axs, amp_dur_axs, rate_axs = ([], [], [],
                                                               [], [])

        for i, task in enumerate(UA.tasks()):

            # Do quality test.
            u = UA.get_unit(uid, task)
            include, first_tr, last_tr = get_selection_params(u, UnTrSel)
            res = test_sorting.test_qm(u, include, first_tr, last_tr)

            # Plot QC results.
            if plot_qm:

                if res is not None:
                    ax_res = pquality.plot_qm(u, fig=fig, sps=gsp[i], **res)

                    # Collect axes.
                    ax_wfs, ax_wf_amp, ax_wf_dur, ax_amp_dur, ax_rate = ax_res
                    wf_axs.extend(ax_wfs)
                    amp_axs.append(ax_wf_amp)
                    dur_axs.append(ax_wf_dur)
                    amp_dur_axs.append(ax_amp_dur)
                    rate_axs.append(ax_rate)

                else:
                    putil.add_mock_axes(fig, gsp[i])

        if plot_qm:

            # Match axis scales across tasks.
            putil.sync_axes(wf_axs, sync_x=True, sync_y=True)
            putil.sync_axes(amp_axs, sync_y=True)
            putil.sync_axes(dur_axs, sync_y=True)
            putil.sync_axes(amp_dur_axs, sync_x=True, sync_y=True)
            putil.sync_axes(rate_axs, sync_y=True)
            [putil.move_event_lbls(ax, y_lbl=0.92) for ax in rate_axs]

            # Save figure.
            if ftempl is not None:
                uid_str = util.format_uid(uid)
                title = uid_str.replace('_', ' ')
                ffig = ftempl.format(uid_str)
                putil.save_fig(ffig, fig, title, rect_height=0.92,
                               w_pad=w_pad)


def report_unit_exclusion_stats(UA, fname):
    """Exclude low quality units."""

    exclude = []
    for u in UA.iter_thru(excl=True):
        exclude.append(u.is_excluded())

    # Log unit exclusion results into file.
    n_tot = len(exclude)
    n_exc, n_inc = sum(exclude), sum(np.invert(exclude))
    perc_exc, perc_inc = 100 * n_exc / n_tot, 100 * n_inc / n_tot
    rep_str = '  {} / {} ({:.1f}%) units {} analysis.\n'

    with open(fname, 'w') as f:
        f.write(UA.Name + '\n\n')
        f.write(rep_str.format(n_inc, n_tot, perc_inc, 'included into'))
        f.write(rep_str.format(n_exc, n_tot, perc_exc, 'excluded from'))


def quality_control(data_dir, proj_name, task_order, plot_qm=True,
                    plot_stab=True, fselection=None):
    """Run quality control (SNR, rate drift, ISI, etc) on each recording."""

    # Data directory with all recordings to be processed in subfolders.
    rec_data_dir = data_dir + 'recordings/'

    # Init combined UnitArray object.
    combUA = unitarray.UnitArray(proj_name, task_order)

    print('\nStarting quality control...')
    putil.inline_off()

    for recording in sorted(os.listdir(rec_data_dir)):

        if recording[0] == '_':
            continue

        # Report progress.
        print('  ' + recording)

        # Init folders.
        rec_dir = rec_data_dir + recording + '/'
        seal_dir = rec_dir + 'SealCells/'
        qc_dir = rec_dir + 'quality_control/'

        # Read in Units.
        f_data = seal_dir + recording + '.data'
        UA = util.read_objects(f_data, 'UnitArr')

        # Test unit quality, save result figures, add stats to units and
        # exclude low quality trials and units.
        ftempl = qc_dir + 'quality_metrics/{}.png'
        quality_test(UA, ftempl, plot_qm, fselection)

        # Report unit exclusion stats.
        report_unit_exclusion_stats(UA)

        # Test stability of recording session across tasks.
        if plot_stab:
            print('  Plotting recording stability...')
            fname = qc_dir + 'recording_stability.png'
            test_stability.rec_stability_test(UA, fname)

        # Add to combined UA.
        combUA.add_recording(UA)

    # Add index to unit names.
    combUA.index_units()

    # Save Units with quality metrics added.
    print('\nExporting combined UnitArray...')
    fname = data_dir + '/all_recordings.data'
    util.write_objects({'UnitArr': combUA}, fname)

    # Export unit and trial selection results.
    if fselection is None:
        print('Exporting automatic unit and trial selection results...')
        fname = data_dir + '/unit_trial_selection.xlsx'
        export.export_unit_trial_selection(combUA, fname)

    # Export unit list.
    print('Exporting combined unit list...')
    export.export_unit_list(combUA, data_dir + '/unit_list.xlsx')

    # Re-enable inline plotting
    putil.inline_on()
