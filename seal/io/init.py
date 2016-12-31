#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions related to importing recorded datasets.

@author: David Samu
"""

import os
import warnings

import numpy as np
import pandas as pd

from seal.io import export
from seal.util import util
from seal.plot import putil
from seal.quality import test_units
from seal.object import constants, unit, unitarray


def convert_TPL_to_Seal(tpl_dir, seal_dir, kernels=constants.R100_kernel,
                        region=None):
    """Convert TPLCells to Seal objects."""

    print('\nStarting unit import...\n')

    # Go through each session.
    for recording in sorted(os.listdir(tpl_dir)):
        print(recording)

        # Get all available task files.
        f_rec_tasks = sorted(os.listdir(tpl_dir + recording))

        # Extract task names and indices from file names.
        task_idxs = [rtdir.split('_')[2] for rtdir in f_rec_tasks]
        tasks, itasks = zip(*[(ti[:-1], int(ti[-1])) for ti in task_idxs])

        # Add distinguishing letter to end of tasks with same name.
        tasks = [t if tasks.count(t) == 1 else t+str(tasks[:(i+1)].count(t))
                 for i, t in enumerate(tasks)]

        # Reorder sessions by task order.
        task_order = np.argsort(itasks)

        # Create and collect all units from each task.
        UA = unitarray.UnitArray(recording)
        for i in task_order:

            # Report progress.
            task = tasks[i]
            print('  ', itasks[i], task)

            # Load in Matlab structure (SimpleTPLCell).
            fname_matlab = tpl_dir + recording + '/' + f_rec_tasks[i]
            TPLCells = util.read_matlab_object(fname_matlab, 'TPLStructs')

            # Create list of Units from TPLCell structures.
            step, stim_params = constants.step, constants.stim_params,
            answ_params, stim_dur = constants.answ_params, constants.stim_dur
            tr_evt = constants.tr_evt
            params = [(TPLCell, kernels, step, stim_params, answ_params,
                       stim_dur, tr_evt, task, region)
                      for TPLCell in TPLCells]
            tUnits = util.run_in_pool(unit.Unit, params)

            # Add them to unit list of recording, combining all tasks.
            UA.add_task(task, tUnits)

        # Save Units.
        rec_dir_no_qc = seal_dir + recording + '/before_qc/'
        fname_seal = rec_dir_no_qc + recording + '.data'
        util.write_objects({'UnitArr': UA}, fname_seal)


def run_preprocessing(data_dir, ua_name, plot_QM=True, plot_SR=True,
                      plot_sum=True, plot_stab=True, creat_montage=True):
    """
    Run preprocessing on Units and UnitArrays, including
      - standard quality control of each unit (SNR, rate drift, ISI, etc),
      - direction selectivity (DS) test,
      - recording stability test,
      - exporting automatic unit and trial selection results.
    """

    # Init data structures.
    combUA = unitarray.UnitArray(ua_name)

    print('\nStarting quality control...\n')
    putil.inline_off()

    for recording in sorted(os.listdir(data_dir)):

        # Report progress.
        print(recording)

        # Init folders.
        rec_dir = data_dir + recording
        bfr_qc_dir = rec_dir + '/before_qc/'
        qc_dir = rec_dir + '/qc_res/'

        ftempl_qm = qc_dir + 'quality_metrics/{}.png'
        ftempl_dr = qc_dir + 'direction_response/{}.png'
        ftempl_sum = qc_dir + 'rate_DS_summary/{}.png'
        ftempl_mont = qc_dir + 'montage/{}.png'

        # Read in Units.
        f_data = bfr_qc_dir + recording + '.data'
        UA = util.read_objects(f_data, 'UnitArr')

        # Test unit quality, save result figures,
        # add stats to units and exclude trials and units.
        print('  Testing unit quality...')
        test_units.quality_test(UA, ftempl_qm, plot_QM)

        # Exclude units with low recording quality.
        exclude_units(UA)

        # Test stimulus response to all directions.
        if plot_SR:
            print('  Plotting direction response...')
            test_units.DS_test(UA, ftempl_dr)

        # Test direction selectivity.
        print('  Calculating direction selectivity...')
        for u in UA.iter_thru(excl=True):
            u.test_DS()

        # Plot trial rate and direction selectivity summary plots.
        if plot_sum:
            print('  Plotting summary figures (rates and DS)...')
            test_units.rate_DS_summary(UA, ftempl_sum)

        # Create montage image of all plots figures above.
        if creat_montage:
            print('  Creating montage images...')
            test_units.create_montage(UA, ftempl_qm, ftempl_dr,
                                      ftempl_sum, ftempl_mont)

        # Test stability of recording session across tasks.
        if plot_stab:
            print('  Plotting recording stability...')
            fname = qc_dir + 'recording_stability.png'
            test_units.rec_stability_test(UA, fname)

        # Add to combined UA
        combUA.add_recording(UA)

    # Add index to unit names.
    combUA.index_units()

    # Save selected Units with quality metrics and direction selectivity.
    print('\nExporting combined UnitArray...')
    comb_data_dir = 'data/all_recordings/'
    fname = comb_data_dir + 'all_recordings.data'
    util.write_objects({'UnitArr': combUA}, fname)

    # Export unit and trial selection results.
    print('Exporting automatic unit and trial selection results...')
    fname = comb_data_dir + 'unit_trial_selection.xlsx'
    export.export_unit_trial_selection(combUA, fname)

    # Export unit list.
    print('Exporting combined unit list...')
    export.export_unit_list(UA, comb_data_dir + 'unit_list.xlsx')

    # Re-enable inline plotting
    putil.inline_on()


def exclude_units(UA):
    """Exclude units with low quality or no direction selectivity."""

    print('  Excluding units...')
    exclude = []
    for u in UA.iter_thru(excl=True):
        exclude.append(u.is_excluded())

    # Report unit exclusion results.
    n_tot = len(exclude)
    n_exc, n_inc = sum(exclude), sum(np.invert(exclude))
    perc_exc, perc_inc = 100 * n_exc / n_tot, 100 * n_inc / n_tot
    rep_str = '  {} / {} ({:.1f}%) units {} analysis.'
    print(rep_str.format(n_exc, n_tot, perc_exc, 'excluded from'))
    print(rep_str.format(n_inc, n_tot, perc_inc, 'included into'))


def select_unit_and_trials(f_data, f_sel_table, clean_UA=True):
    """Exclude low quality units and trials."""

    # Import combined UnitArray and unit&trial selection file.
    UA = util.read_objects(f_data, 'UnitArr')
    UnTrSel = pd.read_excel(f_sel_table)

    # Update units' selection parameters using table.
    for idx, row in UnTrSel.iterrows():

        # Get unit.
        uid = tuple(row[['session', 'channel', 'unit index']])
        u = UA.get_unit(uid, row['task'])

        # Update unit exclusion.
        exc_unit = not row['unit included']
        u.set_excluded(exc_unit)

        # Get index of first and last trials to include.
        first_tr = row['first included trial'] - 1
        last_tr = row['last included trial'] - 1

        # Set values outside of range to limits.
        ntrs = len(u.TrialParams.index)
        first_tr = max(first_tr, 0)
        last_tr = min(last_tr, ntrs-1)
        if last_tr == -1:  # -1: include all the way to the end
            last_tr = ntrs-1

        # Check some simple cases of data inconsistency.
        if not u.is_excluded and first_tr >= last_tr:
            warnings.warn(u.Name + ': index of first included trial is' +
                          ' larger or equal to last! Excluding unit.')
            u.set_excluded(True)
            continue

        # Update included trials.
        tr_idx = np.arange(first_tr, last_tr+1)
        tr_inc = np.zeros(ntrs, dtype=bool)
        tr_inc[tr_idx] = True
        u.update_included_trials(tr_inc)

    # Clean array: remove uids and tasks with all units empty or excluded.
    if clean_UA:
        UA.clean_array(keep_excl=False)

    # Export updated UnitArray.
    util.write_objects({'UnitArr': UA}, f_data)

    # Export unit list.
    print('\nExporting combined unit list...')
    data_dir = os.path.split(f_data)[0] + '/'
    export.export_unit_list(UA, data_dir + 'unit_list.xlsx')
