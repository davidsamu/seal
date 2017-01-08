#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions related to importing recorded datasets.

@author: David Samu
"""

import os

import numpy as np

from seal.io import export
from seal.util import util
from seal.plot import putil
from seal.quality import test_units
from seal.object import constants, unit, unitarray


def convert_TPL_to_Seal(data_dir):
    """Convert TPLCells to Seal objects in project directory."""

    print('\nStarting unit import...\n')

    # Data directory with all recordings to be processed in subfolders.
    rec_data_dir = data_dir + '/recordings/'

    # Go through each session.
    for recording in sorted(os.listdir(rec_data_dir)):
        print(recording)

        # Init folders.
        rec_dir = rec_data_dir + recording + '/'
        tpl_dir = rec_dir + 'TPLCells/'
        seal_dir = rec_dir + 'SealCells/'

        # Get all available task files.
        f_rec_tasks = sorted(os.listdir(tpl_dir))

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
            fname_matlab = tpl_dir + f_rec_tasks[i]
            TPLCells = util.read_matlab_object(fname_matlab, 'TPLStructs')

            # Create list of Units from TPLCell structures.
            region = constants.task_info.loc[task, 'region']
            kernels = constants.kset
            step, stim_params = constants.step, constants.stim_params,
            answ_params, stim_dur = constants.answ_params, constants.stim_dur
            tr_evts = constants.tr_evts
            params = [(TPLCell, region, task, kernels, step, stim_params,
                       answ_params, stim_dur, tr_evts) for TPLCell in TPLCells]
            tUnits = util.run_in_pool(unit.Unit, params)

            # Add them to unit list of recording, combining all tasks.
            UA.add_task(task, tUnits)

        # Save Units.
        fname_seal = seal_dir + recording + '.data'
        util.write_objects({'UnitArr': UA}, fname_seal)


def run_quality_control(data_dir, ua_name, plot_QM=True, fselection=None):
    """Run quality control (SNR, rate drift, ISI, etc) on each recording."""

    # Data directory with all recordings to be processed in subfolders.
    rec_data_dir = data_dir + '/recordings/'

    # Init combined UnitArray object.
    combUA = unitarray.UnitArray(ua_name)

    print('\nStarting quality control...\n')
    putil.inline_off()

    for recording in sorted(os.listdir(rec_data_dir)):

        # Report progress.
        print(recording)

        # Init folders.
        rec_dir = rec_data_dir + recording + '/'
        seal_dir = rec_dir + 'SealCells/'
        qc_dir = rec_dir + '/quality_control/'

        ftempl_qm = qc_dir + 'quality_metrics/{}.png'

        # Read in Units.
        f_data = seal_dir + recording + '.data'
        UA = util.read_objects(f_data, 'UnitArr')

        # Test unit quality, save result figures,
        # add stats to units and exclude low quality trials and units.
        test_units.quality_test(UA, ftempl_qm, plot_QM, fselection)

        # Exclude units with low recording quality.
        if fselection is None:
            print('  Excluding units...')
            test_units.exclude_units(UA)

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


def run_preprocessing(data_dir, ua_name, plot_DR=True, plot_sel=True,
                      plot_stab=True, creat_montage=True):
    """
    Run preprocessing on Units and UnitArrays, including
      - location and direction selectivity tests,
      - recording stability test,
      - exporting automatic unit and trial selection results.
    """

    # Data directory with all recordings to be processed in subfolders.
    rec_data_dir = data_dir + '/recordings/'

    # Init data structures.
    combUA = unitarray.UnitArray(ua_name)

    print('\nStarting quality control...\n')
    putil.inline_off()

    for recording in sorted(os.listdir(rec_data_dir)):

        # Report progress.
        print(recording)

        # Init folders.
        rec_dir = rec_data_dir + recording + '/'
        seal_dir = rec_dir + 'SealCells/'
        qc_dir = rec_dir + '/quality_control/'

        ftempl_qm = qc_dir + 'quality_metrics/{}.png'
        ftempl_dr = qc_dir + 'direction_response/{}.png'
        ftempl_sel = qc_dir + 'stimulus_selectivity/{}.png'
        ftempl_mont = qc_dir + 'montage/{}.png'

        # Read in Units.
        f_data = seal_dir + recording + '.data'
        UA = util.read_objects(f_data, 'UnitArr')

        # Test stimulus response to all directions.
        if plot_DR:
            print('  Plotting direction response...')
            test_units.DR_plot(UA, ftempl_dr)

        # Test direction selectivity.
        print('  Calculating direction selectivity...')
        for u in UA.iter_thru(excl=True):
            u.test_DS()

        # Plot feature selectivity summary plots.
        if plot_sel:
            print('  Plotting selectivity summary figures...')
            test_units.selectivity_summary(UA, ftempl_sel)

        # Create montage image of all plots figures above.
        if creat_montage:
            print('  Creating montage images...')
            test_units.create_montage(UA, ftempl_qm, ftempl_dr,
                                      ftempl_sel, ftempl_mont)

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
    fname = data_dir + '/all_recordings.data'
    util.write_objects({'UnitArr': combUA}, fname)

    # Export unit and trial selection results.
    print('Exporting automatic unit and trial selection results...')
    fname = data_dir + '/unit_trial_selection.xlsx'
    export.export_unit_trial_selection(combUA, fname)

    # Export unit list.
    print('Exporting combined unit list...')
    export.export_unit_list(combUA, data_dir + '/unit_list.xlsx')

    # Re-enable inline plotting
    putil.inline_on()
