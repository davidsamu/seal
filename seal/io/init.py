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
from seal.object import unit, unitarray


def convert_TPL_to_Seal(data_dir, task_info, task_constants):
    """Convert TPLCells to Seal objects in project directory."""

    print('\nStarting unit conversion...')

    # Data directory with all recordings to be processed in subfolders.
    rec_data_dir = data_dir + 'recordings/'

    # Go through each session.
    for recording in sorted(os.listdir(rec_data_dir)):

        if recording[0] == '_':
            continue

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

        # Init task parameters common across tasks.
        kset = task_constants['kset']
        answ_par = task_constants['answ_par']
        task_consts = dict((k, v) for k, v in task_constants.items()
                           if k not in ['kset', 'answ_par'])

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
            params = [(TPLCell, task, task_info.loc[task], task_consts,
                       kset, answ_par) for TPLCell in TPLCells]
            tUnits = util.run_in_pool(unit.Unit, params)

            # Add them to unit list of recording, combining all tasks.
            UA.add_task(task, tUnits)

        # Save Units.
        fname_seal = seal_dir + recording + '.data'
        util.write_objects({'UnitArr': UA}, fname_seal)


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
        test_units.quality_test(UA, ftempl, plot_qm, fselection)

        # Report unit exclusion stats.
        test_units.report_unit_exclusion_stats(UA)

        # Test stability of recording session across tasks.
        if plot_stab:
            print('  Plotting recording stability...')
            fname = qc_dir + 'recording_stability.png'
            test_units.rec_stability_test(UA, fname)

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


def unit_activity(proj_dir, plot_DR=True, plot_sel=True):
    """Plot basic unit activity figures."""

    print('\nStarting plotting unit activity...')
    putil.inline_off()

    # Init folders.
    data_dir = proj_dir + 'data/'
    out_dir = proj_dir + 'results/basic_activity/'

    ftempl_dr = out_dir + 'direction_response/{}.png'
    ftempl_sel = out_dir + 'stimulus_selectivity/{}.png'

    # Read in Units.
    print('  Reading in UnitArray...')
    f_data = data_dir + 'all_recordings.data'
    UA = util.read_objects(f_data, 'UnitArr')
    UA.clean_array(keep_excl=False)

    # Test stimulus response to all directions.
    if plot_DR:
        print('  Plotting direction response...')
        test_units.DR_plot(UA, ftempl_dr)

    # Plot feature selectivity summary plots.
    if plot_sel:
        print('  Plotting selectivity summary figures...')
        test_units.selectivity_summary(UA, ftempl_sel)

    # Re-enable inline plotting
    putil.inline_on()
