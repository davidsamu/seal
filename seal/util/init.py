#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 12:23:54 2016

Function related to importing recorded datasets.

@author: David Samu
"""

import os

import numpy as np

from seal.util import util, plot
from seal.object import constants, unit, unitarray


def convert_TPL_to_Seal(tpl_dir, seal_dir, sub_dirs=[''],
                        kernels=constants.R100_kernel):
    """Convert TPLCells to Seal objects."""

    print('\nStarting unit import...\n')

    for sub_dir in sub_dirs:

        # Init folders.
        tpl_sub_dir = tpl_dir + sub_dir + '/'
        seal_sub_dir = seal_dir + sub_dir + '/'

        # Go through each session.
        for recording in sorted(os.listdir(tpl_sub_dir)):
            print(recording)

            # Get all available task files.
            f_rec_tasks = os.listdir(tpl_sub_dir + recording)

            # Extract task names and indices from file names.
            tasks = [rtdir.split('_')[2] for rtdir in f_rec_tasks]
            tasks, itasks = zip(*[(tsk[:-1], int(tsk[-1])) for tsk in tasks])

            # Reorder sessions by task order.
            task_order = np.argsort(itasks)

            # Create and collect all units from each task.
            UA = unitarray.UnitArray(recording)
            for i in task_order:

                # Report progress.
                print('  ', itasks[i], tasks[i])

                # Load in Matlab structure (SimpleTPLCell).
                fname_matlab = tpl_sub_dir + recording + '/' + f_rec_tasks[i]
                TPLCells = util.read_matlab_object(fname_matlab, 'TPLStructs')

                # Create list of Units from TPLCell structures.
                params = [(TPLCell, constants.t_start, constants.t_stop,
                           kernels, constants.step, constants.tr_params)
                          for TPLCell in TPLCells]
                tUnits = util.run_in_pool(unit.Unit, params)

                # Add them to unit list of recording, combining all tasks.
                UA.add_task(tasks[i], tUnits)

            # Save Units.
            rec_dir_no_qc = seal_sub_dir + recording + '/before_qc/'
            fname_seal = rec_dir_no_qc + recording + '.data'
            util.write_objects({'UnitArr': UA}, fname_seal)

            # Write out unit list and save parameter plot.
            UA.save_params_table(rec_dir_no_qc + 'unit_list.xlsx')
            UA.plot_params(rec_dir_no_qc + 'unit_params.png')


def run_preprocessing(data_dir, ua_name):
    """
    Run preprocessing on Units and UnitArrays, including
        - standard quality control of each unit (SNR, FR drift, ISI, etc)
        - stimulus selectivity (DS)
        - stability of recording(s)
    """

    # Init plotting theme and style.
    rc_args = {'font.family': u'Arial'}
    plot.set_seaborn_style_context('white', 'notebook', rc_args)
    plot.inline_off()  # turn it off to flush plotted figures out of memory!

    # Init data structures.
    UA = unitarray.UnitArray(ua_name)
    task_order = []  # TODO: how to deal with this??

    print('\nStarting quality control...\n')

    for recording in sorted(os.listdir(data_dir)):

        # Report progress.
        print(recording)

        # Init folders.
        rec_dir = data_dir + recording
        bfr_qc_dir = rec_dir + '/before_qc/'
        qc_dir = rec_dir + '/qc/'
        aft_qc_dir = rec_dir + '/after_qc/'

        # Read in Units.
        f_data_bfr_qc = bfr_qc_dir + recording + '.data'
        UA_bfr_qc = util.read_objects(f_data_bfr_qc, 'UnitArr')
        rec_tasks = UA_bfr_qc.tasks()

        # Check direction response over trial time.
    #    ftemp = qc_dir + '/direction_response/{}.png'
    #    quality.direction_response_test(UA_bfr_qc, 'R100', ftemp, tasks=['dd', 'ddX'],
    #                                    match_FR_scale_across_tasks=False)

        # Test unit quality, save result figures,
        # add stats to units and exclude trials and units.
        print('Testing unit quality...')
        ffig_templ = qc_dir + 'quality_metrics/{}.png'
        for u in UA_bfr_qc.iter_thru():
            #quality.test_qm(u, do_trial_rejection=False, ffig_template=ffig_templ)
            _ = quality.test_qm(u, do_trial_rejection=False)  # plotting disabled

        # Test direction selectivity.
        print('Testing direction selectivity...')
        do_plot = True
        ffig_templ = qc_dir + 'direction/{}.png'
        for u in Unit_list_bqc:
            u.test_direction_selectivity(do_plot=do_plot, ffig_tmpl=ffig_templ)

        # Exclude units and trials using Tania's selection.
        f_rejected_trials = 'data/included_unit_trials/' + recording + '.xlsx'
        rej_trials = pd.read_excel(f_rejected_trials)
        # rej_trials.task = [task[:-1] for task in rej_trials.task]
        Unit_list_bqc_2tasks = []
        for u in Unit_list_bqc:
            task = u.SessParams['task']
            ichn = u.SessParams['channel #']
            iunit = u.SessParams['unit #']
            i = np.where((rej_trials.task == task) &
                         (rej_trials.channel == ichn) &
                         (rej_trials.unit == iunit))[0]
            if i.size == 0:
                continue
            elif i.size == 1:
                Unit_list_bqc_2tasks.append(u)
                row = rej_trials.iloc[i[0]]
                bstart, bstop, estart, estop = row.tail(4)

                # Assamble list of trials to reject (exclude).
                rej_tr = []
                if not (bstart == -1 and bstop == -1):
                    rej_tr += list(range(bstart-1, bstop))
                if not (estart == -1 and estop == -1):
                    rej_tr += list(range(estart-1, estop))

                # Exclude trials (if any).
                ntrials = u.TrialParams.index.size
                tr_inc = np.array(ntrials*[True])
                tr_inc[rej_tr] = False
                tr_exc = np.invert(tr_inc)
                u.QualityMetrics['NTrialsTotal'] = ntrials
                u.QualityMetrics['NTrialsIncluded'] = np.sum(tr_inc)
                u.QualityMetrics['NTrialsExcluded'] = np.sum(tr_exc)
                u.QualityMetrics['IncludedTrials'] = trials.Trials(tr_inc, 'included trials')
                u.QualityMetrics['ExcludedTrials'] = trials.Trials(tr_exc, 'excluded trials')

            else:   # more than one match
                print('Warning: multiple rows match unit!')

        # Extend full set of tasks by potentially new tasks from this recording.
        task_order.extend([t for t in rec_tasks if t not in task_order])

        # Add remaining units to dictionary containing units by task.
        Unit_list.extend(Unit_list_bqc_2tasks)

    # Add unit's index to unit name.
    task_order = task_order[:2]
    UA = unitarray.UnitArray("PFC Inactivation (Tania's selection)",
                                  Unit_list, task_order)
    for task in UA.tasks():
        Unit_task_list = UA.unit_list([task], return_empty=True)
        for i, u in enumerate(Unit_task_list):
            if not u.is_empty():
                u.Name = 'Unit {:0>3}  '.format(i+1) + u.Name

    # Save selected Units with quality metrics and direction selectivity.
    ts = util.timestamp()
    n_units = UA.n_units()
    data_dir = 'data/Units_qc_combined/ChingChi_inactivation_n{}_{}/'.format(n_units, ts)
    util.write_objects({'UnitArr': UA}, data_dir + 'units_combined.data')

    # Write out unit list and save parameter plot.
    UA.save_params_table(data_dir + 'unit_list.xlsx')
    UA.plot_params(data_dir + 'unit_params.png')






## Test
#TPLCell = TPLCells[0]
#u = unit.Unit(TPLCell, constants.t_start, constants.t_stop,
#              kernels, constants.step, constants.tr_params)