#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 12:23:54 2016

Function related to importing recorded datasets.

@author: David Samu
"""

import os

import numpy as np

from seal.util import util
from seal.plot import plot
from seal.quality import test_sorting, test_units
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



def run_preprocessing(data_dir, ua_name, fname, do_plot=True,
                      rej_trials=True, exc_units=False):
    """
    Run preprocessing on Units and UnitArrays, including
      - standard quality control of each unit (SNR, rate drift, ISI, etc)
      - stimulus selectivity (DS)
      - stability of recording(s)
    """

    # Init plotting theme and style.
    rc = {'font.family': 'sans-serif', 'font.sans-serif': 'Helvetica',
          'font.style': u'normal'}
    plot.set_style(context='notebook', style='white', rc=rc)
    plot.inline_off()  # turn it off to flush plotted figures out of memory!

    # Init data structures.
    UA = unitarray.UnitArray(ua_name)

    print('\nStarting quality control...\n')

    for recording in sorted(os.listdir(data_dir)):

        # Report progress.
        print(recording)

        # Init folders.
        rec_dir = data_dir + recording
        bfr_qc_dir = rec_dir + '/before_qc/'
        qc_dir = rec_dir + '/qc_res/'

        # Read in Units.
        f_data = bfr_qc_dir + recording + '.data'
        recUA = util.read_objects(f_data, 'UnitArr')

        # Test unit quality, save result figures,
        # add stats to units and exclude trials and units.
        print('  Testing unit quality...')
        ftempl = qc_dir + 'quality_metrics/{}.png' if do_plot else None
        for u in recUA.iter_thru():
            test_sorting.test_qm(u, rej_trials=rej_trials, ftempl=ftempl)

        # Test stimulus response to all directions.
        if do_plot:
            print('  Plotting direction response...')
            ftempl = qc_dir + 'direction_response/{}.png'
            test_units.direction_response_test(recUA, ftempl=ftempl,
                                               match_scale=False)

        # Test direction selectivity by tuning.
        print('  Testing direction selectivity...')
        ftempl = qc_dir + 'direction_tuning/{}.png'
        for u in recUA.iter_thru():
            u.test_DS(do_plot=do_plot, ftempl=ftempl)

        # Plot trial rate and direction selectivity summary plots.
        print('  Plotting summary figures (rates and DS)...')
        ftempl = qc_dir + 'rate_DS_summary/{}.png'
        test_units.rate_DS_summary(recUA, ftempl=ftempl)

        # Test stability of recording session across tasks for selected units only.
        fname_stability = qc_dir + recording + '_recording_stability.png'
        test_units.check_recording_stability(recUA, fname_stability)


        # Exclude units of low quality or no direction selectivity.
        if exc_units:
            print('  Excluding units...')
            exclude = []
            for u in recUA.iter_thru():
                to_excl = test_sorting.test_rejection(u)
                u.set_excluded(to_excl)
                exclude.append(to_excl)

            # Report unit exclusion results.
            n_tot = len(exclude)
            n_exc, n_inc = sum(exclude), sum(np.invert(exclude))
            perc_exc, perc_inc = 100 * n_exc / n_tot, 100 * n_inc / n_tot
            rep_str = '  {} / {} ({:.1f}%) units {} analysis.'
            print(rep_str.format(n_exc, n_tot, perc_exc, 'excluded from'))
            print(rep_str.format(n_inc, n_tot, perc_inc, 'included into'))

        UA.add_recording(recUA)

    # Add index to unit names.
    UA.index_units()

    # Save selected Units with quality metrics and direction selectivity.
    ts = util.timestamp()
    n_units = UA.n_units()
    fname = util.format_to_fname(ua_name)
    data_dir = 'data/combined_recordings/{}_n{}_{}/'.format(fname, n_units, ts)
    util.write_objects({'UnitArr': UA}, data_dir + fname + '.data')

    # Write out unit list and save parameter plot.
    UA.save_params_table(data_dir + 'unit_list.xlsx')
    UA.plot_params(data_dir + 'unit_params.png')



## Test
#TPLCell = TPLCells[0]
#u = unit.Unit(TPLCell, constants.t_start, constants.t_stop,
#              kernels, constants.step, constants.tr_params)