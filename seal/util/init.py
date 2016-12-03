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




## Test
#TPLCell = TPLCells[0]
#u = unit.Unit(TPLCell, constants.t_start, constants.t_stop,
#              kernels, constants.step, constants.tr_params)