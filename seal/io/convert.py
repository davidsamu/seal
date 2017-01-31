"""
Functions related to converting TPLCell data into Seal data.

@author: David Samu
"""

import os

import numpy as np

from seal.util import util
from seal.object import unit, unitarray


def convert_TPL_to_Seal(data_dir, task_info, task_constants):
    """Convert TPLCells to Seal objects in recording folder."""

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
            params = [(TPLCell, task, task_info.loc[task], task_consts, kset)
                      for TPLCell in TPLCells]
            tUnits = util.run_in_pool(unit.Unit, params)

            # Add them to unit list of recording, combining all tasks.
            UA.add_task(task, tUnits)

        # Save Units.
        fname_seal = seal_dir + recording + '.data'
        util.write_objects({'UnitArr': UA}, fname_seal)
