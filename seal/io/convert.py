"""
Functions related to converting TPLCell data into Seal data.

@author: David Samu
"""

import os

import pandas as pd

from seal.util import util, constants
from seal.object import unit, unitarray


def task_TPL_to_Seal(f_tpl, f_seal, task, rec_info):
    """Convert TPLCell data to Seal data of single task."""

    # Load in Matlab structure (SimpleTPLCell).
    TPLCells = util.read_matlab_object(f_tpl, 'TPLStructs')

    # TPLCell data not iterable, i.e. empty (?).
    if not hasattr(TPLCells, '__iter__') or not len(TPLCells):
        print('Error: TPLCell data is empty in ', f_tpl)
        return

    # Create UnitArray (list of units) from TPLCell structures.
    kset = constants.kset
    params = [(TPLCell, rec_info, kset) for TPLCell in TPLCells]
    tUnits = util.run_in_pool(unit.Unit, params)

    # Add them to unit list of recording, combining all tasks.
    UA = unitarray.UnitArray(task)
    UA.add_task(task, tUnits)

    # Save Units.
    util.write_objects({'UnitArr': UA}, f_seal)


def rec_TPL_to_Seal(tpl_dir, seal_dir, rec_info, excl_tasks=[]):
    """Convert TPLCell data to Seal data in recording folder."""

    if not os.path.exists(tpl_dir):
        print('Error: Mssing TPLCell folder: ', tpl_dir)
        return

    # Query available TPLCell data files.
    f_tpl_cells = sorted([f for f in os.listdir(tpl_dir)
                          if f[-4:] == '.mat'])

    # Extract task names from file names.
    tasks = pd.Series(f_tpl_cells, name='f_tpl_cell')
    tasks.index = [util.params_from_fname(f_tpl).loc['task']
                   for f_tpl in f_tpl_cells]

    # Check that there's no duplication in task names.
    dupli = tasks.index.duplicated()
    if dupli.any():
        print('Error: Duplicated task names found: ' +
              ', '.join(tasks.index[dupli]))
        print('Please give unique names and rerun Seal Unit creation..')
        return

    # Exclude some tasks.
    to_include = [util.params_from_fname(f_tpl).loc['task'] not in excl_tasks
                  for f_tpl in f_tpl_cells]
    tasks = tasks[tasks.index[to_include]]

    if not len(tasks):
        print('Error: No TPLCell object found in ' + tpl_dir)
        return

    # Create units for each task.
    for task, f_tpl_cell in tasks.iteritems():
        print(' ', f_tpl_cell)
        f_tpl = tpl_dir + f_tpl_cell
        f_seal = seal_dir + f_tpl_cell[:-4] + '.data'
        task_TPL_to_Seal(f_tpl, f_seal, task, rec_info)
