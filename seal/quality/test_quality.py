# -*- coding: utf-8 -*-

"""
Meta-function to plot sorting quality and stability plots for recordings.

@author: David Samu
"""

import os
import numpy as np

from seal.io import export
from seal.util import util
from seal.plot import putil
from seal.object import unitarray
from seal.quality import test_units, test_stability


def quality_control(rec_name, rec_data_dir, qc_dir, comb_data_dir,
                    plot_qm=True, plot_stab=True, fselection=None):
    """Run quality control (SNR, rate drift, ISI, etc) on given recording."""

    # Check that Seal folder with data exists.
    if not os.path.exists(rec_data_dir):
        print(rec_data_dir, 'not found.')
        return
    ftask_data = sorted(os.listdir(rec_data_dir))
    if not len(ftask_data):
        print(rec_data_dir, 'does not contain task data.')
        return

    # Determine task order.
    tasks = [util.params_from_fname(f).loc['taskidx'] for f in ftask_data]
    itask = [util.params_from_fname(f).loc['idx'] for f in ftask_data]
    task_order = np.argsort(itask)

    # Init combined UnitArray object.
    UA = unitarray.UnitArray(rec_name)

    putil.inline_off()

    # Import task recordings into combined UA.
    for i in task_order:
        ftask = rec_data_dir + ftask_data[i]
        tUA = util.read_objects(ftask, 'UnitArr')
        UA.add_task(tasks[i], list(tUA.iter_thru()))

    # Test unit quality, save result figures, add stats to units and
    # exclude low quality trials and units.
    ftempl = qc_dir + 'quality_metrics/{}.png'
    test_units.quality_test(UA, ftempl, plot_qm, fselection)

    # Report unit exclusion stats.
    fname = qc_dir + 'exclusion.log'
    test_units.report_unit_exclusion_stats(UA, fname)

    # Test stability of recording session across tasks.
    if plot_stab:
        fname = qc_dir + 'recording_stability.png'
        test_stability.rec_stability_test(UA, fname)

    # Add index to unit names.
    UA.index_units()

    # Save Units with quality metrics added.
    fname = comb_data_dir + 'all_recordings.data'
    util.write_objects({'UnitArr': UA}, fname)

    # Export unit and trial selection results.
    if fselection is None:
        fname = comb_data_dir + '/unit_trial_selection.xlsx'
        export.export_unit_trial_selection(UA, fname)

    # Export unit list.
    #export.export_unit_list(UA, comb_data_dir + 'unit_list.xlsx')

    # Re-enable inline plotting
    putil.inline_on()