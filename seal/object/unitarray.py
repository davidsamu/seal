#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 17:14:25 2016

Class representing an array of units.

@author: David Samu
"""


import pandas as pd
from collections import OrderedDict as OrdDict

from seal.object import unit
from seal.util import plot, util


class UnitArray:
    """
    Generic class to store a 2D array of units (neurons or groups of neurons),
    by channel (rows) and task/experiment (columns).
    """

    # %% Constructor.
    def __init__(self, name, Unit_list, task_order=None):
        """Create UnitArray instance from list of units."""

        # Init instance.
        self.Name = name
        self.Units = pd.DataFrame()

        # Fill Units array with unit list provided.
        # Get available tasks, if task_order not provided.
        if not task_order:
            task_order = sorted(set([u.SessParams['experiment']
                                     for u in Unit_list]))

        # Add units to UnitArray in task order
        # (determining column order of unit table).
        for task in task_order:
            units = [u for u in Unit_list
                     if u.SessParams['experiment'] == task]
            self.add_task(task, units)

    # %% Utility methods.

    def tasks(self):
        """Return task names."""

        task_names = self.Units.columns
        return task_names

    def n_tasks(self):
        """Return number of tasks."""

        nsess = len(self.tasks())
        return nsess

    def rec_ch_unit_indices(self, req_tasks=None):
        """
        Return (recording, channel #, unit #) index triples of units
        with data available across required tasks (optional).
        """

        ch_unit_idxs = self.Units.index.to_series()

        # Select only channel unit indices with all required tasks available.
        if req_tasks is not None:
            idx = [len(self.unit_list(req_tasks, [cui])) == len(req_tasks)
                   for cui in ch_unit_idxs]
            ch_unit_idxs = ch_unit_idxs[idx]

        return ch_unit_idxs

    def rec_ch_unit_task_indices(self, req_tasks=None):
        """
        Return (recording, channel #, unit #, task) index quadruples of units
        with data available across required tasks (optional).
        """

        ch_un_idxs = self.rec_ch_unit_indices(req_tasks)
        uidxs = [u.get_rec_ch_un_task_index() for cu in ch_un_idxs
                 for u in self.unit_list(req_tasks, [cu])]
        names = ['rec', 'ch', 'unit', 'task']
        uidxs = pd.MultiIndex.from_tuples(uidxs, names=names).to_series()

        return uidxs

    def n_units(self):
        """Return number of units (number of rows of UnitArray)."""

        nunits = len(self.rec_ch_unit_indices())
        return nunits

    def recordings(self):
        """Return list of recordings in Pandas Index object."""

        ch_units = self.rec_ch_unit_indices()
        recordings = ch_units.index.get_level_values('rec')
        unique_recordings = list(OrdDict.fromkeys(recordings))
        return unique_recordings

    def n_recordings(self):
        """Return number of recordings."""

        n_recordings = len(self.recordings())
        return n_recordings

    def init_task_ch_unit(self, tasks=None, ch_unit_idxs=None):
        """Init tasks and channels/units to query."""

        # Default tasks and units: all tasks/units.
        if tasks is None:
            tasks = self.tasks()
        if ch_unit_idxs is None:
            ch_unit_idxs = self.rec_ch_unit_indices()

        return tasks, ch_unit_idxs

    # %% Methods to add units.

    def add_task(self, task_name, task_units):
        """Add new task data as extra column to Units table of UnitArray."""

        # Concatenate new task as last column.
        # This ensures that channels and units are consistent across
        # tasks (along rows) by inserting extra null units where necessary.
        idxs = [(u.SessParams['monkey'] + '_' + util.date_to_str(u.SessParams['date']),
                 u.SessParams['channel #'], u.SessParams['unit #'])
                for u in task_units]
        names = ['rec', 'ch', 'unit']
        multi_idx = pd.MultiIndex.from_tuples(idxs, names=names)
        task_df = pd.DataFrame(task_units, columns=[task_name], index=multi_idx)
        self.Units = pd.concat([self.Units, task_df], axis=1, join='outer')

        # Replace missing (nan) values with empty Unit objects.
        self.Units = self.Units.fillna(unit.Unit())

    # %% Methods to query units.

    def unit_list(self, tasks=None, ch_unit_idxs=None, return_empty=False):
        """Return units in a list."""

        tasks, ch_unit_idxs = self.init_task_ch_unit(tasks, ch_unit_idxs)
        ch_unit_set = set(ch_unit_idxs)

        # Put selected units from selected tasks into a list.
        unit_list = [r for row in self.Units[tasks].itertuples()
                     for r in row[1:]
                     if row[0] in ch_unit_set]

        # Exclude empty units.
        if not return_empty:
            unit_list = [u for u in unit_list if not u.is_empty()]

        return unit_list

    # %% Exporting and reporting methods.

    def unit_params(self):
        """Return unit parameters as Pandas table."""

        unit_params = [u.get_unit_params() for u in self.unit_list()]
        unit_params = pd.DataFrame(unit_params, columns=unit_params[0].keys())
        return unit_params

    def save_params_table(self, fname):
        """Save unit parameters as Excel table."""

        writer = pd.ExcelWriter(fname)
        unit_params = self.unit_params()
        util.write_table(unit_params, writer)

    def plot_params(self, ffig):
        """Plot group level histogram of unit parameters."""

        unit_params = self.unit_params()
        plot.group_params(unit_params, ffig=ffig)
