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
    by recording/channel/unit index (rows) and task (columns).
    """

    # %% Constructor.
    def __init__(self, name):
        """
        Create UnitArray empty instance.

        To fill it with data, use add_task and add_recording methods.
        """

        # Init instance.
        self.Name = name
        self.Units = pd.DataFrame()

    # %% Utility methods.

    def init_tasks_uids(self, tasks=None, uids=None):
        """Init tasks and recordings/channels/units to query."""

        # Default tasks and units: all tasks/units.
        if tasks is None:
            tasks = self.tasks()
        if uids is None:
            uids = self.uids()

        return tasks, uids

    # %% Iterator methods.

    # Usage: e.g. [u for u in UnitArray.iter_throu(kwargs)]

    def iter_thru(self, tasks=None, uids=None, ret_empty=False,
                  ret_excl=False):
        """Custom iterator init over selected tasks and units."""

        # List of tasks and uids to iterate over.
        self._iter_tasks, self._iter_uids = self.init_tasks_uids(tasks, uids)
        self._iter_empty, self._iter_excl = ret_empty, ret_excl

        return self

    def __iter__(self):
        """Init iterator. Required to implement iterator class."""

        # Init task and unit IDs to point to first unit to be returned.
        self._itask, self._iuid = 0, 0

        return self

    def __next__(self):
        """Return next Unit."""

        # Terminate iteration if we have run out of units.
        if self._itask >= len(self._iter_tasks):
            raise StopIteration

        # Get current unit.
        task = self._iter_tasks[self._itask]
        uid = self._iter_uids[self._iuid]
        u = self.Units.loc[uid, task]

        # Update (increment) index variables to point to next unit.
        self._iuid += 1
        if self._iuid >= len(self._iter_uids):  # switch to next task
            self._itask += 1
            self._iuid = 0

        # Let's not return empty and/or excluded unit if not requested.
        if ((u.is_empty() and not self._iter_empty) or
            (u.is_excluded() and not self._iter_excl)):
            return self.__next__()

        return u

    # %% Other methods to query units.

    def unit_list(self, tasks=None, uids=None, return_empty=False):
        """Return units in a list."""

        tasks, uids = self.init_tasks_uids(tasks, uids)

        # Put selected units from selected tasks into a list.
        unit_list = [u for row in self.Units[tasks].itertuples()
                     for u in row[1:] if row[0] in uids]

        # Exclude empty units.
        if not return_empty:
            unit_list = [u for u in unit_list if not u.is_empty()]

        return unit_list

    # %% Reporting methods.

    def tasks(self):
        """Return task names."""

        task_names = self.Units.columns
        return task_names

    def n_tasks(self):
        """Return number of tasks."""

        nsess = len(self.tasks())
        return nsess

    def uids(self, req_tasks=None):
        """
        Return unit IDs [(recording, channel #, unit #) index triples]
        for units with data available across all required tasks (optional).
        """

        uids = self.Units.index.to_series()

        # Select only channel unit indices with all required tasks available.
        if req_tasks is not None:
            idx = [len(self.unit_list(req_tasks, [cui])) == len(req_tasks)
                   for cui in uids]
            uids = uids[idx]

        return uids

    def utids(self, req_tasks=None):
        """
        Return unit&task IDs [(rec, channel #, unit #, task) index quadruples]
        for units with data available across required tasks (optional).
        """

        uids = self.uids(req_tasks)
        utids = [u.get_utid() for uid in uids
                 for u in self.unit_list(req_tasks, [uid])]
        names = ['rec', 'ch', 'unit', 'task']
        utids = pd.MultiIndex.from_tuples(utids, names=names).to_series()

        return utids

    def n_units(self):
        """Return number of units (# of rows of UnitArray)."""

        nunits = len(self.uids())
        return nunits

    def recordings(self):
        """Return list of recordings."""

        uids = self.uids()
        recordings = uids.index.get_level_values('rec')
        unique_recordings = list(OrdDict.fromkeys(recordings))
        return unique_recordings

    def n_recordings(self):
        """Return number of recordings."""

        n_recordings = len(self.recordings())
        return n_recordings

    # %% Methods to add units.

    def add_task(self, task_name, task_units):
        """Add new task data as extra column to Units table of UnitArray."""

        # Concatenate new task as last column.
        # This ensures that channels and units are consistent across
        # tasks (along rows) by inserting extra null units where necessary.
        uids = [u.get_uid() for u in task_units]
        midx = pd.MultiIndex.from_tuples(uids, names=['rec', 'ch', 'unit'])
        task_df = pd.DataFrame(task_units, columns=[task_name], index=midx)
        self.Units = pd.concat([self.Units, task_df], axis=1, join='outer')

        # Replace missing (nan) values with empty Unit objects.
        self.Units = self.Units.fillna(unit.Unit())

    def add_recording(self, UA):
        """Add Units from new recording UA to UnitArray as extra rows."""

        self.Units = pd.concat([self.Units, UA.Units], axis=0, join='outer')

    # %% Methods to manipulate units.

    def index_units(self):
        """Add index to Units in UnitArray per each task."""

        for task in self.tasks():
            for i, u in enumerate(self.iter_thru(tasks=[task],
                                                 ret_empty=True)):
                # skip empty units to keep consistency of indexing across tasks
                if not u.is_empty():
                    u.add_index_to_name(i+1)

    # %% Exporting methods.

    def unit_params(self):
        """Return unit parameters in DataFrame."""

        unit_params = [u.get_unit_params() for u in self.iter_thru()]
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
        plot.group_params(unit_params, self.Name, ffig=ffig)
