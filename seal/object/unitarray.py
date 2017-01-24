#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class representing an array of units.

@author: David Samu
"""


import pandas as pd

from seal.object import unit


# Constants.
idx_names = ['rec', 'ch', 'unit']


class UnitArray:
    """
    Generic class to store a 2D array of units (neurons or groups of neurons),
    by recording/channel/unit index (rows) and task (columns).
    """

    # %% Constructor.
    def __init__(self, name, task_order=None):
        """
        Create UnitArray empty instance.

        To fill it with data, use add_task and add_recording methods.
        """

        # Init instance.
        self.Name = name
        self.Units = pd.DataFrame(columns=task_order)

    # %% Utility methods.

    def init_tasks_uids(self, tasks=None, uids=None):
        """Init tasks and recordings/channels/units to query."""

        # Default tasks and units: all tasks/units.
        if tasks is None:
            tasks = self.tasks()
        if uids is None:
            uids = self.uids()

        if isinstance(uids, pd.Series):
            uids = uids.tolist()

        return tasks, uids

    # %% Iterator methods.

    # Usage: e.g. [u for u in UnitArray.iter_thru(args)]
    # to return units in list: list(UnitArray.iter_thru(args))

    def iter_thru(self, tasks=None, uids=None, miss=False, excl=False):
        """Custom iterator init over selected tasks and units."""

        # List of tasks and uids to iterate over.
        self._iter_tasks, self._iter_uids = self.init_tasks_uids(tasks, uids)
        self._iter_missing, self._iter_excl = miss, excl

        return self

    def __iter__(self):
        """Init iterator. Required to implement iterator class."""

        # Init task and unit IDs to point to first unit to be returned.
        self._itask, self._iuid = 0, 0

        return self

    def __next__(self):
        """Return next Unit."""

        # Terminate iteration if we have run out of units.
        if (self._itask >= len(self._iter_tasks) or
            self._iuid >= len(self._iter_uids)):
            raise StopIteration

        # Get current unit.
        task = self._iter_tasks[self._itask]
        uid = self._iter_uids[self._iuid]
        u = self.get_unit(uid, task)

        # Update (increment) index variables to point to next unit.
        self._iuid += 1
        if self._iuid >= len(self._iter_uids):  # switch to next task
            self._itask += 1
            self._iuid = 0

        # Let's not return empty and/or excluded unit if not requested.
        if ((u.is_empty() and not self._iter_missing) or
            (u.is_excluded() and not self._iter_excl)):
            return self.__next__()

        return u

    # %% Other methods to query and yield units.

    def get_unit(self, uid, task):
        """Return unit of given task and uid."""

        u = self.Units.loc[[uid], task][0]
        return u

    def get_unit_by_name(self, uname):
        """Return unit of specific name."""

        task, subj, date, elec, chun, isort = uname.split()
        uid = (subj+'_'+date, int(chun[2:4]), int(chun[-1]))
        u = self.get_unit(uid, task)

        return u

    def get_uids_of_rec(self, rec):
        """Return list of units from specific recording."""

        rec_uids = self.uids().loc[rec]
        return rec_uids

    # %% Reporting methods.

    def tasks(self):
        """Return task names."""

        task_names = self.Units.columns
        return task_names

    def n_tasks(self):
        """Return number of tasks."""

        nsess = len(self.tasks())
        return nsess

    def n_units_per_rec_task(self, recs=None, tasks=None):
        """
        Return all pairs of recording-task combinations (excluding tasks not
        performed on a given recording day or with all units excluded).
        """

        if recs is None:
            recs = self.recordings()
        if tasks is None:
            tasks = self.tasks()

        MI_rec_tasks = pd.MultiIndex.from_product([recs, tasks])
        nunits = pd.Series(0, index=MI_rec_tasks, dtype=int)
        for rec in recs:
            uids = list(self.get_uids_of_rec(rec))
            for task in tasks:
                nunits[rec, task] = len(list(self.iter_thru([task], uids)))

        return nunits

    def rec_task_order(self):
        """Return original recording task order."""

        tasks, recs = self.tasks(), self.recordings()
        task_order = pd.DataFrame(index=recs, columns=tasks)

        for rec in recs:
            uids = list(self.get_uids_of_rec(rec))
            for task in tasks:
                ulist = list(self.iter_thru([task], uids, excl=True))
                if not len(ulist):  # no non-empty unit in task recording
                    continue
                u = ulist[0]  # just take first unit (all should be the same)
                task_order.loc[rec, task] = u.SessParams.loc['task #'] - 1

        return task_order

    def uids(self, req_tasks=None):
        """
        Return unit IDs [(recording, channel #, unit #) index triples]
        for units with data available across all required tasks (optional).
        """

        uids = self.Units.index.to_series()

        # Select only channel unit indices with all required tasks available.
        if req_tasks is not None:
            idx = [len(list(self.iter_thru(req_tasks, [ui]))) == len(req_tasks)
                   for ui in uids]
            uids = uids[idx]

        return uids

    def utids(self, req_tasks=None):
        """
        Return unit&task IDs [(rec, channel #, unit #, task) index quadruples]
        for units with data available across required tasks (optional).
        """

        uids = self.uids(req_tasks)
        utids = [u.get_utid() for uid in uids
                 for u in list(self.iter_thru(req_tasks, [uid]))]
        names = idx_names + ['task']
        utids = pd.MultiIndex.from_tuples(utids, names=names).to_series()

        return utids

    def n_units(self):
        """Return number of units (# of rows of UnitArray)."""

        nunits = len(self.uids())
        return nunits

    def recordings(self):
        """Return list of recordings."""

        recordings = self.Units.index.get_level_values('rec').unique()
        return recordings

    def n_recordings(self):
        """Return number of recordings."""

        n_recordings = len(self.recordings())
        return n_recordings

    def unit_params(self):
        """Return unit parameters in DataFrame."""

        unit_params = [u.get_unit_params() for u in self.iter_thru()]
        unit_params = pd.DataFrame(unit_params, columns=unit_params[0].keys())
        return unit_params

    # %% Methods to add sets of units.

    def add_task(self, task_name, task_units):
        """Add new task data as extra column to Units table of UnitArray."""

        # Concatenate new task as last column.
        # This ensures that channels and units are consistent across
        # tasks (along rows) by inserting extra null units where necessary.
        uids = [u.get_uid() for u in task_units]
        midx = pd.MultiIndex.from_tuples(uids, names=idx_names)
        task_df = pd.DataFrame(task_units, columns=[task_name], index=midx)
        self.Units = pd.concat([self.Units, task_df], axis=1, join='outer')

        # Replace missing (nan) values with empty Unit objects.
        self.Units = self.Units.fillna(unit.Unit())

    def add_recording(self, UA):
        """Add Units from new recording UA to UnitArray as extra rows."""

        # Append new tasks to end.
        new_tasks = UA.tasks().difference(self.tasks())
        task_order = self.tasks().append(new_tasks)

        # Do concatenation.
        self.Units = pd.concat([self.Units, UA.Units], join='outer')

        # Reorder columns.
        self.Units = self.Units[task_order]

        # Reformat index.
        self.Units.index = pd.MultiIndex.from_tuples(self.Units.index,
                                                     names=idx_names)

        # Replace missing (nan) values with empty Unit objects.
        self.Units = self.Units.fillna(unit.Unit())

    # %% Methods to remove sets of units.

    def remove_unit(self, uid, task, clean_array=True):
        """Remove a single unit."""

        self.Units.loc[uid, task] = unit.Unit()

        if clean_array:
            self.clean_array()

    def remove_recording(self, rec, tasks=None, clean_array=True):
        """Remove a recording across all or selected tasks."""

        if tasks is None:
            tasks = self.tasks()

        for uid in self.get_uids_of_rec(rec):
            for task in tasks:
                self.remove_unit(uid, task, False)

        if clean_array:
            self.clean_array()

    def remove_task(self, task, recs=None, clean_array=True):
        """Remove a task across all or selected recordings."""

        if recs is None:
            recs = self.recordings()

        for rec in recs:
            for uid in self.get_uids_of_rec(rec):
                self.remove_unit(uid, task, False)

        if clean_array:
            self.clean_array()

    def clean_array(self, keep_excl=True):
        """Remove empty (and excluded) uids (rows) and tasks (columns)."""

        # Clean uids.
        empty_uids = [uid for uid in self.uids()
                      if not len(list(self.iter_thru(uids=[uid],
                                                     excl=keep_excl)))]
        if len(empty_uids):
            self.Units.drop(empty_uids, axis=0, inplace=True)

        # Clean tasks.
        empty_tasks = [task for task in self.tasks()
                       if not len(list(self.iter_thru(tasks=[task],
                                                      excl=keep_excl)))]
        if len(empty_tasks):
            self.Units.drop(empty_tasks, axis=1, inplace=True)

    # %% Methods to manipulate units.

    def index_units(self):
        """Add index to Units in UnitArray per task."""

        for task in self.tasks():
            for i, u in enumerate(self.iter_thru(tasks=[task], miss=True,
                                                 excl=True)):
                # Skip missing units here to keep consistency
                # of indexing across tasks.
                if not u.is_empty():
                    u.add_index_to_name(i+1)
