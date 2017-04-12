#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class representing an array of units.

@author: David Samu
"""


import warnings

import pandas as pd
from seal.object import unit
from seal.util import constants


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
        self.StabilityTest = pd.DataFrame()

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

    # %% Other methods to query units.

    def get_unit(self, uid, task):
        """Return unit of given task and uid."""

        u = self.Units.loc[[uid], task][0]
        return u

    def get_unit_by_utid(self, utid):
        """Return unit of given task and uid."""

        nuid = len(constants.uid_names)
        u = self.Units.loc[[utid[:nuid]], utid[nuid]][0]
        return u

    def get_unit_by_name(self, uname):
        """Return unit of specific name."""

        task, subj, date, elec, chun, isort = uname.split()
        uid = (subj, date, elec, int(chun[2:4]), int(chun[-1]))
        u = self.get_unit(uid, task)

        return u

    # %% Reporting methods.

    def tasks(self, with_idx=True):
        """Return task names."""

        task_names = self.Units.columns
        if not with_idx:  # remove task index
            task_names = task_names.str[:-1]

        return task_names

    def n_tasks(self):
        """Return number of tasks."""

        ntasks = len(self.tasks())
        return ntasks

    def get_param_vals(self, param):
        """Return all values of given uid parameter."""

        if not len(self.Units):
            return []

        if param not in self.Units.index.names:
            warnings.warn('Unknown UID parameter requested: ' + param)
            return []

        par_vals = self.Units.index.get_level_values(param).unique()

        return par_vals

    def get_all_param_val_combs(self, params=None, as_list=False):
        """Return all value combinations of given uid parameters."""

        if not len(self.Units):
            return []

        if params is None:
            params = constants.uid_names

        for param in params:
            if param not in self.Units.index.names:
                warnings.warn('Unknown UID parameter requested: ' + param)
                return []

        pval_combs = self.Units.reset_index().set_index(params).index
        if as_list:
            pval_combs = pval_combs.tolist()

        return pval_combs

    def get_unique_param_val_combs(self, params=None, as_list=True):
        """Return unique value combinations of given uid parameters."""

        pval_combs = self.get_all_param_val_combs(params, as_list=False)
        if not len(pval_combs):
            return pval_combs

        unique_pval_combs = pval_combs.unique()

        if as_list:
            unique_pval_combs = unique_pval_combs.tolist()

        return unique_pval_combs

    def recordings(self, params=None, as_list=True):
        """
        Return all recordings: (subject, date) pairs. Optionally, recordings
        can be defined by more/other parameters, e.g. subject, date, electrode.
        """

        if params is None:
            params = constants.rec_levels

        recs = self.get_unique_param_val_combs(params, as_list=as_list)
        return recs

    def subjects(self):
        """Return list of subjects."""

        subjects = self.get_param_vals('subj')
        return subjects

    def n_subjects(self):
        """Return number of subjects."""

        nsubj = len(self.subjects())
        return nsubj

    def dates(self):
        """Return list of dates."""

        dates = self.get_param_vals('date')
        return dates

    def electrodes(self):
        """Return list of electrodes."""

        elecs = self.get_param_vals('elec')
        return elecs

    def channels(self):
        """Return list of channels."""

        chs = self.get_param_vals('ch')
        return chs

    def uxs(self):
        """Return list of unit indexes."""

        uxs = self.get_param_vals('ux')
        return uxs

    def n_units(self):
        """Return number of units (# of rows of UnitArray)."""

        nunits = len(self.Units)
        return nunits

    def uids(self, recs=None, levels=None, drop_levels=False, as_series=False):
        """
        Return uids [(subj, date, elec, ch, ux) quintets] from given
        recordings [list of (subj, date) pairs].
        No check for missing or excluded units is performed!
        """

        if self.Units.empty:
            return []

        # Init.
        if levels is None:
            levels = constants.rec_levels
        if recs is None:
            recs = self.recordings(params=levels)

        # Select rows of recordings.
        irec_rows = self.get_all_param_val_combs(levels).isin(recs)

        # Get uids of rows.
        uids = self.Units.index[irec_rows]

        # Remove levels values not in index any more.
        uids = pd.MultiIndex.from_tuples(uids.tolist(), names=uids.names)

        if drop_levels:
            for level in levels:
                uids = uids.droplevel(level)

        # Convert to Series.
        if as_series:
            uids = pd.Series(list(uids), index=uids)

        return uids

    def utids(self, tasks=None, recs=None, levels=None,
              miss=False, excl=False, as_series=False):
        """
        Return utids [(subj, date, elec, ch, ux, task) sextets] for units with
        data available across given tasks and from given recordings (optional).
        """

        # Init.
        if tasks is None:
            tasks = self.tasks()

        # Get all uids of requested recordings.
        uids = self.uids(recs, levels, drop_levels=False)

        # Query utids.
        utids = [u.get_utid() for u in self.iter_thru(tasks, uids, miss, excl)]

        # Format as MultiIndex.
        utid_names = constants.utid_names
        if len(utids):
            utids = pd.MultiIndex.from_tuples(utids, names=utid_names)
        else:
            utids = pd.MultiIndex(levels=len(utid_names)*[[]],  # no units
                                  labels=len(utid_names)*[[]],
                                  names=utid_names)

        # Convert to Series.
        if as_series:
            utids = pd.Series(list(utids), index=utids)

        return utids

    def n_units_per_rec_task(self, recs=None, tasks=None, levels=None,
                             empty=False, excluded=False):
        """
        Return number of units for all pairs of recording-task combinations
        (excluding tasks not performed on a given recording day or with all
        units excluded).
        """

        if recs is None:
            recs = self.recordings(levels)
        if tasks is None:
            tasks = self.tasks()

        MI_rec_tasks = pd.MultiIndex.from_product([recs, tasks])
        nUnits = pd.Series(0, index=MI_rec_tasks, dtype=int)
        for rec in recs:
            for task in tasks:
                nUnits[rec, task] = len(self.utids([task], [rec], levels))

        return nUnits

    def rec_task_order(self, levels=None):
        """Return original recording task order."""

        tasks, recs = self.tasks(), self.recordings(levels)
        task_order = pd.DataFrame(index=pd.MultiIndex(recs), columns=tasks)

        for rec in recs:
            uids = list(self.uids([rec], levels))
            for task in tasks:
                ulist = list(self.iter_thru([task], uids, excl=True))
                if not len(ulist):  # no non-empty unit in task recording
                    continue
                u = ulist[0]  # just take first unit (all should be the same)
                task_order.loc[rec, task] = u.SessParams.loc['task #'] - 1

        return task_order

    # %% Methods to add sets of units.

    def add_task(self, task_name, task_units):
        """Add new task data as extra column to Units table of UnitArray."""

        if not len(task_units):
            return

        # Concatenate new task as last column.
        # This ensures that channels and units are consistent across
        # tasks (along rows) by inserting extra null units where necessary.
        uids = [u.get_uid() for u in task_units]
        midx = pd.MultiIndex.from_tuples(uids, names=constants.uid_names)
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
                                                     names=constants.uid_names)

        # Replace missing (nan) values with empty Unit objects.
        self.Units = self.Units.fillna(unit.Unit())

    # %% Methods to remove sets of units.

    def remove_unit(self, uid, task, clean_array=True):
        """Remove a single unit."""

        self.Units.loc[uid, task] = unit.Unit()

        if clean_array:
            self.clean_array()

    def remove_recording(self, rec, tasks=None, levels=None, clean_array=True):
        """Remove a recording across all or selected tasks."""

        if tasks is None:
            tasks = self.tasks()

        for uid in self.uids([rec], levels):
            for task in tasks:
                self.remove_unit(uid, task, False)

        if clean_array:
            self.clean_array()

    def remove_task(self, task, recs=None, levels=None, clean_array=True):
        """Remove a task across all or selected recordings."""

        if recs is None:
            recs = self.recordings(levels)

        for uid in self.uids(recs, levels):
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

    def unit_params(self):
        """Return unit parameters in DataFrame."""

        unit_params = [u.get_unit_params() for u in self.iter_thru()]
        unit_params = pd.DataFrame(unit_params, columns=unit_params[0].keys())
        return unit_params
