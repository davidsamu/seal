#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 17:14:25 2016

Class representing an array of units.

@author: David Samu
"""

import numpy as np
import pandas as pd

from seal.object import unit
from seal.util import plot, util


class UnitArray:
    """
    Generic class to store a 2D array of units (neurons or groups of neurons),
    by channel (rows) and session/experiment (columns).
    """

    # %% Constructor.
    def __init__(self, name, session_dict):
        """
        Create UnitArray instance from dictionary of
        session name - list of Units key-value pairs.
        """

        # Init instance.
        self.Name = name
        self.Units = pd.DataFrame()

        # Fill Units array with session data provided.
        [self.add_session(s_name, s_units)
         for s_name, s_units in session_dict.items()]

    # %% Utility methods.
    def get_n_units(self):
        """Return number of units."""

        nunits = len(self.Units.index)
        return nunits

    def get_n_sessions(self):
        """Return number of sessions."""

        nsess = len(self.Units.columns)
        return nsess

    def get_rec_chan_unit_indices(self):
        """Return (recording, channel, unit) index triples."""

        chan_unit_idxs = self.Units.index.to_series()
        return chan_unit_idxs

    def get_sessions(self):
        """Return session names."""

        session_names = self.Units.columns
        return session_names

    def add_session(self, session_name, session_units):
        """Add new session data as extra column to Units table of UnitArray."""

        # Concatenate new session as last column.
        # This ensures that channels and units are consistent across
        # sessions (along rows) by inserting extra null units where necessary.
        idxs = [(u.SessParams['monkey'] + '_' + util.date_to_str(u.SessParams['date']),
                 u.SessParams['channel #'], u.SessParams['unit #'])
                for u in session_units]
        names = ['rec', 'chan # ', 'unit #']
        multi_idx = pd.MultiIndex.from_tuples(idxs, names=names)
        session_df = pd.DataFrame(session_units, columns=[session_name],
                                  index=multi_idx)
        self.Units = pd.concat([self.Units, session_df], axis=1, join='outer')

        # Replace missing (nan) values with empty Unit objects.
        self.Units = self.Units.fillna(unit.Unit())

    def get_unit_list(self, sessions=None, chan_unit_idxs=None,
                      return_empty=False):
        """Returns units in a list."""

        if sessions is None:
            sessions = self.get_sessions()
        if chan_unit_idxs is None:
            chan_unit_idxs = self.get_rec_chan_unit_indices()

        # Put selected units from selected sessions into a list.
        unit_list = [r for row in self.Units[sessions].itertuples()
                     for r in row[1:]
                     if row[0] in chan_unit_idxs]

        # Exclude empty units.
        if not return_empty:
            unit_list = [u for u in unit_list if not u.is_empty()]

        return unit_list

    # %% Exporting and reporting methods.
    def get_unit_params(self):
        """Return unit parameters as Pandas table."""

        unit_params = [u.get_unit_params() for u in self.get_unit_list()]
        unit_params = pd.DataFrame(unit_params, columns=unit_params[0].keys())
        return unit_params

    def save_params_table(self, fname):
        """Save unit parameters as Excel table."""

        writer = pd.ExcelWriter(fname)
        unit_params = self.get_unit_params()
        util.write_table(unit_params, writer)

    def plot_params(self, ffig):
        """Plot group level histogram of unit parameters."""

        unit_params = self.get_unit_params()
        plot.group_params(unit_params, ffig=ffig)
