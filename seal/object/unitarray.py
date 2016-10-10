#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 17:14:25 2016

Class representing an array of units.

@author: David Samu
"""

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

    def get_unit_indices(self):
        """Return (channel, unit) indices."""

        chan_unit_idxs = self.Units.index.tolist()
        return chan_unit_idxs

    def get_sessions(self):
        """Return session names."""

        session_names = self.Units.columns
        return session_names

    def add_session(self, session_name, session_units):
        """Add new session data as extra column to Units table of UnitArray."""

        # Concatenate new session as last column.
        # This ensures  that channels and units are consistent across
        # sessions (along rows) by inserting extra null units where necessary.
        index = [(u.SessParams['channel #'], u.SessParams['unit #'])
                 for u in session_units]
        session_df = pd.DataFrame(session_units, columns=[session_name],
                                  index=index)
        self.Units = pd.concat([self.Units, session_df], axis=1, join='outer')

        # Replace missing (nan) values with empty Unit objects.
        self.Units = self.Units.fillna(unit.Unit())

    # %% Exporting and reporting methods.
    def get_unit_params(self):
        """Return unit parameters as Pandas table."""

        unit_params = [u.get_unit_params()
                       for idx, row in self.Units.iterrows()
                       for u in row]
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
