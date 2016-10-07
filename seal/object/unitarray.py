#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 17:14:25 2016

Class representing an array of units.

@author: David Samu
"""

import pandas as pd

from seal.util import plot, util


class UnitArray:
    """
    Generic class to store a 2D array of units (neurons or groups of neurons),
    by channel (rows) and session/experiment (columns).
    """

    # %% Constructor
    def __init__(self, name=None, unit_array=None):
        """Create UnitArray instance from array of Unit objects."""

        # Create instance.
        self.Name = name
        self.Units = unit_array

    # %% Utility methods.
    def get_n_channels(self):
        """Return number of channels."""

        nchan = len(self.Units.index)
        return nchan

    def get_n_sessions(self):
        """Return number of sessions."""

        nsess = len(self.Units.columns)
        return nsess

    def get_sessions(self):
        """Return session names."""

        return self.Units.columns

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
