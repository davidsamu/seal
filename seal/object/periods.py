#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 12:52:16 2016

Class representing a list of trial periods.

@author: David Samu
"""


import pandas as pd
from collections import OrderedDict as OrdDict


class Periods:
    """Class representing a list of trial periods."""

    # %% Constructor
    def __init__(self, prds, col_names=['start', 'end']):
        """Create Unit instance, optionally from TPLCell data structure."""

        # If list is passed.
        if isinstance(prds, list):
            prds = pd.DataFrame(OrdDict(prds), index=col_names).T

        # Create basic data structure.
        self.prds = prds

    # %% Utility methods.

    def names(self):
        """Return stored period names."""

        names = self.prds.index
        return names

    def n_prds(self):
        """Return number of periods."""

        nprds = len(self.names())
        return nprds

    def periods(self, names=None):
        """Return start times."""

        if names is None:
            names = self.names()

        prds = self.prds.loc[names]
        return prds

    def start(self, names=None):
        """Return start times."""

        starts = self.periods(names)['start']
        return starts

    def end(self, names=None):
        """Return end times."""

        ends = self.periods(names)['end']
        return ends

    def dur(self, names=None):
        """Return durations."""

        dur = self.end(names) - self.start(names)
        return dur

    def prd_info_str(self, names=None, sep=' '):
        """Return formatted period information string in dictionary."""

        if names is None:
            names = self.names()

        prd_info_list = [(name, '{}{}({} - {})'.format(name, sep,
                                                       self.start(name),
                                                       self.end(name)))
                         for name in names]
        prd_info_dict = OrdDict(prd_info_list)

        return prd_info_dict

    # %% Mutating methods.
    def delay_periods(self, delay, which_list, where_list):
        """Return periods dataframe with delay stimulus periods."""

        prds = self.periods().copy()

        for which in which_list:
            for where in where_list:
                v = prds.get_value(which, where)
                prds.set_value(which, where, v+delay)

        return prds
