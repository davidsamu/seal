# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 17:43:02 2016

Class for sets of trials and associated properties.

@author: David Samu
"""

import numpy as np


class Trials:
    """Class for sets of trials and associated properties."""

    # %% Constructor
    def __init__(self, trials=None, value=None, name=None):
        """Create a Trials instance."""

        # Create empty instance.
        self.trials = np.array(trials)
        self.value = value
        self.name = name if name is not None else str(value)

    # %% Utility methods.

    def num_trials(self):
        """Return number of trials."""

        num_trials = np.sum(self.trials)
        return num_trials

    def trial_indices(self):
        """Return indices of trials."""

        trial_indices = np.where(self.trials)[0]
        return trial_indices

    # %% Static methods.

    @staticmethod
    def combine_trials(lTrials, comb='or', name=None):
        """Creates a Trials object by combining Trials objects"""

        # Set up combining and naming functions.
        if comb == 'or':  # any of the trials
            comb_trials = lambda mat: np.any(mat, 0)
            comb_names = lambda names: ' or '.join(names)
        elif comb == 'and':  # all of the trials
            comb_trials = lambda mat: np.all(mat, 0)
            comb_names = lambda names: ' and '.join(names)
        elif comb == 'xor':  # exactly one of the trials
            comb_trials = lambda mat: mat.sum(0) == (mat.shape[0]-1)
            comb_names = lambda names: 'exactly one of ' + ' or '.join(names)
        elif comb == 'diff':  # only the first one of the trials
            comb_trials = lambda mat: np.logical_and(mat[0, ], mat[1:, ].sum(0) == 0)
            comb_names = lambda names: ' and '.join([names[0]] + ['not '+n for n in names[1:]])

        # Assamble parameters.
        trs = comb_trials(np.array([tr.trials for tr in lTrials]))
        vals = [tr.value for tr in lTrials]
        if name is None:
            name = comb_names([tr.name for tr in lTrials])

        # Create combined Trials object.
        trials = Trials(trs, vals, name)

        return trials
