#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:50:39 2016

Definition of constants relating to task and analysis.

@author: David Samu
"""

import pandas as pd
from quantities import ms, deg, cm

from seal.util import util


# %% Task constants.

# Stimulus parameters.
stim_params = pd.DataFrame({('S1', 'Dir'): ('markS1Dir', deg),
                            ('S2', 'Dir'): ('markS2Dir', deg),
                            ('S1', 'LocX'): ('markS1LocX', cm),
                            ('S1', 'LocY'): ('markS1LocY', cm),
                            ('S2', 'LocX'): ('MarkS2LocX', cm),
                            ('S2', 'LocY'): ('MarkS2LocY', cm),
                            ('S1', 'Rng'): ('markS1range', deg),
                            ('S2', 'Rng'): ('markS2range', deg)},
                            index=('name', 'dim')).T

# Subject answer reports.
answ_params = pd.DataFrame({('AnswCorr'): ('subjectAnswer', None)},
                           index=('name', 'dim')).T

# Stimulus durations.
S1_dur = 500*ms  # these are fixed at the moment
S2_dur = 500*ms

# All 8 directions.
all_dirs = util.quantity_linspace(0*deg, 315*deg, 8)


# %% Neurophysiological constants.

# Region-specific latency values (stimulus response delay).
latency = pd.Series({'MT': 50*ms, 'PFC': 100*ms})


# %% Relative timing of different trial events and periods.

# Trial events are defined relative to the anchor events coming from Tempo,
# ['S1 on', 'S1 off', 'S2 on', 'S2 off'].
# The relative timing of these anchor events can change from trial to trial.

tr_evt = [  # Basic task events.
          ('fixate', ('S1 on', -1000*ms)),
          ('S1 on', ('S1 on', 0*ms)),
          ('S1 off', ('S1 off', 0*ms)),
          ('S2 on', ('S2 on', 0*ms)),
          ('S2 off', ('S2 off', 0*ms)),
          ('saccade', ('S2 off', 1000*ms)),

          # Delay sub-period limits.
          ('1/3 delay', ('S1 off', 500*ms)),
          ('2/3 delay', ('S2 on', -500*ms)),

          # Cue-related events.
          ('no cue', ('S1 off', 750*ms)),  # latest time without cue on
          ('cue', ('S2 on', -750*ms))]

tr_evt = pd.DataFrame.from_items(tr_evt, ['rel to', 'shift'], 'index')

# Trial periods are defined relative to trial events (the exact timing of which
# are relative themselves to the anchor events, see above).

tr_prd_names = [('whole trial', ('fixate', 'saccade')),

                # Basic task periods.
                ('fixation', ('fixate', 'S1 on')),
                ('S1', ('S1 on', 'S1 off')),
                ('delay', ('S1 off', 'S2 on')),
                ('S2', ('S2 on', 'S2 off')),
                ('post-S2', ('S2 off', 'saccade')),

                # Extended stimulus periods.
                ('around S1', ('fixate', 'no cue')),
                ('around S2', ('cue', 'saccade')),

                # Delay sub-periods.
                ('early delay', ('S1 off', '1/3 delay')),
                ('late delay', ('2/3 delay', 'S2 on'))]

tr_prd = pd.DataFrame.from_items(tr_prd_names, ['start', 'stop'], 'index')


# %% Analysis constants.

# Kernel sets for firing rate estimation.
R100_kernel = util.kernel_set(['R100'])
RG_kernels = util.kernel_set(['G20', 'G40', 'R100', 'R200'])
shrtR_kernels = util.kernel_set(['R50', 'R75', 'R100'])
lngR_kernels = util.kernel_set(['R100', 'R200', 'R500'])

step = 10*ms

# Default rate name (to be used when rate type is not specified).
def_nrate = 'R100'
