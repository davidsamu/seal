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

# Define parameter info.
tr_params = [('markS1Dir', ('S1Dir', deg)), ('markS2Dir', ('S2Dir', deg)),
             ('markS1LocX', ('S1LocX', cm)), ('markS1LocY', ('S1LocY', cm)),
             ('MarkS2LocX', ('S2LocX', cm)), ('MarkS2LocY', ('S2LocY', cm)),
             ('markS1range', ('S1Rng', deg)), ('markS2range', ('S2Rng', deg)),
             ('subjectAnswer', ('AnswCorr', None))]
tr_params = pd.DataFrame.from_items(tr_params, ['seal_name', 'dimension'],
                                    'index')

# Trial start and stop times.
t_start = -1000*ms
t_stop = 4000*ms  # this should give enough time for longer delay (+500ms)

S1_len = 500*ms  # these are fixed at the moment
S2_len = 500*ms

# All 8 directions.
all_dirs = util.quantity_linspace(0*deg, 315*deg, 8)


# %% Neurophysiological constants.

# Region-specific latency values (stimulus response delay).
latency = pd.Series({'MT': 50*ms, 'PFC': 100*ms})


# %% Relative timing of different trial events and periods.

# Trial events are defined relative to the anchor events coming from Tempo, and
# stored in Unit.Events ['S1 onset', 'S1 offset', 'S2 onset', 'S2 offset'].
# The relative timing of these anchor events can change from trial to trial.

tr_evt = [  # Basic task events.
          ('fixation start', ('S1 onset', -1000*ms)),
          ('S1 onset', ('S1 onset', 0*ms)),
          ('S1 offset', ('S1 offset', 0*ms)),
          ('S2 onset', ('S2 onset', 0*ms)),
          ('S2 offset', ('S2 offset', 0*ms)),
          ('saccade', ('S2 offset', 1000*ms)),

          # Delay sub-period limits.
          ('1/3 delay', ('S1 offset', 500*ms)),
          ('2/3 delay', ('S2 onset', -500*ms)),

          # Cue-related events.
          ('no cue', ('S1 offset', 750*ms)),  # latest time without cue on
          ('cue', ('S2 onset', -750*ms))]

tr_evt = pd.DataFrame.from_items(tr_evt, ['rel to', 'offset'], 'index')

# Trial periods are defined relative to trial events (the exact timing of which
# are relative themselves to the anchor events, see above).

tr_prd_names = [('whole trial', ('fixation start', 'saccade')),

                # Basic task periods.
                ('fixation', ('fixation start', 'S1 onset')),
                ('S1', ('S1 onset', 'S1 offset')),
                ('delay', ('S1 offset', 'S2 onset')),
                ('S2', ('S2 onset', 'S2 offset')),
                ('post-S2', ('S2 offset', 'saccade')),

                # Extended stimulus periods.
                ('around S1', ('fixation start', 'no cue')),
                ('around S2', ('cue', 'saccade')),

                # Delay sub-periods.
                ('early delay', ('S1 offset', '1/3 delay')),
                ('late delay', ('2/3 delay', 'S2 onset'))]

tr_prd = [(prd, tr_evt.loc[ev1].append(tr_evt.loc[ev2]))
          for prd, (ev1, ev2) in tr_prd_names]
tr_prd = pd.DataFrame.from_items(tr_prd, ['start rel to', 'start offset',
                                          'stop rel to', 'stop offset'],
                                 orient='index')


# %% Analysis constants.

# Kernel sets for firing rate estimation.
R100_kernel = util.kernel_set(['R100'])
RG_kernels = util.kernel_set(['G20', 'G40', 'R100', 'R200'])
shrtR_kernels = util.kernel_set(['R50', 'R75', 'R100'])
lngR_kernels = util.kernel_set(['R100', 'R200', 'R500'])

step = 10*ms

# Default rate name (to be used when rate type is not specified).
def_nrate = 'R100'
