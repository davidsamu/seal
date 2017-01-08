#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of constants related to experiment, task and analysis.

@author: David Samu
"""

import pandas as pd
from quantities import ms, deg, cm

from seal.util import util, kernels


# %% Task constants and parameters.

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
answ_params = pd.Series({'AnswCorr': 'subjectAnswer'})

# Stimulus durations. These are fixed, fundamental constants for now.
stim_dur = pd.Series({'S1': 500*ms, 'S2': 500*ms})

# Delay length(s). Predefined, actual lengths are coarsed to these values.
delay_lengths = pd.Series([1500*ms, 2000*ms])

# All presented directions.
all_dirs = util.quantity_linspace(0*deg, 315*deg, 8)

# Task information.
# WARNING: Target value, if not None, overwrites any target feature information
# in TPLCell's data (define in TrialType column of TrialParams)!
# This is necessary for not combined tasks at the moment.
# Possible target values: 'loc', 'dir' or None.
task_info = [('com', ('PFC', None, False)),
             ('loc', ('PFC', 'loc', False)),
             ('dd1', ('PFC', 'dir', False)),
             ('dd2', ('PFC', 'dir', False)),
             ('comPas', ('PFC', None, True))]
cols = ['region', 'toreport', 'ispassive']
task_info = pd.DataFrame.from_items(task_info, cols, 'index')


# %% Relative timing of different trial events and periods.

# Trial events are defined relative to the anchor events coming from Tempo,
# ['S1 on', 'S1 off', 'S2 on', 'S2 off'].
# The relative timing of these anchor events can change from trial to trial.

tr_evts = [  # Basic task events.
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
           ('cue', ('S2 on', -750*ms)),

           # Baseline period limits.
           ('base on', ('S1 on', -700*ms)),
           ('base off', ('S1 on', -300*ms))]

tr_evts = pd.DataFrame.from_items(tr_evts, ['rel to', 'shift'], 'index')


# Trial periods are defined relative to trial events (the exact timing of which
# are relative themselves to the anchor events, see above).

# ***: times do not match across trials for variable delay lengths!

tr_prds = [('whole trial', ('fixate', 'saccade')),

           # Basic task periods.
           ('fixation', ('fixate', 'S1 on')),
           ('S1', ('S1 on', 'S1 off')),
           ('delay', ('S1 off', 'S2 on')),  # ***
           ('S2', ('S2 on', 'S2 off')),
           ('post-S2', ('S2 off', 'saccade')),

           # Extended stimulus periods.
           # Maximal consistent (stimulus matching) periods around stimuli.
           ('extended S1', ('fixate', 'no cue')),
           ('extended S2', ('cue', 'saccade')),

           # Trial halves.
           ('S1 half', ('fixate', 'S2 on ')),  # ***
           ('S2 half', ('S2 on', 'saccade')),

           # Delay sub-periods.
           ('early delay', ('S1 off', '1/3 delay')),
           ('late delay', ('2/3 delay', 'S2 on')),

           # Baseline activity period.
           ('baseline', ('base on', 'base off')),

           # Cue related periods.
           ('S1 to cue', ('S1', 'cue')),  # ***
           ('cue to S2', ('cue', 'S2 on'))]

tr_prds = pd.DataFrame.from_items(tr_prds, ['start', 'stop'], 'index')


# Stimulus start and stop time relative to each event.
ev_stims = pd.DataFrame(columns=['stim', 'start', 'stop'])
for ev, (rel_to, shift) in tr_evts.iterrows():
    stim, on_off = rel_to.split()
    stim_start = -shift
    stim_stop = -shift + (1 if on_off == 'on' else -1) * stim_dur[stim]
    ev_stims.loc[ev] = [stim, stim_start, stim_stop]


# Constituent time periods to build "full trial" raster and rate plots.
S2_S1_lbl_shift = stim_dur['S1'] + delay_lengths.min()
rr_prds = [('extended S1', ('S1', 'S1 on', 0*ms)),
           ('cue to S2', ('S1', 'S2 on', S2_S1_lbl_shift)),
           ('S2 half', ('S2', 'S2 on', S2_S1_lbl_shift))]

rr_prds = pd.DataFrame.from_items(rr_prds, ['stim', 'ref', 'lbl_shift'],
                                  'index')


# %% Analysis constants.

# Kernel sets for firing rate estimation.
R100_kernel = kernels.kernel_set(['R100'])
RG_kernels = kernels.kernel_set(['G20', 'G40', 'R100', 'R200'])
shrtR_kernels = kernels.kernel_set(['R50', 'R75', 'R100'])
lngR_kernels = kernels.kernel_set(['R100', 'R200', 'R500'])

step = 10*ms

# Kernel set to be used for project.
kset = R100_kernel

# Default rate name (to be used when rate type is not specified).
def_nrate = 'R100'


# %% Neurophysiological constants.

# Region-specific latency values (stimulus response delay).
latency = pd.Series({'MT': 50*ms, 'PFC': 100*ms})
