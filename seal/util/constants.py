#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of generic constants (not specific to any experiment/task).

@author: David Samu
"""


import numpy as np
import pandas as pd

from quantities import ms, deg, cm

from seal.util import kernels


# %% Stimulus constants.

# All presented directions.
all_dirs = np.linspace(0, 315, 8) * deg

# Stimulus durations. These are fixed, fundamental constants for now.
stim_dur = pd.Series({'S1': 500*ms, 'S2': 500*ms})

# Delay length(s). Predefined, actual lengths are rounded to these values.
del_lens = pd.Series([1500*ms, 2000*ms])

# Stimulus parameters.
stim_params = pd.DataFrame({('S1', 'Dir'): ('markS1Dir', deg, int),
                            ('S2', 'Dir'): ('markS2Dir', deg, int),
                            ('S1', 'LocX'): ('markS1LocX', cm, None),
                            ('S1', 'LocY'): ('markS1LocY', cm, None),
                            ('S2', 'LocX'): ('MarkS2LocX', cm, None),
                            ('S2', 'LocY'): ('MarkS2LocY', cm, None),
                            ('S1', 'Rng'): ('markS1range', deg, None),
                            ('S2', 'Rng'): ('markS2range', deg, None),
                            ('S1', 'Size'): ('StimSize', deg, None),
                            ('S2', 'Size'): ('StimSize', deg, None)},
                           index=('name', 'dim', 'type')).T

# Trial features specific to stimulus.
stim_feats = ('Dir', 'Loc', 'LocX', 'LocY', 'Rng', 'Size')


# %% Experiment constants.

def to_report(task):
    """Return name of feature to be reported."""

    if 'Pas' in task:
        return 'pas'
    elif task.startswith('dd'):
        return 'dd'
    elif task.startswith('rng'):
        return 'dd'
    elif task.startswith('loc'):
        return 'loc'
    else:
        print('Cannot find feature to report for task', task)
        return None


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
           ('base on', ('S1 on', -750*ms)),
           ('base off', ('S1 on', -250*ms))]

tr_evts = pd.DataFrame.from_items(tr_evts, ['rel to', 'shift'], 'index')


# Trial periods are defined relative to trial events (the exact timing of which
# are relative themselves to the anchor events, see above).

# ***: times do not match across trials for variable delay lengths!

tr_prds = [('whole trial', ('fixate', 'saccade')),  # ***

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
           ('S1 half', ('fixate', 'S2 on')),  # ***
           ('S2 half', ('S2 on', 'saccade')),

           # Delay sub-periods.
           ('early delay', ('S1 off', '1/3 delay')),
           ('late delay', ('2/3 delay', 'S2 on')),

           # Baseline activity period.
           ('baseline', ('base on', 'base off')),

           # Cue related periods.
           ('S1 to cue', ('S1 off', 'cue')),  # ***
           ('cue to S2', ('cue', 'S2 on'))]

tr_prds = pd.DataFrame.from_items(tr_prds, ['start', 'stop'], 'index')


# Stimulus start and stop time relative to each event.
ev_stims = pd.DataFrame(columns=['stim', 'start', 'stop'])
for _ev, (_rel_to, _shift) in tr_evts.iterrows():
    _stim, _on_off = _rel_to.split()
    _stim_start = -_shift
    _stim_stop = -_shift + (1 if _on_off == 'on' else -1) * stim_dur[_stim]
    ev_stims.loc[_ev] = [_stim, _stim_start, _stim_stop]


# %% Periods to build across-trial raster and rate plots.

_S2_S1_lbl_shift = stim_dur['S1'] + del_lens.min()
_prd_lbls = ['stim', 'ref', 'lbl_shift', 'max_len']


# Classic two periods for delay length split plotting.
tr_half_prds = [('S1 half', ('S1', 'S1 on', 0*ms, 3500*ms)),
                ('S2 half', ('S2', 'S2 on', _S2_S1_lbl_shift, 1500*ms))]
tr_half_prds = pd.DataFrame.from_items(tr_half_prds, _prd_lbls, 'index')
# Add cue timing.
tr_half_prds['cue'] = [(tr_evts.loc['cue', 'shift']
                        if ref == tr_evts.loc['cue', 'rel to'] else None)
                       for ref in tr_half_prds.ref]


# Three periods consistent across different delay lengths.
tr_third_prds = [('extended S1', ('S1', 'S1 on', 0*ms, 2750*ms)),
                 ('cue to S2', ('S1', 'S2 on', _S2_S1_lbl_shift, 750*ms)),
                 ('S2 half', ('S2', 'S2 on', _S2_S1_lbl_shift, 1500*ms))]
tr_third_prds = pd.DataFrame.from_items(tr_third_prds, _prd_lbls, 'index')
# Add cue timing.
tr_third_prds['cue'] = [(tr_evts.loc['cue', 'shift']
                         if ref == tr_evts.loc['cue', 'rel to'] else None)
                        for ref in tr_third_prds.ref]


# %% Periods to group level analysis.

# To be handled with care! Stimulus timing is not guaranteed to follow this
# across all experiments, e.g. in Combined task! Effected periods are marked
# by ***!

fixed_tr_prds = [('whole trial', (-1000*ms, 3500*ms)),  # ***

                 # Basic task periods.
                 ('fixation', (-1000*ms, 0*ms)),
                 ('S1', (0*ms, 500*ms)),
                 ('delay', (500*ms, 2000*ms)),  # ***
                 ('S2', (2000*ms, 2500*ms)),    # ***
                 ('post-S2', (2500*ms, 3500*ms)),  # ***

                 # Trial halves.
                 ('S1 half', (-1000*ms, 2000*ms)),  # ***
                 ('S2 half', (2000*ms, 3500*ms)),   # ***

                 # Delay sub-periods.
                 ('early delay', (500*ms, 1000*ms)),
                 ('mid delay', (1000*ms, 1500*ms)),   # ***
                 ('late delay', (1500*ms, 2000*ms)),  # ***

                 # Baseline activity period.
                 ('baseline', (-750*ms, -250*ms))]

fixed_tr_prds = pd.DataFrame.from_items(fixed_tr_prds,
                                        ['start', 'stop'], 'index')

classic_tr_prds = [('fixation', ('S1 on',)),
                   ('S1', ('S1 on',)),
                   ('delay', ('S1 on',)),
                   ('S2', ('S2 on',)),
                   ('post-S2', ('S2 on',))]
classic_tr_prds = pd.DataFrame.from_items(classic_tr_prds,
                                          ['ref_ev'], 'index')

fixed_tr_len = 4500*ms


# %% Constants related to firing rate estimation.

# Kernel set to be used.
kset = kernels.RG_kernels


# %% Neurophysiological constants.

nphy_cons = pd.DataFrame.from_items([('MT', (50*ms, 500*ms)),
                                     ('PFC', (100*ms, 200*ms)),
                                     ('MT/PFC', (50*ms, 200*ms))],
                                    ['latency', 'DSwindow'], 'index')
