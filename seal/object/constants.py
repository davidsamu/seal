#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:50:39 2016

Definition of constants relating to task and analysis.

@author: David Samu
"""

from quantities import ms, deg, cm

from seal.util import util
from seal.object.periods import Periods


# %% Task constants.

# Define parameter info.
tr_params = dict(markS1Dir=('S1Dir', deg), markS2Dir=('S2Dir', deg),
                 markS1LocX=('S1LocX', cm), markS1LocY=('S1LocY', cm),
                 MarkS2LocX=('S2LocX', cm), MarkS2LocY=('S2LocY', cm),
                 markS1range=('S1Rng', deg), markS2range=('S2Rng', deg),
                 subjectAnswer=('AnswCorr', None))

# Trial start and stop times.
t_start = -1000*ms
t_stop = 3500*ms


# %% Constants related to different trial periods.

tr_prds = Periods([('Whole trial', [-1000*ms, 3500*ms]),
                   ('Fixation',    [-1000*ms,    0*ms]),
                   ('S1',          [    0*ms,  500*ms]),
                   ('Delay',       [  500*ms, 2000*ms]),
                   ('S2',          [ 2000*ms, 2500*ms]),
                   ('Post-S2',     [ 2500*ms, 3500*ms])])

# Stimulus periods.
stim_prds = Periods(tr_prds.periods(['S1', 'S2']))

# Extended stimulus periods.
ext_stim_prds = Periods([('S1', [ -1000*ms, 1900*ms]),
                         ('S2', [  1900*ms, 3500*ms])])

# Delay sub-periods.
delay_prds = Periods([('early',  [  500*ms, 1000*ms]),
                      ('middle', [ 1000*ms, 1500*ms]),
                      ('late',   [ 1500*ms, 2000*ms])])

# Delay sub-periods.
narrow_delay_prds = Periods([('early',  [  500*ms, 700*ms]),
                             ('middle', [ 1150*ms, 1350*ms]),
                             ('late',   [ 1800*ms, 2000*ms])])

# Delay sub-periods.
narrow_fixation_prds = Periods([('early',  [-1000*ms, -800*ms]),
                                ('middle', [ -600*ms, -400*ms]),
                                ('late',   [ -200*ms,    0*ms])])

# Stimulus response delay in MT.
MT_stim_resp_delay = 50*ms

# Delayed trial and stimulus periods.
what_delayed = ['S1', 'S2']
where_delayed = ['start']
del_tr_prds = Periods(tr_prds.delay_periods(MT_stim_resp_delay,
                                            what_delayed, where_delayed))
del_stim_prds = Periods(stim_prds.delay_periods(MT_stim_resp_delay,
                                                what_delayed, where_delayed))


# %% Analysis constants.

# Kernel sets for firing rate estimation.
R100_kernel = util.kernel_set(['R100'])
RG_kernels = util.kernel_set(['G20', 'G40', 'R100', 'R200'])
shrtR_kernels = util.kernel_set(['R50', 'R75', 'R100'])
lngR_kernels = util.kernel_set(['R100', 'R200', 'R500'])

step = 10*ms

# Default rate name (to be used/preferred when rate type is specified).
def_nrate = 'R100'


# %% Direction related constants.

all_dirs = util.quantity_linspace(0*deg, 315*deg, 8)
