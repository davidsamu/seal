#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:50:39 2016

Definition of constants relating to task and analysis.

@author: David Samu
"""

import numpy as np
from quantities import ms, deg, cm

from elephant.kernels import GaussianKernel, RectangularKernel

from seal.object.periods import Periods


# %% Task constants.

# Define parameter info.
pinfo = dict(markS1Dir=('S1Dir', deg), markS2Dir=('S2Dir', deg),
             markS1LocX=('S1LocX', cm), markS1LocY=('S1LocY', cm),
             MarkS2LocX=('S2LocX', cm), MarkS2LocY=('S2LocY', cm),
             markS1range=('S1Rng', deg), markS2range=('S2Rng', deg),
             subjectAnswer=('AnswCorr', None))


# Trial start and stop times.
t_start = -1000*ms
t_stop = 4000*ms


# %% Constants related to different trial periods.

tr_prds = Periods([('Whole trial', [-1000*ms, 4000*ms]),
                   ('Fixation',    [-1000*ms,    0*ms]),
                   ('S1',          [    0*ms,  500*ms]),
                   ('Delay',       [  500*ms, 2000*ms]),
                   ('S2',          [ 2000*ms, 2500*ms]),
                   ('Post-S2',     [ 2500*ms, 4000*ms])])

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

def get_rec_kernel_sigma(width):
    return width.rescale(ms)/2/np.sqrt(3)

# Kernels for firing rate estimation.
kernels = {'G20': GaussianKernel(sigma=20*ms),
           'G40': GaussianKernel(sigma=40*ms),
           'R50': RectangularKernel(sigma=get_rec_kernel_sigma(50*ms)),
           'R100': RectangularKernel(sigma=get_rec_kernel_sigma(100*ms))}

kernels = {'R100': RectangularKernel(sigma=get_rec_kernel_sigma(100*ms)),
           'R200': RectangularKernel(sigma=get_rec_kernel_sigma(200*ms)),
           'R300': RectangularKernel(sigma=get_rec_kernel_sigma(300*ms))}

step = 10*ms
