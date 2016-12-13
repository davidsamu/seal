#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:55:34 2016

@author: upf
"""

# Template script to
# 1) import SimpleTPLCell structures (created in Matlab),
# 2) convert them into Seal Units and UnitArrays,
# 3) test unit quality, recording drifts, stimulus response properties, and
# 4) automatically exclude trials and units.

# To get it work, you need to:
#   - install all necessary Python packages
#       - see 'install_requires' field in https://github.com/davidsamu/seal/blob/master/setup.py
#   - pull from GitHub, and import, the latest version of Seal (see "Import Seal" below),
#   - set your paths:
#       - working directory (my_wd),
#       - folder with .mat TPLCell objects (tpl_dir), with optional subfolders (sub_dirs)
#       - output folder for Seal Units (seal_dir and data_dir)

# Preprocessing analysis is done in two steps:
# 1) init.convert_TPL_to_Seal: this simply converts the .mat objects into Seal objects, without any analysis
# 2) init.run_preprocessing: using the converted Seal objects, runs preprocessing analyses, plots results and excludes trials/units


# %% Init
import os
import sys

# Import Seal.

# Pull Seal from GitHub either by
# a) command line: "git clone https://github.com/davidsamu/seal", OR
# b) using a GUI tool, e.g. GitKraken

# Set this to your Seal installation path
sys.path.insert(1, '/home/upf/Research/tools/python/Seal/')

from seal.io import init
from seal.util import util
from seal.plot import putil, pplot, prate, ptuning
from seal.analysis import decoding, roc, tuning
from seal.object import constants, periods, trials, unit

# Set working directory
my_wd = '/home/upf/Research/projects/PFC inactivation/'
os.chdir(my_wd)


# %% Convert all Matlab structs to Seal UnitArray.

# Input/output folders.
tpl_dir = 'data/matlab_cells/'     # folder containig MATLAB TPLCell objects per recording
seal_dir = 'data/seal_units/'      # folder to write seal units into
sub_dirs = ['normal', 'saline', 'excluded']   # subfolders, set to [''] if there are no subfolders

# Kernel set to be used for firing rate estimation.
# See constants.py for available options.
kernels = constants.R100_kernel

init.convert_TPL_to_Seal(tpl_dir, seal_dir, sub_dirs, kernels)


# %% Run quality test on Units.

data_dir = 'data/seal_units/normal/'  # folder with recording(s) to process
ua_name = 'PFC inactivation'          # name of UnitArray to be created

# Automatic trial and unit rejection.
rej_trials = True   # automatically reject trials?
exc_units = False   # automatically exclude units?

# Plotting parameters.
plot_QM = True       # plot quality metrics (SNR, ISI, rate, etc) of each unit?
plot_SR = True       # plot stimulus response of each unit? (3x3 figure)
plot_DS = True       # plot direction selectivity/tuning plot of each unit?
plot_sum = True      # plot summary plot of each unit? (all trials + tuning + pref/anti trials)
plot_stab = True     # plot recording stability plot of each recording?
init.run_preprocessing(data_dir, ua_name, rej_trials, exc_units,
                       plot_QM, plot_SR, plot_DS, plot_sum, plot_stab)
