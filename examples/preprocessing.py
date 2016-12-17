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
# 4) exclude trials and units automatically and/or manually.

# To get it work, you need to:
#   - install all necessary Python packages
#       - see 'install_requires' field in https://github.com/davidsamu/seal/blob/master/setup.py
#   - pull from GitHub, and import, the latest version of Seal (see "Import Seal" below),
#   - set your paths:
#       - project folder (proj_dir)
#       - subfolder with .mat TPLCell objects (tpl_dir)
#       - subfolder into which to output Seal Units (seal_dir)

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

from seal.object import constants
from seal.io import init

# Set working directory
proj_dir = '/home/upf/Research/projects/Combined'  # DON'T add slash to end!
os.chdir(proj_dir)


# %% Convert all Matlab structs to Seal UnitArray.

# Input/output folders.
tpl_dir = 'data/matlab_cells/'     # folder containig MATLAB TPLCell objects per recording
seal_dir = 'data/seal_units/'      # folder to write seal units into

# Kernel set to be used for firing rate estimation,
# see constants.py for available options.
kernels = constants.R100_kernel

init.convert_TPL_to_Seal(tpl_dir, seal_dir, kernels)


# %% Run quality test on Units.

# Plotting parameters.
plot_QM = True       # plot quality metrics (SNR, ISI, rate, etc) of each unit?
plot_SR = True       # plot stimulus response of each unit? (3x3 figure)
plot_DS = True       # plot direction selectivity/tuning plot of each unit?
plot_sum = True      # plot summary plot of each unit? (all trials + tuning + pref/anti trials)
plot_stab = True     # plot recording stability plot of each recording?

ua_name = os.path.split(proj_dir)[1]
init.run_preprocessing(seal_dir, ua_name, plot_QM, plot_SR, plot_DS,
                       plot_sum, plot_stab)


# %% Exclude low quality units and trials.

# Before running this section:
# 1. Check results from previous section (QM and DS plots of each unit)
# 2. Based on these results, edit unit and trial selection table of combined
#    UnitArray if necessary (table is saved into 'combined_recordings' folder).
#    - unit included: 1: include unit, 0: excluded unit
#    - first included trial: anything less or equal to 1 includes all trials from beginning
#    - last included trial: anything larger or equal to # trials (or -1, as a shorthand) includes all trials till end
# 3. After editing, save table into the same folder under a different name,
#    and set 'f_sel_table' to that file.

# Folder with combined recordings and unit & trial selection table
# (both created in previous step).
data_dir = 'data/combined_recordings/'
f_sel_table = 'unit_trial_selection_modified.xslx'

init.exclude_unit_and_trials(data_dir, f_sel_table)
