#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:55:34 2016

@author: David Samu
"""

# Template script to
# 1) import SimpleTPLCell structures (created in Matlab),
# 2) convert them into Seal Units and UnitArrays,
# 3) test unit quality, recording drifts, stimulus response properties, and
# 4) exclude trials and units automatically and/or manually.

# To get the code run, you need to:
#   - install all necessary Python packages
#       - see 'install_requires' field in
#         https://github.com/davidsamu/seal/blob/master/setup.py
#   - pull from GitHub, and import, the latest version of Seal
#     (see section "Import Seal" below),
#   - set your paths:
#       - project folder (proj_dir)
#       - subfolder with .mat TPLCell objects (tpl_dir)
#       - subfolder into which to output Seal Units (seal_dir)
#   - set recording region name (region, "PFC" or "MT")

# Preprocessing analysis is done in three steps:
# 1) init.convert_TPL_to_Seal: this simply converts the .mat objects
#    into Seal objects, without any analysis
# 2) init.run_preprocessing: using the converted Seal objects,
#    runs preprocessing analyses, plots results and exports automatic
#    trial and unit selection results
# 3) OPTIONAL: init.select_unit_and_trials: update selected units and trials
#    using Excel table modified manually by user (based on figures generated
#    in step 2)



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

region = 'PFC'  # recording region: "PFC" or "MT"

# Kernel set to be used for firing rate estimation,
# see constants.py for available options.
kernels = constants.R100_kernel

init.convert_TPL_to_Seal(tpl_dir, seal_dir, kernels, region)

# Output:
# - Seal UnitArray data per recording inside 'seal_dir'


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

# Output:
# - quality control figures for each recording in
#   [seal_dir]/[recording]/qc_res
# - aggregate UnitArray data combined across all recordings in
#   combined_recordings/[project folder name]/[project folder name].data


# %% OPTIONAL: Update automatically excluded units and trials.

# Before running this section:
# 1. Check results from previous section (QM and DS plots of each unit)
# 2. Based on these results, edit unit and trial selection table of combined
#    UnitArray if necessary (table is saved into 'combined_recordings' folder).
#    - unit included: 1: include unit, 0: excluded unit
#    - first included trial [inclusive]: anything less or equal to 1 includes
#      all trials from beginning of task
#    - last included trial [inclusive]: anything larger or equal to # trials
#      (or -1, as a shorthand) includes all trials till end of task
# 3. After editing, save table into the same folder under a different name,
#    and set 'f_sel_table' to that file.

data_dir = 'data/all_recordings/'  # folder with combined recordings
f_data = data_dir + 'all_recordings.data'    # combined recordings data file
f_sel_table = data_dir + 'unit_trial_selection_modified.xslx'  # unit&trial selection table

init.select_unit_and_trials(f_data, f_sel_table)

# Output:
# - UnitArray data of all recordings, updated with unit&trial selection
#   in combined_recordings/[project folder name]/[project folder name].data
