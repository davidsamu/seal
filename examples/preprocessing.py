#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:55:34 2016

@author: David Samu
"""

# Template script to
#
# 1) import SimpleTPLCell structures (created in Matlab),
# 2) convert them into Seal Units and UnitArrays,
# 3) test unit quality and recording drifts,
# 4) exclude trials and units automatically and/or manually,
# 5) plot some simple activity plots per unit.

# To get the code run, you need to:
#   - install all necessary Python packages
#       - see 'install_requires' field in
#         https://github.com/davidsamu/seal/blob/master/setup.py
#   - pull from GitHub, and import, the latest version of Seal
#     (see section "Import Seal" below),
#   - set your paths:
#       - project name, folder and data dir (proj_name, proj_dir and data_dir)

# Preprocessing analysis is done in three steps:
# 1) init.convert_TPL_to_Seal: this simply converts the .mat objects
#    into Seal objects, without any analysis
# 2) init.run_preprocessing: using the converted Seal objects,
#    runs preprocessing analyses, plots results and exports automatic
#    trial and unit selection results
# 3) OPTIONAL: init.select_unit_and_trials: update selected units and trials
#    using Excel table modified manually by user (based on figures generated
#    in step 2)

# An annotated example of the composite figures generated during the second
# step can be found here:
# https://github.com/davidsamu/seal/blob/master/doc/Preprocessing%20methods/preprocessing_example.svg


# %% Init
import os
import sys

# Import Seal.

# Pull Seal from GitHub either by
# a) command line: "git clone https://github.com/davidsamu/seal", OR
# b) using a GUI tool, e.g. GitKraken

# Set this to your Seal installation path.
seal_installation_folder = '/home/upf/Research/tools/python/Seal/'
sys.path.insert(1, seal_installation_folder)

from seal.io import init


# %% Set up project paths.

# Set project parameters.
proj_name = 'Combined'
proj_dir = '/home/upf/Research/projects/' + proj_name
data_dir = proj_dir + '/data/'

os.chdir(proj_dir)


# %% Convert TPLCells to Seal UnitArrays, per recording.

init.convert_TPL_to_Seal(data_dir)

# Output:
# - Seal UnitArray data per recording in
#     data/recordings/[recording]/SealCells


# %% Run quality test on units, per recording.

plot_qm = True    # plot quality metrics (SNR, ISI, rate, etc) of each unit?
plot_stab = True  # plot recording stability plot of each recording?
init.quality_control(data_dir, proj_name, plot_qm, plot_stab)


# Output:
#
# - quality control figures for each unit in each recording in
#    data/recordings/[recording]/quality_control/
# - recording stability figure of each recording in
#    data/recordings/[recording]/recording_stability.png
# - combined UnitArray data containing all recordings in
#    data/all_recordings.data
# - unit and trial exclusion table in
#    data/unit_trial_selection.xlsx
# - unit parameter table
#    data/unit_list.xlsx


# %% Exclude low quality units and trials.

# Before running this section:
# 1. Check QM plot of each unit (generated in section above)
# 2. Based on QM plots, edit unit and trial selection table of combined
#    UnitArray if necessary (table is saved into folder data_dir).
#    - unit included: 1: include unit, 0: excluded unit
#    - first included trial [inclusive]: anything less or equal to 1 includes
#      all trials from beginning of task
#    - last included trial [inclusive]: anything larger or equal to # trials
#      (or -1, as a shorthand) includes all trials till end of task
# 3. After editing, save table into the same folder under a different name,
#    and set 'f_selection' to that file.

fselection = data_dir + 'unit_trial_selection_modified.xlsx'

plot_QM = True   # Replot quality metrics (SNR, ISI, rate, etc) of each unit?
init.run_quality_control(data_dir, proj_name, plot_QM, fselection)


# Output:
# - UnitArray data of all recordings, updated with unit & trial selection in
#    data/all_recordings.data
# - updated quality control figures per unit in
#    data/recordings/[recording]/quality_control/
# - updated recording stability figure of each recording in
#    data/recordings/[recording]/recording_stability.png
# - updated unit parameter table
#    data/unit_list.xlsx


# %% Plot basic activity plots.

# Plotting options.
plot_DR = True        # plot direction response of each unit? (3x3 figure)
plot_sel = True       # plot feature selectivity plot of each unit?

creat_montage = False  # create montage from all activity figures created
                       # need to have ImageMagick install for this option.
                       # https://www.imagemagick.org/script/binary-releases.php

init.unit_activity(data_dir, proj_name, plot_DR, plot_sel, creat_montage)

# Output:

