TODO list of Seal project
-------------------------

- write unit tests

- go through master scripts, move functionality into Seal, collect TODOs

- calculate waveform duration in a way that deals with truncated WFs!


trials
------
  - make named Series of fields
  - distinguish between trials and list of trials in units and elsewhere


unit
----
  - add RF coverage information
  - seprate DS calculation from plotting (add plotting params as fields to Unit?)


test_sorting
------------
  - add RF coverage to unit selection(?)
  - WF shape plotting: could for loop be vectorized for speed?
  - add autocorrelation function to QMs.


test_units
----------
  - refactor DR_test and DS_summary by moving inner plotting into Unit
  - to all function: add option to show excluded units?
  - tuning plot scales to be matched after DS plotting separated in Unit
  - check_recording_stability: add grand total slope


init
----
  - add option to exclude trials/units using user's excel file
  - finish exporting cell list table and params plot


plot
----
  - refactor group_params plotting, and get_gs_subplots in the process!
  - update colors and other theme elements after upgrading to matplotlib 2.0
  - move group params and unit info plots from putil
  - finish categorical plots


plot.scatter
------------
  - add diagonal distribution to scatter
  - report and test significant values separately

util
----
  - split into submodules
  - add non-parametric AND non-paired test!


decoding
--------
  - extend with error trial analysis


tuning
-----------
  - gaus: option to provide bounds and init values for fit
  - gaus: check if bounds and init values are consistent, change them otherwise
  - stats: calculate R2 and RMSE on all samples (rather than means to direction?)

