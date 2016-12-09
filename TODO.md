TODO list of Seal project
-------------------------

- write unit tests

- go through master scripts, move functionality into Seal, collect TODOs

- calculate waveform duration in a way that deals with truncated WFs!


periods
-------
  - implement iterator methods


trials
------
  - make named Series of fields
  - distinguish between trials and list of trials in units and elsewhere


unit
----
  - rename 'empty' units to 'missing' units
  - add RF coverage information
  - seprate DS calculation from plotting (add plotting params as fields to Unit?)


test_sorting
------------
  - add RF coverage to unit selection(?)
  - add quality test results to unit for dynamic unit inclusion


test_units
----------
  - refactor DR_test and DS_summary by moving inner plotting into Unit
  - to all function: add option to show excluded units?
  - tuning plot scales to be matched after DS plotting separated in Unit
  - check_recording_stability: check task order, UA order is not recording order! 
  - check_recording_stability: add grand total slope


init
----
  - add option to exclude trials/units using user's excel file


plot
----
  - split into submodules (constants, raster&rate, DS, basic plots, different formatting, etc)
  - add functions to easily change plot style
  - consider changing each plot to seaborn API? Rate (FR DF)?
  - refactor group_params plotting
  - refactor rate, raster_rate, 
  - rewrite mean_tuning curve(?)
  - option to add background color to raster


plot.scatter
------------
  - add diagonal distribution to scatter
  - add side histogram to scatter (# http://seaborn.pydata.org/generated/seaborn.jointplot.html#seaborn.jointplot)
  - report and test significant values separately
  - add confidence interval to linear fit using seaborn (other seaborn API functionality?)

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

