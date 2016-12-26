TODO list of Seal project
-------------------------

- write unit tests

- go through master scripts, move functionality into Seal, collect TODOs

- write Python script to automatically move and rename files after splitting sorted recording files



waveform duration calculation
-----------------------------
- calculate waveform duration in a way that deals with truncated WFs
- exclude WFs from excluded trials from mean WF calculation


export
------
  - add export decoding data (from Anna's script) into pickle and mat
  - add export unit list function

unit
----
  - add RF coverage information
  - "region" parameter should come from TPLCell (either file name, or unique data field)


test_sorting
------------
  - add RF coverage to unit selection(?)
  - make trial selection consider additional criteria (magnitude and stability of FR, SNR, etc)
  - trial selection: weight max FR deviation factor by FR magnitude? lower FRs may have higher deviations that are still acceptable, than higher FRs


init
----
  - finish exporting cell list table


plot
----
  - refactor get_gs_subplots
  - refactor subplot passing to plotting functions all around
  - update colors and other theme elements after upgrading to matplotlib 2.0


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
