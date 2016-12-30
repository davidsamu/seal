TODO list of Seal project
-------------------------

- write unit tests

- go through master scripts, move functionality into Seal, collect TODOs

- write Python script to automatically move and rename files after splitting sorted recording files


spikes and rates
----------------
- returning excluded trials should be made impossible in some way!


waveform duration calculation
-----------------------------
- calculate waveform duration in a way that deals with truncated WFs
- exclude WFs from excluded trials from mean WF calculation


export
------
  - add export decoding data (from Anna's script) into pickle and mat

unit
----
  - add RF coverage information
  - "region" parameter should come from TPLCell (either file name, or unique data field)


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
