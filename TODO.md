TODO list of Seal project
-------------------------

preprocessing pipeline
----------------------
  - write Python script to automatically move and rename files after splitting sorted recording files
  - try automatic sorting using OpenElectrophys


analyses to add from DD scirpts
-------------------------------
  - anticipatory delay activity
  - Fano factor
  - average tuning curve
  - tuning during task periods(?), or better way to measure encoding during delay?
  - decoding
    - preprocessing
    - run
    - analysis
  - comparison effect


permutation tests
-----------------
  - consider using Numba


init
----
  - add meta function to create collage figure, in order to drop requirement of ImageMagick


spikes and rates
----------------
  - returning excluded trials should be made impossible in some way!


trial rejection
---------------
 - recalculate unit stats (mean FR, WF duration, etc) with updated included trials


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
