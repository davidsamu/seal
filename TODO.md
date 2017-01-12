TODO list of Seal project
-------------------------


preprocessing pipeline
----------------------
  - try automatic sorting using OpenElectrophys


analyses to add from DD scripts
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

constants
---------
  - define a constants/parameter file per project


unitarray / unit
----------------
  - add test and register whether unit's are matched across tasks


permutation tests
-----------------
  - consider using Numba


init
----
  - add meta function to create collage figure, in order to drop requirement of ImageMagick


spikes and rates
----------------
  - returning excluded trials should be made impossible in some way!


waveform duration and amplitude calculation
-------------------------------------------
- calculate waveform duration in a way that deals with truncated WFs


export
------
  - add export decoding data (from Anna's script) into pickle and mat


unit
----
  - add RF coverage information
  - add type information: suppressive?, broad spiking? motor? etc


plot
----
  - update colors and other theme elements after upgrading to matplotlib 2.0


plot.scatter
------------
  - add diagonal distribution to scatter
  - report and test significant values separately


util
----
  - add non-parametric AND non-paired test!


decoding
--------
  - extend with error trial analysis
