TODO list of Seal project
-------------------------


preprocessing pipeline
----------------------
  - add recording location to session table
  - try automatic sorting using OpenElectrophys or some Matlab solutions
  - improve waveform duration estimation (e.g. by smoothing spline fit?)
  - task-responsiveness test should be done on trial groups sorted by stimulus feature and on all trials combined
  - latency calculation: timing and direction (by method of Zaksas 2006), per direction/location and to all combined
  - separate QC metrics based on recording location, e.g. base rate


analyses to add from DD scripts
-------------------------------
  - decoding
    - preprocessing
    - run
    - analysis
  - comparison effect


unitarray / unit
----------------
  - add test and register whether unit's are matched across tasks


permutation tests
-----------------
  - consider using Numba
  - identify p-value based on false positive rate during fixation


spikes and rates
----------------
  - returning excluded trials should be made impossible in some way!


unit
----
  - add RF coverage information
  - add type information: suppressive?, broad spiking? motor? etc
  - test task-relatedness by stimulus type


plot
----
  - update colors and other theme elements after upgrading to matplotlib 2.0



decoding
--------
  - extend with error trial analysis
  - choice probability
  
