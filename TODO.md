TODO list of Seal project
-------------------------


preprocessing pipeline
----------------------
  - try automatic sorting using OpenElectrophys
  - improve waveform duration estimation (e.g. by smoothing spline fit?)


analyses to add from DD scripts
-------------------------------
  - decoding
    - preprocessing
    - run
    - analysis
  - comparison effect


unitarray / unit
----------------
  - add monkey to UA (monkey, rec, ch, idx)
  - add test and register whether unit's are matched across tasks


permutation tests
-----------------
  - consider using Numba


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
