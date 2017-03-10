# Unit and trial rejection

These steps are done after [spike sorting](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Spike%20sorting%20SOPs.md) and calculating an plotting the [quality metrics](https://github.com/davidsamu/seal/blob/master/doc/Preprocessing%20methods/Quality%20metrics.md) of the units. See [SOPs](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Preprocessing%20SOPs.md) on the whole process.


## Trial rejection

The goal is to detect electrode drifts and changes in the state of the unit. We do this by checking for strong drifts, drops and jumps in firing rate (see [Quality Metrics](https://github.com/davidsamu/seal/blob/master/doc/Preprocessing%20methods/Quality%20metrics.md)). 

More specifically, the algorithm automatically selects the period of the session that

1. has at most a factor of 3 between the lowest and the highest firing rate values within the period, and
2. contains the highest number of trials.

Trials outside of this period are excluded. Quality metrics are calculated only on those waveforms, spike times and trials that are kept after trial rejection. 

Included/excluded trials can later be modified manually by editing the unit selection Excel file exported by Seal's preprocessing script (see [SOP](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Preprocessing%20SOPs.md)).


## Unit rejection

The goal is to remove units that do not show elementary properties necessary for analysis. This is mostly the result of issues with recording or spike sorting, e.g. strong drifts in firing rate, inseparable, noisy multi-unit activity or unaccaptably low firing rate.

The metrics below are calculated after excluding the strongly drifting trials (see above).

Any unit that does not pass any of the below criteria is excluded:

1. Waveform consistency: SNR > 1.
2. ISI violation ratio: ISI v.r. < 1%.
3. Total number of trials: total # trials > 20 (in case monkey quit at beginning of recording)
4. Number of remaining trials: # trials (after trial rejection) > 50% of total # of trials.
5. Sufficient activity: Mean firing rate > 5 sp/s for at least one direction during any part of the trial (S1, delay, S2 or post-S2).
6. Task-related activity: There exists a minimum 50ms period in any part of the trial (S1, delay, S2 or post-S2), when the firing rate for any of the different direction or location values of current or preceding stimulus is significantly different from baseline activity (between -700ms and -300ms during fixation). [Mann-Whitney (aka unpaired Wilcoxon) test with p < 0.001.]

The specific threshold values for each criterion can be changed in the "Constants" section on the top of seal/quality/test_sorting.py.
