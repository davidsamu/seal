# Unit and trial rejection

These steps are done after [spike sorting](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Spike%20sorting%20SOPs.md) and calculating an plotting the [quality metrics](https://github.com/davidsamu/seal/blob/master/doc/Preprocessing%20methods/Quality%20metrics.md) and [stimulus response properties](https://github.com/davidsamu/seal/blob/master/doc/Preprocessing%20methods/Direction%20selectivity.md) of the units. See [SOP](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Preprocessing%20SOPs.md) on whole process.


## Trial rejection

The goal is to detect electrode drifts and changes in the state of the unit. We do this by checking for strong drifts, drops and jumps in firing rate (see [Quality Metrics](https://github.com/davidsamu/seal/blob/master/doc/Preprocessing%20methods/Quality%20metrics.md)). 

More specifically, the algorithm automatically selects the period of the session that

1. has at most a factor of 2 between the lowest and the highest firing rate values within the period, and
2. contains the highest number of trials.

Trials outside of this period are excluded. Quality metrics are calculated only on those waveforms, spike times and trials that are kept after this automatic trial rejection. 

Included/excluded trials can later be modified manually by editing the unit selection Excel file exported by the script (see [SOP](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Preprocessing%20SOPs.md)).


## Unit rejection

The goal is to remove units that do not show elementary properties to be meaningfully analysed. This is mostly the result of low quality recording, e.g. strong drifts in firing rate or inseparable, noisy multi-unit activity.

The metrics referred below are calculated for the duration of the entire task, but excluding the trials rejected above. Temporal changes in some of them are checked and taken into account during trial rejection above.

Every unit has to meet all the following criteria:

1. Waveform consistency: SNR > 1.
2. Unit activity: Firing rate > 1 spikes / second.
3. ISI violation ratio: ISI v.r. < 1%.
4. Number of remaining trials: # trials (after trial rejection) > 50% of total # of trials.
5. Stimulus response: DSI > 0.1 for both stimuli.
6. Coverage of receptive field: **To be added.**

The specific threshold values for each criterion can be changed in the "Constants" section on the top of seal/quality/test_sorting.py.
