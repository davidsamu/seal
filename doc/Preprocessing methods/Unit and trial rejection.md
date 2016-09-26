# Unit and trial rejection

These steps are done after [spike sorting](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Spike%20sorting%20SOPs.md) and calculating the [quality metrics](https://github.com/davidsamu/seal/blob/master/doc/Preprocessing%20methods/Quality%20metrics.md) and [stimulus response properties](https://github.com/davidsamu/seal/blob/master/doc/Preprocessing%20methods/Direction%20selectivity.md) of the units. See also [SOP](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Preprocessing%20SOPs.md) on whole process.


## Unit rejection

The goal is to remove units that do not show elementary properties to be meaningfully analysed. This is mostly the result of low quality data (e.g. strong, inseparable multi-unit activity), and therefore may be considered as a necesary quality assurance step, rather then cherrypicking the nice neurons.

The metrics referred below are calculated for the duration of the entire session. Temporal changes in them are check during trial rejection below.

Criteria:

1. Extremely low waveform consistency: SNR < 0.5.
2. Insufficient unit activity: Firing rate < 1 spikes / second.
3. Extreme ISI violation ratio: ISI v.r. > 2%.
3. Insufficient coverage of receptive field: **To be added.**
4. Insufficient stimulus response: **To be added.**



## Trial rejection

The goal is to detect electrode drifts and changes in the state of the unit. We do this by checking for strong drifts, drops and jumps in firing rate (see [Quality Metrics](https://github.com/davidsamu/seal/blob/master/doc/Preprocessing%20methods/Quality%20metrics.md)). 

More specifically, the algorithm selects the period of the session that

1. has at most a factor of 2 between the lowest and the highest firing rate values within the period, and
2. contains the highest number of trials.

Trials outside of this period are excluded.
