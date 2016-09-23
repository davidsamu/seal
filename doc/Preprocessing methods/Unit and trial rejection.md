# Unit and trial rejection

These steps are done after spike sorting and calculating the quality metrics and stimulus response properties of the units.


## Unit rejection

The goal is to remove units that do not show elementary properties to be meaningfully analysed. This is mostly the result of low quality data (e.g. strong, inseparable multi-unit activity), and therefore may be considered as a quality assurance step, rather then cherrypicking the nice neurons.

Criteria:

1. Insufficent activity: Overall session-average firing rate larger than 2 spikes / second.
2. Insufficent coverage of receptive field: **To be added.**
3. Insufficent stimulus response: **To be added.**



## Trial rejection

The goal is to detect electrode drifts and changes in the state of the unit. We do this by checking for strong drift, drop and jump in various quality metrics (see *Quality Metrics*).

Criteria:

1. Firing rate: **To be added.**
2. Waveform amplitude: **To be added.**
3. Waveform duration: **To be added.**
4. ISI violation: **To be added.**
