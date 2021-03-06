# Quality metrics

These quality metrics are calculated after spike sorting and generating the Unit objects (see [SOPs](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Preprocessing%20SOPs.md)). For more details, see  [Hill et al., 2011](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3123734/) and references therein.


## Spike Sorting Quality Metrics


### Based on spike waveforms

#### Signal to noise (SNR)

SNR is calculated by dividing the standard deviation of the mean waveform by the total standard deviation of the residual waveforms (after subtracting the mean waveform from each of them). See [Hill et al., 2011](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3123734/).


#### Waveform amplitude

Waveform height, as measured from waveform onset (time of downward crossing of '0' level, aligned to 9th sample of waveform by Plexon).


#### Waveform duration

Time for waveform to reach maximal value, relative to waveform onset (waveform amplitude).
- Calculated during TPLCell creation using some spline model to increase precision of estimate.


### Based on spike times

All the below statistics are calculated over the entire session, as well as over the session using a non-overlapping 1 minute long sliding window (equal width and step of ~120 s, precise value may be slighly higher due to session length not being exact multiple of 120 s).


#### Firing rate

Number of spikes under time window divided by window width.
  - This a primary measure to detect electrode drifts and changes in the state of the unit, aiding trial rejection if necessary (see [Unit and trial rejection](https://github.com/davidsamu/seal/blob/master/doc/Preprocessing%20methods/Unit%20and%20trial%20rejection.md)).


#### ISI violation ratio (ISI v.r.)

Percentage of spike-pairs with inter-spike-interval (ISI) smaller than 1 ms. 
  - WARNING: This is **not** an estimate of the fraction of spikes coming from another neuron! We estimate that by True Spike Ratio (see below).


#### True Spikes Ratio (TSR)

Estimate of the percentage of spikes emitted by the single dominant neuron captured by the unit. The estimation fails (quadratic equation with both solutions being imaginary numbers) if there is no single dominant neuron (ISI violation ratio is too high, given the length of the recording). See [Hill et al., 2011](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3123734/).


## Stimulus Response Properties

### Task-related activity

Simply tests across all trials whether unit's activity deviates from baseline activity at any time during task.


### Receptive field coverage

Receptive fields of each channel are mapped at the beginning of the experiments.


### Direction selectivity

See [Direction selectivity](https://github.com/davidsamu/seal/blob/master/doc/Preprocessing%20methods/Direction%20selectivity.md).


## Unit characterisation

### Single versus multi unit

- Definite single unit: SNR > 3 and TRS = 100% (or, equivavlently, ISI v.r. = 0).
- Probable single unit: SNR > 2 and TRS > 90%.
- Multi unit: SNR < 2 or TRS < 90%.


### Narrow versus broad spiking

TBA


## References

Hill, N. H., Mehta, S. B., and Kleinfeld, D. (2011). *Quality metrics to accompany spike sorting of extracellular signals.* J Neurosci 31, 8699–8705. ([PubMed link](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3123734/))
