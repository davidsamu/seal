# Spike Sorting 
# Standard Operating Procedures

## Before  starting

This scholarpedia page is good background reading before reading the SOPs:

http://www.scholarpedia.org/article/Spike_sorting


## Steps and approaches to spike sorting

- Go through the different combinations of PC1, PC2, PC3, “Non-linear energy”, and “Timestamps” axes and find the most “informative” axis-pairs (planes), i.e., the axes on which spikes are the most clustered into separate clouds (“clusters”), judging by eye.
- CHECK EVERY AXIS-PAIR! Clusters can appear in any of these pairs independently of the other dimensions! Ideally, you want to base your final decision about the number and position of clusters on the information contained in many planes.
- Select each cloud of points (normally just one, maybe two or three) that clump into (more or less) separate clusters (and have realistic waveforms) in the plane with the highest degree of separation (e.g. PC1 – PC2).
- If you see no separation in any of the planes (just a continuous cloud of points), select the plane with the highest spread, and delineate region(s) (mostly around the periphery or “tentacles” of the cloud) with spike-like waveforms.
- At this point, be inclusive. The strategy is to start by selecting a slightly excessive region(s), which we will improve by dropping subsets of spikes based on deviations
  - in the spike waveform (trimming waveforms), and 
  - from the center of the cloud on other planes (subtracting subregions).
- Use ISI violation ratio (ISI v.r., which is probably high, i.e. > 0.5%) to justify your trimming and subtractions (it should be decreased by each trimming and subtraction). 
  - You want to gain the largest decrease in ISI v.r. by removing the lowest number of spikes. In practice you will have to find a balance between the two, but keep in mind that ideally you want to subtract regions with small number of spikes (from the fringes of the selected spike cloud on a given plane) with noticeable ISI v.r. decrease. 
  - Do this in an exploratory manner, try to remove regions with different sizes from difference locations, and find the best ratio between ISI v.r. drop and number of spikes removed. With every cut, you want to remove the highest number of false positive spikes, while sacrificing the lowest number of true positive ones.
- As you go through the axis-pairs, try to improve the separation (judging by eye), the consistency of the waveforms (SNR) and ISI v.r. by adding/removing spikes to/from each cluster.
- To detect and remove waveforms, scan through each axis-pair (by moving the mouse while keeping the left mouse button pressed) and see how the shapes of the waveforms change in the waveform inspector window (top left).
  - General principle: choose spikes that have greater amplitude (“Nonlinear Energy” axis) and show a reasonably consistent waveform across time (higher SNR).
- Use “Timestamp” axis to check signal drift and change, especially in terms of firing rate (spike frequency)! This is very important, as heavily drifting (dropping and jumping) parts of the session will have to be removed during [further preprocessing](https://github.com/davidsamu/seal/blob/master/doc/Preprocessing%20methods/Unit%20and%20trial%20rejection.md). 
- When checking waveform consistency, remove outlier waveforms (trim). Do not overdo this! You want to base your trimming on the “ideal” waveform, AND justify the trimming by verifying that it decreases ISI v.r. 
In other words, just keeping the waveforms that look the best would be too subjective of an approach, not a valid, objective strategy! You need to combine trimming with an independent metric to measure improvement.
- While doing all the above, keep an eye on the number of selected spikes: spike rate should not be too low (< ~3 Hz), nor too high (> ~50 Hz), depending on cell type and brain area.
- Use ISI v.r. to split or trim cluster. It should be less than 1.0% (never more than 1.5%), ideally 0.0-0.2% for a very good unit, but this depends on the firing rate.
- Check for peak separation between clusters in Surface view.
- Check auto- and cross-correlograms. Open Cross-correlogram view from the toolbar. In the auto-correlograms (main diagonal), normally you should see a dip at time 0 (corresponding to the refractory period), then some increased values (corresponding to bursting activity or high firing rate periods), and then an attenuation to a steady value (proportional to the mean firing rate). If the cross-correlogram profiles of any two of the clusters look similar to this (off-diagonal plots), consider merging them. Watch out: structured cross-correlation can also arise between separate cells if they are functional connected or have similar stimulus response properties. Some of these cases are detectable in a "significant" drop in the merged cluster's SNR and/or increase in its ISI v.r. in comparison to those of the original clusters. ("Some" SNR drop and ISI v.r. is expected though.)
- Check number of spikes, and consider merging clusters with low number of spikes that close to each other, having similar waveforms and some cross-correlation structure (see above). If there is a "significant" increase in ISI v.r., the two original cluster of spikes probably come from different neurons.


## Guidelines & Rules of Thumb

- Altogether, you should spend ~2-5 minutes per channel, so between roughly between 20 and 60 mins total per recording.
- As a hard limit, the minimum number of spikes for an hour-long session should be 7200, meaning minimum 2 spikes/second.
- ISI v.r. should normally be much less than 1.5%, ideally 0.0-0.5%.
- It is possible, but rare that there is no (separable) unit in a channel. You should attempt to find at least some multi-unit activity even if there is no obvious separation on any of the planes. Start by searching for regions with spike-like waveforms at certain segments of the big cloud. On the other hand, do not be afraid of deleting a unit you could not get to a satisfactory level of quality.


## Tips

- You can do spike sorting much faster (and better) if you learn some of the keyboard shortcuts of the common operations.
  - add cluster: Alt + draw circle using cursor
  - subtract region: TBA
  - (TODO: add more shortcuts here!)
- You can turn on the feature to show ISI violations on cluster view (View → Cluster View Options → Show Short ISI Lines) to help find regions/spikes to remove.
  - WARNING: DON’T OVERUSE THIS FEATURE! You can easily “overfit” the criterion of minimsing ISI v.r. by removing just the problematic spikes, but this is would not result in a genuine removal of false positive spikes. ISI v.r. is just a statistical proxy metric we use to validate our selection of (peripheral) subregions to be removed from the cluster. Bottomline: do not rely ONLY on this feature, but use it as a guide in combination with other metrics (e.g. waveform shape and location on plane).
- Use Short Summary page for a more precise ISI value (4 digits) than what is displayed in the cluster list (only 1 digit).

