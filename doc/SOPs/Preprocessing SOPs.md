# Preprocessing Standard Operating Procedures

## Steps of generating data files, quality metrics and basic figures for recordings.

1. Concatenate recording sessions using PlexUtil.
2. Do spike sorting in OfflineSorter.
3. Split sorted recording using PlexUtil.
4. Generate TPLCell data structure.
5. Generate quality metrics and basic unit activity figures.


## 1. Concatenate recording sessions using PlexUtil.

- Open data folder in left panel and select files to be merged. Make sure you only select raw recordings, and not e.g. previously merged or sorted files that may have accidentally been left in the folder during previous sorting!
- In right panel, arrange data files in order of recording time given by index following experiment name (see lab’s [file naming convention](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Naming%20conventions.md)), e.g. dd1, ddRem2, ddPas3. Use ‘Move Up’ and ‘Move Down’ buttons below panel.
- Press 'Merge' button in toolbar.
- For merge type, select second option: ‘Merge files consecutively in specified order (top to bottom)’.
- Press 'Next'.
- Set output file directory and name folder on next window. Keeping the directory of the unsorted recording, change the name of the merged output file to *SSS_MMDDYYee_mrg.plx*, e.g. 201_071516v1_mrg.plx (see [file naming convention](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Naming%20conventions.md)).
- Press 'Next'.
- Press 'Finish' to generate merged Plexon file (_mrg.plx).

## 2. Do spike sorting in OfflineSorter.

- Open the merged file (output of above section).
- Do spike sorting on all channels (1 – 16). See [SOPs on spike sorting](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Spike%20sorting%20SOPs.md).
- When finished with all channels, save sorted data. File name is the same as described in section above, *SSS_MMDDYYee_mrg.plx*, e.g. 201_071516v1_mrg.plx (see [file naming convention](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Naming%20conventions.md)), but change output directory to: *Y:\yesbackup\ElectrophysiologyData\Plexon Sorted\SSS\SSS_MMDDYY\*, .e.g. *Y:\yesbackup\ElectrophysiologyData\Plexon\201\201_071516\*. If there are subfolders within *SSS*, select appropriate folder.

## 3. Split sorted recording using PlexUtil.

- Open Plexon Utility, select sorted file, press split button in toolbar.
- Select ‘Split into Multiple PLX Files’, then ‘All Channels’, then ‘By Frames’.
- After finished with splitting, rename each output file according to [naming convention](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Naming%20conventions.md), i.e., restore task name, session index and add sorting index.


## 4. Generate TPLCell data structure.

- Use Matlab script *generate_TPLCells.m*.
- In the script, set parameters of the recordings and tasks to be processed, as explained in the script's comments. Multiple recordings, each with multiple tasks, can be processed at the same time.
- For each recording, add to *session_list* all sorted task-specific Plexon files as generated in above.
- Run script.

## 5. Generate quality metrics and basic unit activity figures.

- Use [preprocessing.py](https://github.com/davidsamu/seal/blob/master/examples/preprocessing.py) from [examples](https://github.com/davidsamu/seal/tree/master/examples).
- Set paths to folders with recordings, as explained in the script's comments.
- Depending on the plotting settings you choose ('plot_X') and the number of units and spikes to process, this may run for up to 15-20 mins per recording, and generate various quality metric and stimulus response figures.

