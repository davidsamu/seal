# Preprocessing Standard Operating Procedures


**WARNING**: Sections from *Further preprocessing* and onwards are under development, and will change soon.

## TODO:

- Add lab’s file naming convention. Maybe put it into separate file?
- Update Further preprocessing part once steps have been worked out and code is in place.
- Add a figure about a few nice, acceptable and bad waveforms.


## Create spike sorted Plexon files.

### 1. Concatenate recording sessions using PlexUtil.

- Open data folder in left panel and select files to be merged. Make sure you only select raw recordings, and not e.g. previously merged or sorted files that may also be in the folder!
- In right panel, arrange data files in order of recording time given by index following experiment name (see lab’s file naming convention), e.g. dd1, ddRem2, ddPas3. Use ‘Move Up’ and ‘Move Down’ buttons below panel.
- Press Merge button in toolbar.
- For merge type, select ‘Merge files consecutively in specified order (top to bottom)’ (second option). 
- For output file, use 0 as final index for unsorted data and increment last index by one if necessary (0: unsorted, 1: sorted once, etc., see file naming convention).
- Press OK to generate merged Plexon file (_mrg.plx).

### 2. Do spike sorting in OfflineSorter.

- Open the merged file.
- Do spike sorting on all channels (1 – 16). See [SOPs on spike sorting](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Spike%20sorting%20SOPs.md).
- When finished with all channels, save sorted data. File name is Monkey#_DateElectrode_TaskSession#_Sorter#_mrg.plx, e.g. 201_071516v1_locMot1_1_mrg.plx.

### 3. Split sorted recording using PlexUtil.

- Open Plexon Utility, select sorted file, press split button in toolbar.
- Select ‘Split into Multiple PLX Files’, then ‘All Channels’, then ‘By Frames’.
- After splitting into original recording sessions, rename each output file according to naming convention (restore task name and session #, etc.).


## Further preprocessing

### 4. Generate TPLCell data structure.

- Use Matlab script extractTPLcell.m
- Use scan_task_files() to get task files from the selected folder .
- Use Expand_cell() to extract and expand cell components.
- Use mapD_load_trials_mixed() to fill trial-wise data fields.

### 5. Assess unit quality.

- Run drop_trials() to detect drift in recording and decide which trials to drop (by setting RejectTrials fields). Make sure that consecutive low activity trials from drifting units are excluded!
- Run plot_asssort() to see SNR and %true spike of each unit.

