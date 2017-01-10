# Preprocessing Standard Operating Procedures

## Steps of generating data files, quality metrics and basic figures for recordings.

1. Concatenate recording sessions (in PlexUtil).
2. Do spike sorting (in Plexon OfflineSorter).
3. Split sorted recording (in PlexUtil).
4. Generate TPLCell data structure (in Matlab).
5. Generate quality metrics (in Python).
6. Select units and trials (in Excel and Python).


## 1. Generate merged (concatenated) recording file.

- Start PlexUtil (shortcut is on Desktop).
- Open data folder of recording in left panel and select Plexon files to be merged. 
  - Make sure you only select raw recordings, and not e.g. previously merged or sorted files that may have accidentally been left in the folder during previous sorting!
- In right panel, arrange data files in order of recording time given by index following experiment name (see lab’s [file naming convention](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Naming%20conventions.md)), e.g. dd1, ddRem2, ddPas3. Use ‘Move Up’ and ‘Move Down’ buttons below panel.
  - Doublecheck that you selected all the Plexon files, and nothing else, and that they are in the right order.
- Press 'Merge' button in toolbar.
- For merge type, select second option: ‘Merge files consecutively in specified order (top to bottom)’, then press 'Next'.
- Set output folder and file name in next window.
  - Folder should be: ..\Sorted\Merged\ (see [folder structure conventions](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Data%20folder%20structure%20conventions.md))
  - File name should be: *SSS_MMDDYYee_mrg.plx*, e.g. 201_071516v1_mrg.plx (see [file naming convention](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Naming%20conventions.md)).
- Press 'Next'.
- Press 'Finish' to generate merged Plexon file.

## 2. Do spike sorting.

- Start OfflineSorter (shortcut is on Desktop, the one without any version number).
- Open the merged file (output of above section).
- Do spike sorting on all channels (1 – 16). See [SOPs on spike sorting](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Spike%20sorting%20SOPs.md).
- When finished with all channels, save sorted data to the same folder. You can accept the file name suggested by OfflineSorter.
  - To save disk space, do not keep multiple versions of the merged and sorted data file. Delete any previous and intermediate versions that is not needed any more (such as the unsorted one create in the previous step).

## 3. Split sorted recording.

- Start PlexUtil (shortcut is on Desktop).
- Select sorted file (output of above section) and press 'Split' button in toolbar.
- Select ‘Split into Multiple PLX Files’, then ‘All Channels’, then ‘By Frames’.
- Set output folder and file name in next window.
  - Folder should be: ..\Split\ (see [folder structure conventions](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Data%20folder%20structure%20conventions.md))
  - File name should be: *SSS_MMDDYYee*, e.g. 201_071516v1 (see [file naming convention](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Naming%20conventions.md)), i.e., delete '_mrg_0X' from end of offered file name
- Click 'Finish'.

## 4. Generate TPLCell data structure.

- Use Matlab script [generate_TPLCells.m](https://github.com/davidsamu/seal/blob/master/doc/Preprocessing%20methods/Matlab%20preprocessing%20scripts/generate_TPLCells.m).
- In the second cell of the script, set parameters of recording(s) to be processed.
  - Multiple recordings can be processed at the same time.
- Run third cell to pair and rename split recording files. Check if it finished successfully.
  - If any error occurred, fix problem and run this cell again.
- Run fourth cell. This will take a while.

The rest of the preprocessing steps require running Seal's preprocessing script in Python.

## 5. Generate quality metric figures.

- Use [preprocessing.py](https://github.com/davidsamu/seal/blob/master/examples/preprocessing.py).
- Set paths to folders with recordings, as explained in the script's comments.
- Depending on the number of units and amount of spikes to process, it may take up to 20-40 mins per recording to generate quality metric figures.

## 6. Select units and trials.

- Using the figures generated in step 5, edit and save the unit/trial selection Excel file (also generated in step 5), as explained in [preprocessing.py](https://github.com/davidsamu/seal/blob/master/examples/preprocessing.py).
- To apply and save unit/trial selection, run corresponding section in [preprocessing.py](https://github.com/davidsamu/seal/blob/master/examples/preprocessing.py).
