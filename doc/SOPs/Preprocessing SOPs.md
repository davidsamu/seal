# Preprocessing Standard Operating Procedures

## Steps of spike sorting.

1. Concatenate recording sessions (in PlexUtil).
2. Do spike sorting (in Plexon OfflineSorter).
3. Split sorted recording (in PlexUtil).


## 1. Generate merged (concatenated) recording file.

- Start PlexUtil (shortcut is on Desktop).
- Open data folder of recording in left panel and select Plexon files to be merged. 
  - Make sure you select all unsorted task files of recording session and nothing else, e.g. previously merged or sorted files that may have accidentally been left in the folder during previous sorting!
- In right panel, arrange data files in order of recording time given by index following experiment name (see lab’s [file naming convention](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Naming%20conventions.md)), e.g. dd1, ddRem2, ddPas3. Use ‘Move Up’ and ‘Move Down’ buttons below panel.
  - Doublecheck that you selected all task recording plx files, and nothing else, and that they are in the right order.
- Press 'Merge' button in toolbar.
- For merge type, select second option: ‘Merge files consecutively in specified order (top to bottom)’, then press 'Next'.
- Set output folder and file name in next window.
  - Change folder from ..\Plexon\Unsorted\* to ..\Plexon\Sorted\* (see [folder structure conventions](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Data%20folder%20structure%20conventions.md))
  - Keep suggested filename unchanged.
- Press 'Next'.
- Press 'Finish' to generate merged Plexon file.

## 2. Do spike sorting.

- Start OfflineSorter (shortcut is on Desktop, the one without any version number).
- Open the merged file (output of above section).
- Do spike sorting on all channels (1 – 16). See [SOPs on spike sorting](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Spike%20sorting%20SOPs.md).
- When finished with all channels, save sorted data to the same folder. You can accept the file name suggested by OfflineSorter.

## 3. Split sorted recording.

- Start PlexUtil (shortcut is on Desktop).
- Select sorted file (output of above section) and press 'Split' button in toolbar.
- Select ‘Split into Multiple PLX Files’, then ‘All Channels’, then ‘By Frames’.
- Set output folder and file name in next window.
  - Folder should be: ..\Sorted (see [folder structure conventions](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Data%20folder%20structure%20conventions.md))
  - Accept suggested file name without any change.
- Click 'Finish'.
