# Recording folder structure conventions

The folder structure storing all recording data files, and additional data files created during [preprocessing](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Preprocessing%20SOPs.md), should follow the convention below.

Y:\yesbackup\ElectrophysiologyData\Recordings\
  - "subject name"\    (e.g. 201\)
    - "subfolder"\ (a descriptive name to sub-group recordings, e.g. PFC Recordings\ or Inactivation\\)
      - "recording name"\    (in subject_date format, e.g. 201_062416\\)
        - Plexon\
          - Unsorted\
            - [list of unsorted Plexon files per task, e.g. 201_062416v1_com1_0.plx, ..., see [file naming convention](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Data%20file%20naming%20conventions.md)]
          - Sorted\
            - [list of unsorted Plexon files per task, e.g. 201_062416v1_com1_1.plx, ..., see [file naming convention](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Data%20file%20naming%20conventions.md)]
        - RF Mapping  (folder containing receptive field mapping recordings, parameters and analysis results, only for MT recordings)
        - Matlab\
          - TPLCell\
            - [list of TPLCell Matlab data files per task, e.g. 201_062416v1_com1_1.mat]
          - TPLStruct\
            - [list of TPLStruct Matlab data files per task, e.g. 201_062416v1_com1_1.mat]
        - Python\
          - SealUnits\
          - SealCombined\
        - QC\    (folder with quality control results and figures)
        - RP\    (folder with response plots over trial time)

For an example, see y:\yesbackup\ElectrophysiologyData\Recordings\202\MT Recordings\202_021116\


# Subfolder naming

These are the folders immediately below the monkey folder (see "subfolder" above). They group experiements into subcategories by monkey. Naming conventions of these folders are the following (all case sensitive!):

- MT recordings must contains the string 'MT'
- PFC recordings must contains the string 'PFC'
- simultaneous recordings must contains the string 'Simultaneous'
- inactivation experiments must contains the string 'Inactivation'
- combined experiments must contains the string 'Combined'

Please use existing subfolders for new recordings, and only create a new subfolder if the available ones do not match current recordings (e.g. starting a new type experiment or recording site for a give monkey).

