# Recording folder structure conventions

The folder structure storing all recording data files, and additional data files created during [preprocessing](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Preprocessing%20SOPs.md), should follow the convention below.

Y:\yesbackup\ElectrophysiologyData\Recordings\
  - "subject name"\    (e.g. 201\)
    - "subfolder"\ (a descriptive name to sub-group recordings, e.g. PFC\ or Inactivation\)
      - "recording name"\    (in subject_date format, e.g. 201_062416\)
        - Plexon\
          - Unsorted\
            - [list of unsorted Plexon files per task, e.g. 201_062416v1_com1_0.plx, ..., see [file naming convention](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Data%20file%20naming%20conventions.md)]
          - Sorted\
            - [list of unsorted Plexon files per task, e.g. 201_062416v1_com1_1.plx, ..., see [file naming convention](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Data%20file%20naming%20conventions.md)]
        - RF Mapping  (folder containing receptive field mapping results, only for MT recordings)
        - Matlab\
          - TPLCell\
            - [list of TPLCell Matlab data files per task, e.g. 201_062416v1_com1_1.mat]
          - TPLStruct\
        - Python\
          - SealUnits\
          - SealCombined\
        - QC\    (folder with quality control results and figures)
        - RP\    (folder with response plots over trial time)

For an example, see y:\yesbackup\ElectrophysiologyData\Recordings\202\MT Recordings\202_021116\
