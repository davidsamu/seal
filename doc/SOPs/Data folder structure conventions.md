# Recording folder structure conventions

The folder structure storing all recording data files, and additional data files created during [preprocessing](https://github.com/davidsamu/seal/blob/master/doc/SOPs/Preprocessing%20SOPs.md), should follow the convention below.

Y:\yesbackup\ElectrophysiologyData\Plexon\
  - "subject name"\    (e.g. 201\)
    - "subfolder"\ (a descriptive name to sub-group recordings, e.g. PFC\ or Inactivation\)
      - "recording name"\    (in subject_date fromat, e.g. 201_062416\)
        - Unsorted\
          - [list of unsorted Plexon files per task, e.g. 201_062416v1_com1_0.plx]
        - Sorted\
          - Merged\
            - [single merged Plexon file, e.g. 201_062416v1_mrg-02.plx]
          - Split\
            - [list of sorted and split Plexon files per task, e.g. 201_062416v1_com1_0.plx]
        - TPLCell\
          - [list of TPLCell Matlab data files per task, e.g. 201_062416v1_com1_0.mat]

For an example, see Y:\yesbackup\ElectrophysiologyData\Plexon\201\PFC\201_062416\
