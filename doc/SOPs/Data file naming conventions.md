# Naming convention for recorded data files

Recordings and generated data files should follow the following naming template:

*SSS_MMDDYYee_TTi_j.EXT*

where

- *SSS*: subject name (monkey ID, three digits), e.g. *201*
- *MMDDYY*: date of the recording (in format MMDDYY, i.e. in month, day, year order, two digits each), e.g. *062416*
- *ee*: electrode type (one lowercase letter) and number (one digit), e.g. *v1*
- *TT*: task abbreviation (letters of varying length and case, see below), e.g. *ddRem*
- *i*: order index of task among all tasks of the recording session, e.g. *3*
- *j*: sorting index (legacy), how many times the plx files has been sorted. 0 for unsorted files, >= 1 for sorted files.
- *EXT*: extension of the file, *plx*, *mat* or *data*

Example:

*202_123116v1_ddX2_1.plx*


# Task naming convention

This is the *TT* part of the filename above. It composed of the following keywords, all case sensitive (i.e. *dd* must be all lowercase, NOT capitalised *Dd* or uppercase *DD*; similarly, *Rem* must be capitalised, not lower- or uppercase).

- 1st part: main task (stimulus feature to be reported), all lowercase!
  - *com*: combined (direction and location)
  - *dd*: direction
  - *loc*: location
  - *rng*: range
  - *mem*: memory saccade (?)
  - *map*: RF mapping, this can contain additional characters, e.g. *mapping*
- 2nd part: task modifier, capitalized!
  - *Pas*: passive (no saccade required, just fixation)
  - *Rem*: remote (both, or one(?) of S1 and S2 appear out of RF, at the ipsilater hemifield)
  - *Sep*: separated (one of S1 or S2 appears within RF, the other out of RF)
  - *Sim*: simultaneous MT + PFC recordings
  - *Uncert*: uncertain location (S2 appears at same/different location to S1 at 50/50% of the times)
  - *X*: PFC inactivated
  
Multiple modifier keywords can be used, e.g. ddRemPasX, but keep with order applied for previous recordings. For all task names that occur in previous recordings, see column headers of recording table files in 
y:\yesbackup\ElectrophysiologyData\Session Notes\Generated Recording Tables\Recording Tables\

