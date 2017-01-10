# Direction selectivity

Direction selectivity can be tested by several different methods. A method with intermediate complexity is presented below.

One basic method to assess direction selectivity is to calculate the (normalised) average of the weighted 2D direction vectors, using unit activity (mean spike counts) during stimulus presentation as the weight of each direction. This method results in a mean directional activity vector (mDAV), that is used to derive further metrics about the direction specific response of the unit. mDAV is calculated to both (all) stimuli of the experiment (S1 and S2).


## Preferred direction (PD)

The **angle** of mDAV.


## Coarse-graded preferred direction (PD8)

As PD, but coarse-graded (rounded) to the closest one of the 8 original directions of the experiment.


## Direction selectivity index (DSI)

The **length** of mDAV, normalised by the maximum directional activity. Roughly similar to the inverse of the spread of activity across all directions, but perhaps more strict, as opposite directions cancel each other out. It is

- 0, if there is no direction selectivity in the unit's response. This happens if spikes are equally distributed across directions, or, more generally, the direction-specific activity vectors cancel out each other.
- 1, if there is maximal (100%) selectivity. This is the case when all spikes occur during one direction only.





