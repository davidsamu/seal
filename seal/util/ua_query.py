# -*- coding: utf-8 -*-
"""
Utility functions to query information of units in UnitArray objects.

@author: David Samu
"""

import pandas as pd


def get_DSInfo_table(UA, utids=None):
    """Return data frame with direction selectivity information."""

    # Init.
    if utids is None:
        utids = UA.utids()

    DSInfo = []
    for utid in utids:
        u = UA.get_unit(utid[:3], utid[3])

        # Test DS if it has not been tested yet.
        if not len(u.DS):
            u.test_DS()

        # Get DS info.
        PD = u.DS.PD.cPD[('S1', 'max')]
        DSI = u.DS.DSI.mDS.S1

        DSInfo.append((utid, (PD, DSI)))

    DSInfo = pd.DataFrame.from_items(DSInfo, columns=['PD', 'DSI'],
                                     orient='index')

    return DSInfo
