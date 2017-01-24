#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of generic constants (not specific to any experiment/task).

@author: David Samu
"""

import pandas as pd
from quantities import ms


# %% Neurophysiological constants.

nphy_cons = pd.DataFrame.from_items([('MT', (50*ms, 500*ms)),
                                     ('PFC', (100*ms, 200*ms))],
                                    ['latency', 'DSwindow'], 'index')
