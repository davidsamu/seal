#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 13:06:55 2016

Functions for performing and processing decoding analyses.

@author: David Samu
"""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold

from seal.util import plot, util


# %% Core decoding functions.

def run_logreg(X, y, ncv=5):
    """
    Run logistic regression with given number of cross-validation, 
    return performance and weights of each fold.
    """
    
    # Init.
    perf = pd.DataFrame(index=range(ncv), columns=['train', 'test'])
    weights = pd.DataFrame(index=range(ncv), columns=X.columns)
    skf = StratifiedKFold(y, ncv)
    
    # Run Logistic Regression on each fold and save performance and weights.
    LR = LogisticRegression()
    for igrp, (train, test) in enumerate(skf):
        LR.fit(X.iloc[train], y.iloc[train])
        perf.train[igrp] = LR.score(X.iloc[train], y.iloc[train])
        perf.test[igrp] = LR.score(X.iloc[test], y.iloc[test])
        weights.loc[igrp] = LR.coef_[0]
    
    # Return all performance and weights.
    return perf, weights
