#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 13:06:55 2016

Functions for performing and processing decoding analyses.

@author: David Samu
"""


import numpy as np
from quantities import ms
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model import LogisticRegression

from seal.util import plot, util


