# -*- coding: utf-8 -*-

"""
Module to configure Seal modules with user-defined experiment constants.

@author: David Samu
"""

import os
import importlib

from types import ModuleType

from quantities import units


# Symbols (labels) of all physical quantities.
pqs = [u.symbol for u in units.__dict__.values()
       if isinstance(u, type(units.deg))]


def init_constants(fconstants):
    """Function to configure package-wide constants about experiment."""

    # Import file with constants as a module.
    cdir, cf = os.path.split(fconstants)
    cfn, cext = os.path.splitext(cf)
    cwd = os.getcwd()
    os.chdir(cdir)
    constants = importlib.import_module(cfn)
    os.chdir(cwd)

    # Put all variables into a dict.
    cvars = vars(constants)

    # Exclude some module variables.
    constants = dict((k, v) for k, v in cvars.items()
                     if ((k[0] != '_') and                     # local vars
                         (not isinstance(v, ModuleType) and    # modules
                         (k not in pqs))))                     # quantities

    # Separate task info from task constants.
    task_info = constants.pop('task_info')

    return task_info, constants
