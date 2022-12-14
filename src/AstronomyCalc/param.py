"""
External Parameters

This is a useful file if this package is a simulator. 
All the parameter can be initialised and them passed as 
an param object to the simulator functions and classes.
"""

import numpy as np

class Bunch(object):
    """
    translates dic['name'] into dic.name 
    """

    def __init__(self, data):
        self.__dict__.update(data)


def code_par(**kwargs): 
    # This is an example showing a few parameters for a cosmological simulations.
    par = {
        "zmin": 0.01,               # min redshift
        "zmax": 9.00,               # max redshift (not tested below 40)
        "Nz": 20,                  # Nb of redshift bins
        "verbose": True,           # If True, then the simulator prints messages.
        }
    par.update(kwargs)
    return Bunch(par)

def cosmo_par(**kwargs): 
    # This is an example showing a few parameters for a cosmological simulations.
    par = {
        "Om" : 0.31,               # matter overdensity
        "Or" : 9e-5,               # radiation overdensity
        "Ok" : 0.0,                # Curvature overdensity
        "Ode": None,               # Dark energy overdensity
        "h"  : 0.68,               # Nb of redshift bins
        }
    par.update(kwargs)
    if par['Ode'] is None: par['Ode'] = 1-par['Om']-par['Or']-par['Ok']
    return Bunch(par)

def param(cosmo=None, code=None):
    par = Bunch({
        "cosmo": cosmo_par() if cosmo is None else cosmo_par(**cosmo),
        "code": code_par() if code is None else code_par(**code),
        })
    return par