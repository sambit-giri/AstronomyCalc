"""
External Parameters

This is a useful file if this package is a simulator. 
All the parameter can be initialised and them passed as 
an param object to the simulator functions and classes.
"""

import numpy as np

class Bunch(object):
    """
    A simple class to translate dictionary keys to object attributes.

    Attributes:
        __dict__ (dict): Updates the object's __dict__ with the provided data dictionary.
    """

    def __init__(self, data):
        self.__dict__.update(data)

def code_par(**kwargs): 
    """
    Define default parameters for cosmological simulations.

    Keyword Args:
        zmin (float): Minimum redshift. Default is 0.01.
        zmax (float): Maximum redshift. Default is 9.00.
        Nz (int): Number of redshift bins. Default is 20.
        verbose (bool): If True, the simulator prints messages. Default is True.

    Returns:
        Bunch: An object with the specified parameters as attributes.
    """
    par = {
        "zmin": 0.01,               # min redshift
        "zmax": 9.00,               # max redshift (not tested below 40)
        "Nz": 20,                   # Nb of redshift bins
        "verbose": True,            # If True, then the simulator prints messages.
    }
    par.update(kwargs)
    return Bunch(par)

def cosmo_par(**kwargs): 
    """
    Define default cosmological parameters.

    Keyword Args:
        Om (float): Matter overdensity. Default is 0.31.
        Or (float): Radiation overdensity. Default is 9e-5.
        Ok (float): Curvature overdensity. Default is 0.0.
        Ode (float): Dark energy overdensity. Default is None (calculated as 1 - Om - Or - Ok).
        h (float): Hubble constant divided by 100. Default is 0.68.
        Tcmb (float): CMB temperature at redshift 0. Default is 2.725.

    Returns:
        Bunch: An object with the specified parameters as attributes.
    """
    par = {
        "Om" : 0.31,                # matter overdensity
        "Or" : 9e-5,                # radiation overdensity
        "Ok" : 0.0,                 # Curvature overdensity
        "Ode": None,                # Dark energy overdensity
        "h"  : 0.68,                # Hubble constant divided by 100
        "Tcmb": 2.725,              # CMB temperature at redshift 0.
    }
    par.update(kwargs)
    if par['Ode'] is None: 
        par['Ode'] = 1 - par['Om'] - par['Or'] - par['Ok']
    return Bunch(par)

def param(cosmo=None, code=None):
    """
    Combine cosmological and code parameters into a single Bunch object.

    Args:
        cosmo (dict, optional): Dictionary of cosmological parameters to override defaults.
        code (dict, optional): Dictionary of code parameters to override defaults.

    Returns:
        Bunch: An object containing both cosmological and code parameters as attributes.
    """
    return Bunch({
        "cosmo": cosmo_par() if cosmo is None else cosmo_par(**cosmo),
        "code": code_par() if code is None else code_par(**code),
    })