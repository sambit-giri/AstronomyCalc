import numpy as np 
from scipy.integrate import quad

from .basic_functions import *
from .cosmo_equations import * 
from . import constants as const

def age_estimator(param, z):
    '''
    Age estimator of the Universe.
    
    Parameters:
        param (object): object containing the parameter values
        z (float): redshift
            
    Returns:
        * The age of the universe in Gyr
    '''
    Feq = FriedmannEquation(param)
    a = z_to_a(z)
    I = lambda a: 1/a/Feq.H(a=a)
    t = lambda a: quad(I, 0, a)[0]*const.Mpc_to_km/const.Gyr_to_s
    return np.vectorize(t)(a) 

def Hubble(param, z=None, a=None):
    """
    Calculate the Hubble parameter at a given redshift or scale factor.

    Parameters:
        param (dict): A dictionary containing cosmological parameters.
        z (float, optional): The redshift at which to evaluate the Hubble parameter.
        a (float, optional): The scale factor at which to evaluate the Hubble parameter.
                             Only one of `z` or `a` should be provided.

    Returns:
        float: The Hubble parameter (H) at the specified redshift or scale factor.
    """
    cosmo = CosmoDistances(param)
    return cosmo.H(z=z, a=a)

def cosmic_age(param, z=None, a=None):
    """
    Calculate the cosmic age at a given redshift or scale factor.

    Parameters:
        param (dict): A dictionary containing cosmological parameters.
        z (float, optional): The redshift at which to calculate the cosmic age.
        a (float, optional): The scale factor at which to calculate the cosmic age.
                             Only one of `z` or `a` should be provided.

    Returns:
        float: The age of the universe in Gyr at the specified redshift or scale factor.
    """
    cosmo = CosmoDistances(param)
    return cosmo.age(z=z, a=a)
    
def Hubble_distance(param):
    """
    Calculate the Hubble distance.

    Parameters:
        param (dict): A dictionary containing cosmological parameters.

    Returns:
        float: The Hubble distance (c / H0) in Mpc.
    """
    cosmo = CosmoDistances(param)
    return cosmo.Hubble_dist()

def comoving_distance(param, z=None, a=None):
    """
    Calculate the comoving distance at a given redshift or scale factor.

    Parameters:
        param (dict): A dictionary containing cosmological parameters.
        z (float, optional): The redshift at which to calculate the comoving distance.
        a (float, optional): The scale factor at which to calculate the comoving distance.
                             Only one of `z` or `a` should be provided.

    Returns:
        float: The comoving distance in Mpc at the specified redshift or scale factor.
    """
    cosmo = CosmoDistances(param)
    return cosmo.comoving_dist(z=z, a=a)

def proper_distance(param, z=None, a=None):
    """
    Calculate the proper distance at a given redshift or scale factor.

    Parameters:
        param (dict): A dictionary containing cosmological parameters.
        z (float, optional): The redshift at which to calculate the proper distance.
        a (float, optional): The scale factor at which to calculate the proper distance.
                             Only one of `z` or `a` should be provided.

    Returns:
        float: The proper distance in Mpc at the specified redshift or scale factor.
    """
    cosmo = CosmoDistances(param)
    return cosmo.proper_dist(z=z, a=a)

def light_travel_distance(param, z=None, a=None):
    """
    Calculate the light travel distance (also known as lookback distance) at a given redshift or scale factor.

    Parameters:
        param (dict): A dictionary containing cosmological parameters.
        z (float, optional): The redshift at which to calculate the light travel distance.
        a (float, optional): The scale factor at which to calculate the light travel distance.
                             Only one of `z` or `a` should be provided.

    Returns:
        float: The light travel distance in Mpc at the specified redshift or scale factor.
    """
    cosmo = CosmoDistances(param)
    return cosmo.light_travel_dist(z=z, a=a)

def angular_distance(param, z=None, a=None):
    """
    Calculate the angular diameter distance at a given redshift or scale factor.

    Parameters:
        param (dict): A dictionary containing cosmological parameters.
        z (float, optional): The redshift at which to calculate the angular distance.
        a (float, optional): The scale factor at which to calculate the angular distance.
                             Only one of `z` or `a` should be provided.

    Returns:
        float: The angular diameter distance in Mpc at the specified redshift or scale factor.
    """
    cosmo = CosmoDistances(param)
    return cosmo.angular_dist(z=z, a=a)

def luminosity_distance(param, z=None, a=None):
    """
    Calculate the luminosity distance at a given redshift or scale factor.

    Parameters:
        param (dict): A dictionary containing cosmological parameters.
        z (float, optional): The redshift at which to calculate the luminosity distance.
        a (float, optional): The scale factor at which to calculate the luminosity distance.
                             Only one of `z` or `a` should be provided.

    Returns:
        float: The luminosity distance in Mpc at the specified redshift or scale factor.
    """
    cosmo = CosmoDistances(param)
    return cosmo.luminosity_dist(z=z, a=a)

def horizon_distance(param):
    """
    Calculate the horizon distance, which is the maximum distance from which light has traveled to the observer
    in the age of the universe.

    Parameters:
        param (dict): A dictionary containing cosmological parameters.

    Returns:
        float: The horizon distance in Mpc.
    """
    cosmo = CosmoDistances(param)
    return cosmo.horizon_dist()

def distance_modulus(param, z=None, a=None):
    """
    Calculate the distance modulus at a given redshift or scale factor.

    Parameters:
        param (dict): A dictionary containing cosmological parameters.
        z (float, optional): The redshift at which to calculate the luminosity distance.
        a (float, optional): The scale factor at which to calculate the luminosity distance.
                             Only one of `z` or `a` should be provided.

    Returns:
        float: The distance modulus in Mpc at the specified redshift or scale factor.
    """
    if z is None: z = a_to_z(a)
    lumdist = lambda z: luminosity_distance(param, z)
    distmod = lambda z: 5*np.log10(lumdist(z))+25
    return distmod(z)