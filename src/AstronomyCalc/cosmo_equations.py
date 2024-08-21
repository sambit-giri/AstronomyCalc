import numpy as np
from scipy.integrate import quad

from .basic_functions import *
from . import constants as const

class FriedmannEquation:
    '''
    Friedmann equation.
    
    Attributes:
        H: function to give the value of the Hubble parameter in km/s/Mpc.
    
    '''
    def __init__(self, param):
        '''
        Initialize the file.
        
        Parameters:
            param (object): object containing the parameter values
        Returns:
            Nothing
        '''
        self.param = param
        self.create_functions()

    def create_functions(self):
        self.Ea = lambda a: np.sqrt(self.param.cosmo.Or/a**4+self.param.cosmo.Om/a**3+\
                            self.param.cosmo.Ode+self.param.cosmo.Ok/a**2)
        self.Ez = lambda z: self.Ea(z_to_a(z))
        self.Ha = lambda a: (self.param.cosmo.h*100)*self.Ea(a)
        self.Hz = lambda z: self.Ha(z_to_a(z))

    def H(self, z=None, a=None):
        assert z is not None or a is not None 
        if z is not None: return self.Hz(z)
        else: return self.Ha(a)
    
    def age(self, z=None, a=None):
        assert z is not None or a is not None
        if a is None: a = z_to_a(z)
        I = lambda a: 1/a/self.H(a=a)
        t = lambda a: quad(I, 0, a)[0]*const.Mpc_to_km/const.Gyr_to_s
        return np.vectorize(t)(a) 

class CosmoDistances(FriedmannEquation):
    '''
    Cosmological distances
    
    Attributes:
        param (object): object containing the parameter values
    
    '''

    def __init__(self, param):
        '''
        Initialize the file.
        
        Parameters:
            param (object): object containing the parameter values
        Returns:
            Nothing
        '''
        self.param = param
        self.create_functions()

    def Hubble_dist(self):
        return const.c_kmps/self.H(z=0)

    def _comoving_dist(self, z):
        I = lambda z: const.c_kmps/self.H(z=z)
        return quad(I, 0, z)[0] # Mpc

    def comoving_dist(self, z=None, a=None):
        if a is not None: z = a_to_z(a)
        return np.vectorize(self._comoving_dist)(z)

    def proper_dist(self, z=None, a=None):
        dc = self.comoving_dist(z=z, a=a)
        return dc/(1+z)

    def light_travel_dist(self, z=None, a=None):
        t0 = self.age(z=0)
        te = self.age(z=z, a=a)
        return const.c_kmps*(t0-te)*const.Gyr_to_s/const.Mpc_to_km

    def angular_dist(self, z=None, a=None):
        dc = self.comoving_dist(z=z, a=a)
        return dc/(1+z)
    
    def luminosity_dist(self, z=None, a=None):
        dc = self.comoving_dist(z=z, a=a)
        return dc*(1+z)

    def horizon_dist(self):
        return self.comoving_dist(self, a=1e-8)


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
