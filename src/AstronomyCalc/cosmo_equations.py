import numpy as np
from scipy.integrate import quad
from astropy import units as u
from astropy import constants as const

from .basic_functions import *

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
        if z is not None: return self.Hz(z)*u.km/u.Mpc/u.s
        else: return self.Ha(a)*u.km/u.Mpc/u.s
    
    def age(self, z=None, a=None):
        assert z is not None or a is not None
        if a is None: a = z_to_a(z)
        I = lambda a: 1/a/self.H(a=a)
        t = lambda a: quad(lambda x: I(x).value, 0, a)[0]
        return (np.vectorize(t)(a)*I(1).unit).to('Gyr') 

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
        return const.c/self.H(z=0)

    def _comoving_dist(self, z):
        I = lambda z: const.c/self.H(z=z)
        return (quad(lambda x: I(x).value, 0, z)[0]*I(0).unit).to('Mpc').value # Mpc

    def comoving_dist(self, z=None, a=None):
        if a is not None: z = a_to_z(a)
        return np.vectorize(self._comoving_dist)(z)*u.Mpc

    def proper_dist(self, z=None, a=None):
        dc = self.comoving_dist(z=z, a=a)
        return dc/(1+z)

    def light_travel_dist(self, z=None, a=None):
        t0 = self.age(z=0)
        te = self.age(z=z, a=a)
        return (const.c*(t0-te)).to('Mpc')

    def angular_dist(self, z=None, a=None):
        dc = self.comoving_dist(z=z, a=a)
        return dc/(1+z)
    
    def luminosity_dist(self, z=None, a=None):
        dc = self.comoving_dist(z=z, a=a)
        return dc*(1+z)

    def horizon_dist(self):
        return self.comoving_dist(self, a=1e-8)
