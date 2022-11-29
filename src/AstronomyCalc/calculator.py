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