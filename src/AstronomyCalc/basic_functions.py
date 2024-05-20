import numpy as np 

def z_to_a(z):
    '''
    Convert redshift (z) to scale factor (a).
    
    Parameters:
        z (float or numpy array): redshift value(s).
    Returns:
        scale factor
    '''
    return 1/(1+z)

def a_to_z(a):
    '''
    Convert scale factor (a) to redshift (z).
    
    Parameters:
        a (float or numpy array): scale factor value(s).
    Returns:
        redshifts
    '''
    return 1./a-1