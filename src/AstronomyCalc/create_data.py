import numpy as np
from .cosmo_equations import *

def distance_modulus(param=None, random_state=42):
    from astroML.datasets import generate_mu_z
    z_sample, mu_sample, dmu = generate_mu_z(100, random_state=random_state)
    return z_sample, mu_sample, dmu