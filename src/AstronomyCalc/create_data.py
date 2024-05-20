import numpy as np
import pandas as pd
from astropy.cosmology import LambdaCDM

from .cosmo_equations import *

def distance_modulus(n_samples=100, z0=0.3, dmu_0=0.1, dmu_1=0.02, random_state=42, cosmo=None, param=None):
    """
    Generate a dataset of distance modulus (mu) vs redshift.
    
    Parameters:
        n_samples (int): Size of generated data.
        z0 (float): Parameter in redshift distribution: p(z) ∼ (z / z0)^2 exp[-1.5 (z / z0)].
        dmu_0 (float): Base error in mu.
        dmu_1 (float): Error in mu as a function of mu.
        random_state (int or np.random.RandomState instance): Random seed or random number generator.
        cosmo (astropy.cosmology instance): Cosmology to use when generating the sample. If not provided, a Flat Lambda CDM model with H0=71, Om0=0.27, Tcmb=0 is used.
        param (object): Object containing the parameter values.

    Returns:
        z (ndarray): Array of redshifts of shape (n_samples,).
        mu (ndarray): Array of distance moduli of shape (n_samples,).
        dmu (ndarray): Array of errors in distance moduli of shape (n_samples,).
    """
    from astroML.datasets import generate_mu_z
    if param is not None:
        cosmo = LambdaCDM(
                        H0=param.cosmo.h*100, 
                        Om0=param.cosmo.Om, 
                        Ode0=param.cosmo.Ode, 
                        Tcmb0=param.cosmo.Tcmb,
                        )
    z_sample, mu_sample, dmu = generate_mu_z(size=n_samples, z0=z0, dmu_0=dmu_0, dmu_1=dmu_1, random_state=random_state, cosmo=cosmo)
    return z_sample, mu_sample, dmu

def Hubble1929_data(data_link=None):
    """
    Load the 1929 Hubble dataset of distance vs. velocity.

    Parameters:
        data_link (str): URL to the CSV file containing the Hubble 1929 data. 
                         If not provided, the default link is used.

    Returns:
        tuple: Two ndarrays containing the distances and velocities, respectively.
               - distances (ndarray): Array of distances.
               - velocities (ndarray): Array of velocities.
    """
    if data_link is None:
        data_link = "https://github.com/behrouzz/astrodatascience/raw/main/data/hubble1929.csv"
    df = pd.read_csv(data_link)
    # print(df.head())
    return np.array(df['distance']), np.array(df['velocity'])