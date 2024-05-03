import numpy as np
import pandas as pd

from .cosmo_equations import *

def distance_modulus(param=None, random_state=42):
    from astroML.datasets import generate_mu_z
    z_sample, mu_sample, dmu = generate_mu_z(100, random_state=random_state)
    return z_sample, mu_sample, dmu

def Hubble1929_data():
    data_link = "https://github.com/behrouzz/astrodatascience/raw/main/data/hubble1929.csv"
    df = pd.read_csv(data_link)
    # print(df.head())
    return np.array(df['distance']), np.array(df['velocity'])