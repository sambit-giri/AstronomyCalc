import numpy as np
import pandas as pd
from astropy.cosmology import LambdaCDM
from astropy import units as u 
import os, requests, zipfile, pkg_resources, wget

from .cosmo_equations import *

def line_data(true_m=2, true_b=1, sigma=1, n_samples=50, error_sigma=None):
    """
    Generate synthetic linear data for testing and modeling.

    This function generates a set of synthetic data points that lie on a line 
    with a specified slope and intercept, while adding Gaussian noise to the 
    y-values to simulate real-world data.

    Parameters:
    - true_m (float): The slope of the line. Defaults to 2.
    - true_b (float): The y-intercept of the line. Defaults to 1.
    - sigma (float): The standard deviation of the Gaussian noise added to the y-values. Defaults to 1.
    - n_samples (int): The number of data points to generate. Defaults to 50.

    Returns:
    - np.ndarray: A 2D array where each row is a data point with two columns: the x-value and the corresponding y-value. The x-values are evenly spaced between 0 and 10, and the y-values are generated according to the line equation `y = true_m * x + true_b` with added Gaussian noise.
    """
    np.random.seed(42)  # For reproducibility
    x = np.linspace(0, 10, n_samples)
    y = true_m * x + true_b + np.random.normal(0, sigma, size=x.shape)
    if error_sigma is None:
        data = np.column_stack((x, y))
    else:
        y_error = np.abs(np.random.normal(sigma, error_sigma, size=x.shape))
        data = np.column_stack((x, y, y_error))
    return data


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
    Load the Hubble (1929) dataset of distance vs velocity.

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

def SPARC_galaxy_rotation_curves_data(filename=None, name=None):
    """
    Retrieves the rotation curve data for a specific galaxy from the SPARC dataset.

    This function utilizes the SPARC_Galaxy_dataset class to read the rotation curves
    of galaxies from the SPARC dataset. The data can be accessed either by directly 
    providing the filename or by specifying the galaxy name.

    Parameters:
        filename (str, optional): The path to the rotation curve file. If provided, 
                                  this file will be used directly.
        name (str, optional): The name of the galaxy for which to retrieve rotation 
                              curve data. If provided, the function will look for 
                              the corresponding file in the dataset.

    Returns:
        dict: A dictionary containing the rotation curve data. The dictionary has two keys:
            - 'values': A dictionary where each key corresponds to a specific quantity 
                        (e.g., 'Rad', 'Vobs') and the value is an array of data for 
                        that quantity.
            - 'units': A dictionary mapping each quantity to its unit (e.g., 'kpc', 
                       'km/s').

    Raises:
        AssertionError: If neither `filename` nor `name` is provided, the function 
                        raises an assertion error, requiring one of the inputs.
    """
    SPARC = SPARC_Galaxy_dataset()
    data = SPARC.read_rotation_curves(filename=filename, name=name)
    return data

class SPARC_Galaxy_dataset:
    """
    A class to handle the SPARC Galaxy dataset (http://astroweb.cwru.edu/SPARC/).

    This class can download, read, and process files from the SPARC Galaxy dataset.

    Attributes:
        package_folder (str): Path to the package input data folder.
        data_folder (str): Path to the folder containing the rotation curve data.
    """

    def __init__(self):
        """
        Initializes the SPARC_Galaxy_dataset class.

        If the data folder does not exist, it triggers the download of the dataset.
        """
        self.package_folder = pkg_resources.resource_filename('AstronomyCalc', 'input_data/')
        self.data_folder = os.path.join(self.package_folder, "Rotmod_LTG")
        if not os.path.exists(self.data_folder):
            self.download_data()

    def read_rotation_curves(self, filename=None, name=None):
        # Reads the rotation curves from the SPARC dataset.

        assert filename is not None or name is not None, "Either filename or name must be provided."
        
        quantities = ['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']
        units_ = ['kpc', 'km/s', 'km/s', 'km/s', 'km/s', 'km/s', 'L/pc^2', 'L/pc^2']
        
        if name is not None:
            filename = os.path.join(self.data_folder, f"{name}_rotmod.dat")
        
        rd = np.loadtxt(filename)
        values = {quantities[i]: rd[:, i] for i in range(len(quantities))}
        units = {quantities[i]: units_[i] for i in range(len(quantities))}
        
        return {'values': values, 'units': units}

    def download_data(self):
        # Downloads and extracts the SPARC Galaxy dataset.

        url = "http://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip"
        target_folder = self.package_folder
        
        wget.download(url, target_folder)
        
        zip_file_path = os.path.join(target_folder, "Rotmod_LTG.zip")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_file_target = os.path.join(target_folder, "Rotmod_LTG")
            zip_ref.extractall(zip_file_target)
        
        os.remove(zip_file_path)
        print("...download and extraction successful.")

        return zip_file_target