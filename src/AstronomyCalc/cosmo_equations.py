import numpy as np
from scipy.integrate import quad

from .basic_functions import *
from . import constants as const

class FriedmannEquation:
    """
    Friedmann equation for cosmology.

    This class provides methods to compute the Hubble parameter and the age of the Universe based on the Friedmann equation.

    Attributes
    ----------
    param : object
        An object containing cosmological parameter values.

    Methods
    -------
    create_functions()
        Initializes the functions for the Hubble parameter and its dependencies.
    H(z=None, a=None)
        Returns the Hubble parameter as a function of redshift `z` or scale factor `a`.
    age(z=None, a=None)
        Returns the age of the Universe as a function of redshift `z` or scale factor `a`.
    """

    def __init__(self, param):
        """
        Initializes the FriedmannEquation with cosmological parameters.

        Parameters
        ----------
        param : object
            An object containing cosmological parameter values.
        """
        self.param = param
        self.create_functions()

    def create_functions(self):
        """
        Initializes the functions for the Hubble parameter and its dependencies based on the cosmological parameters.
        """
        self.Ea = lambda a: np.sqrt(self.param.cosmo.Or/a**4 + self.param.cosmo.Om/a**3 +
                                    self.param.cosmo.Ode + self.param.cosmo.Ok/a**2)
        self.Ez = lambda z: self.Ea(z_to_a(z))
        self.Ha = lambda a: (self.param.cosmo.h * 100) * self.Ea(a)
        self.Hz = lambda z: self.Ha(z_to_a(z))

    def H(self, z=None, a=None):
        """
        Computes the Hubble parameter.

        Parameters
        ----------
        z : float, optional
            The redshift. If provided, the Hubble parameter is computed as a function of `z`.
        a : float, optional
            The scale factor. If provided, the Hubble parameter is computed as a function of `a`.

        Returns
        -------
        float
            The Hubble parameter in km/s/Mpc.
        """
        assert z is not None or a is not None, "Either redshift (z) or scale factor (a) must be provided."
        if z is not None:
            return self.Hz(z)
        else:
            return self.Ha(a)

    def age(self, z=None, a=None):
        """
        Computes the age of the Universe.

        Parameters
        ----------
        z : float, optional
            The redshift. If provided, the age is computed as a function of `z`.
        a : float, optional
            The scale factor. If provided, the age is computed as a function of `a`.

        Returns
        -------
        float
            The age of the Universe in Gyr.
        """
        assert z is not None or a is not None, "Either redshift (z) or scale factor (a) must be provided."
        if a is None:
            a = z_to_a(z)
        I = lambda a: 1/a/self.H(a=a)
        t = lambda a: quad(I, 0, a)[0] * const.Mpc_to_km / const.Gyr_to_s
        return np.vectorize(t)(a)

class CosmoDistances(FriedmannEquation):
    """
    Cosmological distances calculations based on the Friedmann equation.

    This class extends the FriedmannEquation class to provide methods for calculating various cosmological distances.

    Attributes
    ----------
    param : object
        An object containing cosmological parameter values.

    Methods
    -------
    Hubble_dist()
        Returns the Hubble distance at redshift `z=0`.
    comoving_dist(z=None, a=None)
        Returns the comoving distance as a function of redshift `z` or scale factor `a`.
    proper_dist(z=None, a=None)
        Returns the proper distance as a function of redshift `z` or scale factor `a`.
    light_travel_dist(z=None, a=None)
        Returns the light travel distance as a function of redshift `z` or scale factor `a`.
    angular_dist(z=None, a=None)
        Returns the angular diameter distance as a function of redshift `z` or scale factor `a`.
    luminosity_dist(z=None, a=None)
        Returns the luminosity distance as a function of redshift `z` or scale factor `a`.
    horizon_dist()
        Returns the comoving distance to the particle horizon.
    """

    def __init__(self, param):
        """
        Initializes the CosmoDistances with cosmological parameters.

        Parameters
        ----------
        param : object
            An object containing cosmological parameter values.
        """
        super().__init__(param)

    def Hubble_dist(self):
        """
        Computes the Hubble distance at redshift `z=0`.

        Returns
        -------
        float
            The Hubble distance in Mpc.
        """
        return const.c_kmps / self.H(z=0)

    def _comoving_dist(self, z):
        """
        Computes the comoving distance as a function of redshift `z`.

        Parameters
        ----------
        z : float
            The redshift.

        Returns
        -------
        float
            The comoving distance in Mpc.
        """
        I = lambda z: const.c_kmps / self.H(z=z)
        return quad(I, 0, z)[0]  # Mpc

    def comoving_dist(self, z=None, a=None):
        """
        Computes the comoving distance.

        Parameters
        ----------
        z : float, optional
            The redshift. If provided, the comoving distance is computed as a function of `z`.
        a : float, optional
            The scale factor. If provided, the comoving distance is computed as a function of `a`.

        Returns
        -------
        float
            The comoving distance in Mpc.
        """
        if a is not None:
            z = a_to_z(a)
        return np.vectorize(self._comoving_dist)(z)

    def proper_dist(self, z=None, a=None):
        """
        Computes the proper distance.

        Parameters
        ----------
        z : float, optional
            The redshift. If provided, the proper distance is computed as a function of `z`.
        a : float, optional
            The scale factor. If provided, the proper distance is computed as a function of `a`.

        Returns
        -------
        float
            The proper distance in Mpc.
        """
        dc = self.comoving_dist(z=z, a=a)
        return dc / (1 + z)

    def light_travel_dist(self, z=None, a=None):
        """
        Computes the light travel distance.

        Parameters
        ----------
        z : float, optional
            The redshift. If provided, the light travel distance is computed as a function of `z`.
        a : float, optional
            The scale factor. If provided, the light travel distance is computed as a function of `a`.

        Returns
        -------
        float
            The light travel distance in Mpc.
        """
        t0 = self.age(z=0)
        te = self.age(z=z, a=a)
        return const.c_kmps * (t0 - te) * const.Gyr_to_s / const.Mpc_to_km

    def angular_dist(self, z=None, a=None):
        """
        Computes the angular diameter distance.

        Parameters
        ----------
        z : float, optional
            The redshift. If provided, the angular diameter distance is computed as a function of `z`.
        a : float, optional
            The scale factor. If provided, the angular diameter distance is computed as a function of `a`.

        Returns
        -------
        float
            The angular diameter distance in Mpc.
        """
        dc = self.comoving_dist(z=z, a=a)
        return dc / (1 + z)

    def luminosity_dist(self, z=None, a=None):
        """
        Computes the luminosity distance.

        Parameters
        ----------
        z : float, optional
            The redshift. If provided, the luminosity distance is computed as a function of `z`.
        a : float, optional
            The scale factor. If provided, the luminosity distance is computed as a function of `a`.

        Returns
        -------
        float
            The luminosity distance in Mpc.
        """
        dc = self.comoving_dist(z=z, a=a)
        return dc * (1 + z)

    def horizon_dist(self):
        """
        Computes the comoving distance to the particle horizon.

        Returns
        -------
        float
            The comoving distance to the particle horizon in Mpc.
        """
        return self.comoving_dist(z=1e-8)
