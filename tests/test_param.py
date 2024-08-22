import numpy as np
import AstronomyCalc as astro

def test_param():
    par = astro.param()
    assert np.abs(par.cosmo.Tcmb-2.725)<0.1