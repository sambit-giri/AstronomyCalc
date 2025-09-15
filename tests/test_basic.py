import numpy as np
import AstronomyCalc as astro

def test_z_to_a():
    assert np.abs(astro.z_to_a(1)-0.5)<0.01

def test_a_to_z():
    assert np.abs(astro.a_to_z(0.5)-1)<0.01