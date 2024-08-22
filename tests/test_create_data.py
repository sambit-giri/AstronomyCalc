import numpy as np 
import AstronomyCalc as astro

def test_line_data():
    xobs, yobs = astro.line_data(true_m=2, true_b=1, sigma=0.1, n_samples=100).T
    m_mean = np.mean((yobs[1:]-1)/(xobs[1:]))
    assert np.abs(m_mean-2)<0.1