import numpy as np 
import AstronomyCalc as astro

def test_age_estimator():
	param = astro.param()
	t0 = astro.age_estimator(param, 0)
	assert np.abs(t0-13.74)<0.01