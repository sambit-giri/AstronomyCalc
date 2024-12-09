import numpy as np 
import AstronomyCalc as astro

def test_param():
	param = astro.param()
	assert np.abs(param.cosmo.Om-0.31)<0.01 and np.abs(param.cosmo.h-0.68)<0.02

def test_age_estimator():
	param = astro.param()
	t0 = astro.age_estimator(param, 0).value
	assert np.abs(t0-13.74)<0.01

def test_cosmo_dist():
	param = astro.param()
	dist = astro.CosmoDistances(param)
	cdist = dist.comoving_dist(1).value #3380.72792719
	ldist = dist.luminosity_dist(1).value #6761.45585438037
	assert np.abs(cdist-3380.73)<0.1 and np.abs(ldist-6761.46)<0.1

def test_hubble():
	param = astro.param()
	fried = astro.FriedmannEquation(param)
	assert np.abs(fried.Hz(1)-121.1)<0.1 