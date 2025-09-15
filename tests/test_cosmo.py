import numpy as np 
import AstronomyCalc as astro

def test_param():
	param = astro.param()
	assert np.abs(param.cosmo.Om-0.31)<0.01 and np.abs(param.cosmo.h-0.68)<0.02

def test_age_estimator():
	param = astro.param()
	t0 = astro.age_estimator(param, 0).to('Gyr').value #13.73072467 Gyr
	assert np.abs(t0-13.73)<0.01

def test_cosmo_dist():
	param = astro.param()
	Hdist = astro.Hubble_distance(param).to('Mpc').value              #4408.71261765 Mpc
	cdist = astro.comoving_distance(param, 1).to('Mpc').value         #3380.73309202 Mpc
	pdist = astro.proper_distance(param, 1).to('Mpc').value           #1690.36654601 Mpc
	tdist = astro.light_travel_distance(param, 1).to('Mpc').value     #2422.20670428 Mpc
	adist = astro.angular_diameter_distance(param, 1).to('Mpc').value #1690.36654601 Mpc
	ldist = astro.luminosity_distance(param, 1).to('Mpc').value       #6761.46618403 Mpc
	hdist = astro.horizon_distance(param).to('Mpc').value             #7389.99146892 Mpc
	mdist = astro.distance_modulus(param, 1)                          #44.150204401742954
	dists = np.array([Hdist, cdist, pdist, tdist, adist, ldist, hdist, mdist])
	value = np.array([4408.7,3380.7,1690.4,2422.2,1690.4,6761.5,7390.0,44.15])
	assert np.all(np.abs(dists-value)<0.1)

def test_hubble():
	param = astro.param()
	Hz = astro.Hubble(param, 1).to('km/Mpc/s').value #121.0963352 km / (Mpc s)
	assert np.abs(Hz-121.1)<0.1 