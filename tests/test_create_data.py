import numpy as np 
import AstronomyCalc as astro

def test_line_data():
    xobs, yobs = astro.line_data(true_m=2, true_b=1, sigma=0.1, n_samples=100).T
    m_mean = np.mean((yobs[1:]-1)/(xobs[1:]))
    assert np.abs(m_mean-2)<0.1

def test_distance_modulus_data():
    param = astro.param()
    z_sample, mu_sample, dmu = astro.generate_distance_modulus(n_samples=100, z0=0.3, dmu_0=0.1, dmu_1=0.02)
    mu_model = astro.distance_modulus(param, z_sample)
    assert np.all(np.abs(mu_sample-mu_model)<dmu*3)

# def test_Hubble1929_data():
#     dists, vels = astro.Hubble1929_data()
#     vels_model = 500*dists #H0 = 500 km/s/Mpc
#     assert np.all(np.abs(vels_model-vels)<vels.std()*3)

# def test_PantheonPlus_distance_modulus():
#     param = astro.param()
#     data_dict = astro.PantheonPlus_distance_modulus()
#     z_data = data_dict['z']
#     mu_mean = data_dict['data']
#     mu_std = np.sqrt(data_dict['cov'].diagonal())
#     mu_model = astro.distance_modulus(param, z_data)-19.1
#     assert np.all(np.abs(mu_mean-mu_model)<mu_std*9)

# def test_SPARC_galaxy_rotation_curves_data():
#     data = astro.SPARC_galaxy_rotation_curves_data(name='NGC3198')
#     Rad, Vobs = data['values']['Rad'], data['values']['Vobs']
#     assert np.all(Vobs<160.) and np.all(0.<Rad) and np.all(Rad<45.)