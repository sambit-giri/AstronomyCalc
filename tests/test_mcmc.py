import numpy as np 
import AstronomyCalc as astro

def test_ImportanceSampling():
    def target_distribution(x):
        return np.exp(-x**2)

    # Set up the proposal distribution mean and covariance
    proposal_mean = [0]
    proposal_cov = [[0.5]]

    # Initialize the Importance Sampling with default Gaussian proposal
    is_sampler = astro.ImportanceSampling(target_distribution, proposal_mean=proposal_mean, proposal_cov=proposal_cov)
    samples, weights = is_sampler.sample(10)
    assert np.abs(np.sum(weights)-1)<0.1

def test_MetropolisHastings():
    def log_probability(x):
        return -x**2
    
    nwalkers = 2
    ndim = 1
    n_samples = 10
    initial_value = np.random.uniform(size=(nwalkers,ndim))

    mh_sampler = astro.MetropolisHastings(nwalkers=nwalkers, ndim=ndim, log_probability=log_probability, 
                            proposal_cov=np.eye(ndim), n_jobs=1)
    samples = mh_sampler.run_sampler(n_samples=n_samples, initial_value=initial_value)
    assert np.abs(np.array(samples.shape)-np.array([nwalkers,n_samples,ndim])).sum()<0.01