import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed
from tqdm import tqdm
import corner

class ImportanceSampling:
    """
    Simple Importance Sampling for estimating expectations with MCMC.
    """
    def __init__(self, target_distribution, proposal_distribution=None, proposal_sampler=None, n_jobs=1, proposal_mean=None, proposal_cov=None):
        """
        Initialize the Importance Sampling class.
        
        Parameters:
        - target_distribution (function): The target probability distribution (unnormalized) from which we want to sample.
        - proposal_distribution (function, optional): The proposal probability distribution we can sample from. Default is a Gaussian distribution.
        - proposal_sampler (function, optional): A function that generates samples from the proposal distribution. Default is a Gaussian sampler.
        - n_jobs (int): Number of parallel jobs for sampling. Default is 1.
        - proposal_mean (array-like, optional): The mean of the Gaussian proposal distribution. Default is a zero vector.
        - proposal_cov (array-like, optional): The covariance matrix of the Gaussian proposal distribution. Default is the identity matrix.
        """
        self.target_distribution = target_distribution
        self.n_jobs = n_jobs
        
        # Set up default proposal distribution as Gaussian if not provided
        self.proposal_mean = proposal_mean if proposal_mean is not None else np.zeros(2)
        self.proposal_cov = proposal_cov if proposal_cov is not None else np.eye(2)
        
        if proposal_distribution is None:
            self.proposal_distribution = self._gaussian_proposal_distribution
        else:
            self.proposal_distribution = proposal_distribution
        
        if proposal_sampler is None:
            self.proposal_sampler = self._gaussian_proposal_sampler
        else:
            self.proposal_sampler = proposal_sampler

    def _gaussian_proposal_sampler(self, n_samples):
        """
        Default Gaussian proposal sampler.
        
        Parameters:
        - n_samples (int): The number of samples to generate.
        
        Returns:
        - samples (np.ndarray): The generated samples from a Gaussian distribution.
        """
        return np.random.multivariate_normal(self.proposal_mean, self.proposal_cov, size=n_samples)
    
    def _gaussian_proposal_distribution(self, x):
        """
        Default Gaussian proposal distribution (pdf).
        
        Parameters:
        - x (np.ndarray): The points at which to evaluate the proposal distribution.
        
        Returns:
        - prob (np.ndarray): The probability density of the points under the Gaussian distribution.
        """
        size = len(self.proposal_mean)
        det = np.linalg.det(self.proposal_cov)
        norm_const = 1.0 / (np.power((2 * np.pi), float(size) / 2) * np.power(det, 1.0 / 2))
        x_mu = x - self.proposal_mean
        inv = np.linalg.inv(self.proposal_cov)
        result = np.einsum('...k,kl,...l->...', x_mu, inv, x_mu)
        return norm_const * np.exp(-0.5 * result)

    def sample(self, n_samples=10000):
        samples = self.proposal_sampler(n_samples)
        target_probs = self.target_distribution(samples)
        proposal_probs = self.proposal_distribution(samples)

        # Compute the importance weights
        weights = target_probs / proposal_probs
        weights /= np.sum(weights)  # Normalize weights to sum to 1

        return samples, weights

    def estimate_expectation(self, function):
        samples, weights = self.sample()
        expectations = function(samples)
        return np.sum(weights * expectations)

class MetropolisHastings:
    '''
    Metropolis-Hastings Algorithm for Monte Carlo Markov Chains.
    '''
    def __init__(self, nwalkers, ndim, log_probability, 
                 proposal='Normal', proposal_cov=None, n_jobs=1):
        """
        Initialize the Metropolis-Hastings sampler.

        Parameters:
        - nwalkers (int): Number of walkers (independent chains) to run in parallel.
        - ndim (int): Number of dimensions in the parameter space.
        - log_probability (callable): The log-probability function to be sampled. 
        This function should take a parameter array of shape (ndim,) and return the log-probability.
        - proposal (str, optional): The type of proposal distribution. Currently, only 'Normal' 
        is supported. Default is 'Normal'.
        - proposal_cov (ndarray, optional): The covariance matrix of the proposal distribution.
        If None, a default covariance matrix must be set later, or adaptive covariance will be used.
        - n_jobs (int, optional): The number of parallel jobs to run. Default is 1, which runs 
        the chains sequentially. Setting this to a higher number can speed up the sampling process.

        Attributes:
        - proposal (str): The type of proposal distribution.
        - proposal_cov (ndarray): The covariance matrix for the proposal distribution.
        - n_jobs (int): The number of parallel jobs to run.
        - nwalkers (int): Number of independent chains to run.
        - ndim (int): Dimensionality of the parameter space.
        - log_probability (callable): The log-probability function used in sampling.
        - samples (ndarray): Array to store the samples generated by the MCMC sampler. Initialized 
        with zeros and will be filled as sampling progresses.

        Raises:
        - AssertionError: If the proposal distribution is not 'Normal'.
        """
        assert proposal.lower() == 'normal', "Currently only 'Normal' proposal is supported."
        self.proposal = proposal
        self.proposal_cov = proposal_cov
        self.n_jobs = n_jobs
        self.nwalkers = nwalkers
        self.ndim = ndim
        self.log_probability = log_probability

        # Initialize samples
        self.samples = np.zeros((nwalkers, 1, ndim))
    
    def draw_from_proposal(self, mean, cov, size=None):
        if self.proposal.lower() == 'normal':
            next_samples = np.random.multivariate_normal(mean, cov, size=size)
            q_next_mean = 1  # Symmetric proposal, so q(next | mean) = q(mean | next)
            q_mean_next = 1
        else:
            raise NotImplementedError("Proposal type not supported.")
        return next_samples, q_next_mean, q_mean_next
    
    def acceptance_probability(self, mean, next_sample, q_next_mean, q_mean_next):
        logp_next = self.log_probability(next_sample)
        logp_mean = self.log_probability(mean)
        np.seterr(over='ignore')
        r = np.exp(logp_next-logp_mean, dtype=np.float64)*q_mean_next/q_next_mean
        np.seterr(over='warn')
        A = min(1, r)
        return A

    def adapt_proposal_covariance(self, n_step, adapt_window=100, scale_factor=100):
        if adapt_window is None:
            return self.proposal_cov
        
        if n_step % adapt_window != 0 or n_step == 0:
            return self.proposal_cov

        # Flatten samples for covariance estimation
        samples_flattened = self.samples[:, -adapt_window:, :].reshape(-1, self.ndim)
        if len(samples_flattened) == 0:
            return self.proposal_cov
        
        cov_matrix = np.cov(samples_flattened, rowvar=False)
        self.proposal_cov = cov_matrix*scale_factor
        return self.proposal_cov
    
    def initialise(self, initial_value):
        n_start = self.samples.shape[1]
        if initial_value is None and n_start == 1:
            print('Provide initial position values for the walkers.')
            return None
        elif initial_value is not None:
            self.samples[:, 0, :] = initial_value

    def _walk(self, walker_num, proposal_cov):
        mean = self.samples[walker_num, -1, :]
        next_sample, q_next_mean, q_mean_next = self.draw_from_proposal(mean, proposal_cov)
        p_acc = self.acceptance_probability(mean, next_sample, q_next_mean, q_mean_next)
        u_acc = np.random.uniform(0, 1)
        if u_acc <= p_acc:
            new_samples = next_sample
        else:
            new_samples = mean
        # print(new_samples, u_acc, p_acc)
        return np.array(new_samples)
    
    def walk(self, proposal_cov):
        backend = 'threading'  # 'loky' can be used for better performance in some cases
        new_samples = Parallel(n_jobs=self.n_jobs, backend=backend)(
                    delayed(self._walk)(walker_num, proposal_cov) for walker_num in range(self.nwalkers)
                    )
        # new_samples = np.array([self.walk(walker_num, proposal_cov) for walker_num in range(self.nwalkers)])
        self.samples = np.concatenate((self.samples, np.array(new_samples)[:, None, :]), axis=1)
        return new_samples
    
    def run_sampler(self, n_samples, initial_value=None, adapt_window=None):
        n_start = self.samples.shape[1]
        self.initialise(initial_value)
        
        for n_step in tqdm(range(n_start, n_samples+1)):
            if n_step==n_start:
                pass
            else:
                proposal_cov = self.adapt_proposal_covariance(n_step, adapt_window=adapt_window)
                new_samples = self.walk(proposal_cov)
                # self.samples = np.concatenate((self.samples, np.array(new_samples)[:, None, :]), axis=1)

        return self.samples
    
    def get_flat_samples(self, burn_in=0):
        samples_flattened = self.samples[:,burn_in:,:].reshape(-1, self.ndim)
        return samples_flattened
    

def plot_posterior_corner(flat_samples, labels, truths=None, weights=None, confidence_levels=None, smooth=0.5, smooth1d=0.5):
    """
    Plots the posterior distributions with corner plots, including 1-sigma and 2-sigma contours.

    Parameters:
    - flat_samples (dict or array-like): Dictionary with dataset names as keys and MCMC samples as values, or just the samples array if only one dataset is present.
    - labels (list of str, optional): Labels for the parameters.
    - truths (array-like, optional): True parameter values for comparison.
    - weights (array-like, optional): Weights for the samples.
    - confidence_levels (list of float, optional): Confidence levels for the contours (e.g., [0.68, 0.95, 0.99]).
    - smooth (float, optional): Smoothing factor for 2D contours. Default is 0.5.
    - smooth1d (float, optional): Smoothing factor for 1D histograms. Default is 0.5.

    Returns:
    - None
    """
    if not isinstance(flat_samples, dict):
        flat_samples = {None: flat_samples}
    if confidence_levels is None:
        confidence_levels = [0.68, 0.95, 0.99]  # Default to 1-sigma, 2-sigma and 3-sigma levels
    fig = None
    for ii, name in enumerate(flat_samples.keys()):
        flats = flat_samples[name]
        print(flats.shape)
        fig = corner.corner(
            flats,
            weights=weights,
            fig=fig,
            labels=labels,
            truths=truths,
            levels=confidence_levels,
            smooth=smooth,
            smooth1d=smooth1d,
            plot_density=True,
            plot_contours=True,
            fill_contours=True,
            show_titles=True,
            title_fmt=".2f",
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 14},
            truth_color="black",
            color=f'C{ii}',
            contour_kwargs={"linewidths": 1.5},
            hist_kwargs={"color": f'C{ii}'},
        )
        # Add dataset name to the plots if provided
        if name is not None:
            ax = fig.axes[flats.shape[1]-1]
            ax.text(0.95, 0.95-ii*0.1, name, transform=ax.transAxes,
                    fontsize=14, verticalalignment='top', horizontalalignment='right')
    plt.show()

if __name__ == '__main__':
    import AstronomyCalc
    import numpy as np
    import matplotlib.pyplot as plt

    true_m = 2
    true_b = 1
    x_obs, y_obs, y_err = AstronomyCalc.line_data(true_m=true_m, true_b=true_b, sigma=2, error_sigma=0.5).T

    def model(param):
        m, b = param
        return m*x_obs+b
    
    true_param = [true_m,true_b]
    plt.title('Observation')
    plt.errorbar(x_obs, y_obs, yerr=y_err, color='k', ls=' ', marker='o')
    plt.plot(x_obs, model(true_param), color='r', ls='--')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    def log_likelihood(param):
        y_mod = model(param)
        # y_err = 0.5*y_obs
        logL = -np.sum((y_obs-y_mod)**2/2/y_err**2)
        return logL
    
    mins = np.array([-2,-3])
    maxs = np.array([6,5])
    def log_prior(param):
        m, b = param
        if mins[0]<=m<=maxs[0] and mins[1]<=b<=maxs[1]:
            return 0
        else:
            return -np.inf

    def log_probability(param):
        lp = log_prior(param)
        if np.isfinite(lp):
            return lp+log_likelihood(param)
        return -np.inf

    print(log_probability(true_param))
    print(log_probability([0,1]))
    print(log_probability([10,1]))

    # Example usage
    nwalkers = 16
    ndim = 2
    n_samples = 2000
    initial_value = np.random.uniform(size=(nwalkers, ndim))*(maxs-mins)+mins

    mh_sampler = MetropolisHastings(nwalkers=nwalkers, ndim=ndim, log_probability=log_probability, 
                            proposal_cov=np.eye(ndim), n_jobs=2)
    samples = mh_sampler.run_sampler(n_samples=n_samples, initial_value=initial_value)
    print("Samples shape:", samples.shape)
    flat_samples = mh_sampler.get_flat_samples(burn_in=100)
    print('Flat samples shape:', flat_samples.shape)

    labels = ['$m$', '$b$']

    fig, axs = plt.subplots(ndim, 1, figsize=(5,ndim*3))
    for i in range(ndim):
        axs[i].set_ylabel(labels[i], fontsize=16)
        axs[i].set_xlabel('step number', fontsize=16)
        for j in range(nwalkers):
            axs[i].plot(samples[j,:,i], c=f'C{j}')
    plt.tight_layout()
    plt.show()

    plot_posterior_corner(flat_samples, labels, truths=true_param)

    def target_distribution(param):
        if np.array(param).ndim==1: return np.exp(log_probability(param))
        else: return np.array([np.exp(log_probability(par)) for par in param])
    
    print(target_distribution(true_param))
    print(target_distribution([0,1]))
    print(target_distribution([10,1]))

    # Set up the proposal distribution mean and covariance
    proposal_mean = [2.5, 1.5]
    proposal_cov = [[2.0, 1.5], [1.5, 2.0]]

    # Initialize the Importance Sampling with default Gaussian proposal
    is_sampler = ImportanceSampling(target_distribution, proposal_mean=proposal_mean, proposal_cov=proposal_cov)

    # Draw samples and compute expectation
    samples, weights = is_sampler.sample(n_samples=1000)

    # Plot samples
    plt.scatter(samples[:, 0], samples[:, 1], c=weights, cmap='viridis', alpha=0.2)
    plt.scatter(true_m, true_b, marker='X', color='red')
    plt.xlabel('m')
    plt.ylabel('b')
    plt.colorbar(label='Importance Weights')
    plt.show()

    plot_posterior_corner(samples, labels, truths=true_param, weights=weights)

