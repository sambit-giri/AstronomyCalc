import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed
from tqdm import tqdm
import corner

class MetropolisHastings:
    '''
    Metropolis-Hastings Algorithm for Monte Carlo Markov Chains.
    '''
    def __init__(self, nwalkers, ndim, log_probability, 
                 proposal='Normal', proposal_cov=None, n_jobs=1):
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
    

def plot_posterior_corner(flat_samples, labels, truths=None, 
                   confidence_levels=None, smooth=0.5, smooth1d=0.5):
    """
    Plots the posterior distributions with corner plots including 1-sigma and 2-sigma contours.

    Parameters:
    - flat_samples (dict or array-like): Dictionary with dataset names as keys and MCMC samples as values,
      or just the samples array if only one dataset is present.
    - labels (list of str): Labels for the parameters.
    - truths (array-like): True parameter values for comparison.
    - confidence_levels (list of float): Confidence levels for the contours (e.g., [0.68, 0.95]).
    - smooth (float): Smoothing factor for 2D contours.
    - smooth1d (float): Smoothing factor for 1D histograms.

    Returns:
    - None
    """
    if not isinstance(flat_samples, dict):
        flat_samples = {None: flat_samples}
    
    if confidence_levels is None:
        confidence_levels = [0.68, 0.95, 0.99]  # Default to 1-sigma and 2-sigma levels

    fig = None
    for ii, name in enumerate(flat_samples.keys()):
        flats = flat_samples[name]
        print(flats.shape)
        fig = corner.corner(
            flats,
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
    x_obs, y_obs = AstronomyCalc.create_data.line_data(true_m=true_m, true_b=true_b, sigma=2).T

    def model(param):
        m, b = param
        return m*x_obs+b
    
    true_param = [true_m,true_b]
    plt.title('Observation')
    plt.scatter(x_obs, y_obs, color='k')
    plt.plot(x_obs, model(true_param), color='k')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    def log_likelihood(param):
        y_mod = model(param)
        y_err = 0.5*y_obs
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
    nwalkers = 8
    ndim = 2
    n_samples = 1000
    initial_value = np.random.uniform(size=(nwalkers, ndim))*(maxs-mins)+mins

    mh = MetropolisHastings(nwalkers=nwalkers, ndim=ndim, log_probability=log_probability, 
                            proposal_cov=np.eye(ndim), n_jobs=2)
    samples = mh.run_sampler(n_samples=n_samples, initial_value=initial_value)
    print("Samples shape:", samples.shape)
    flat_samples = mh.get_flat_samples(burn_in=10)
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

