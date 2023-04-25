import time
import multiprocessing as mp
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import corner

def _MH(S, target, init, proposal_distribution, burn=0, thin=1):
    samples = []
    likelihoods = []
    current = init
    accepts = 0
    current_prob = target(current)
    for s in range(S):
        proposal = proposal_distribution(current)
        U = np.random.uniform(0, 1)
        proposal_prob = target(proposal)
        mh_prob = proposal_prob/current_prob
        if U < np.min((1, mh_prob)):
            samples.append(proposal)
            likelihoods.append(proposal_prob)
            current = proposal
            current_prob = proposal_prob
            accepts += 1
        else:
            samples.append(current)
            likelihoods.append(current_prob)
    return np.array(samples)[int(burn * S)::thin], np.array(likelihoods)[int(burn * S)::thin], np.array(accepts) * 1. / S


def MH(S, target, init, proposal_distribution, burn=0, thin=1, chains=1, n_workers=1):
    with mp.Pool(processes=n_workers) as pool:
        results = [
            pool.apply_async(_MH, (S, target, init, proposal_distribution, burn, thin))
            for _ in range(chains)
        ]
        results = [result.get() for result in results]
    return list(zip(*results))


# mu = np.array([120, 4, 1, 17, 50, 8]) # mean of the multivariate Gaussian
# Sigma = 1*np.ones((6, 6)) + 2*np.eye(6) # covariance matrix of the multivariate Gaussian
# M = np.random.rand(6, 6)*2-1
# Sigma = M.T @ M # covariance matrix of the multivariate Gaussian
# Sigma = np.load("sigma.npy") # covariance matrix of the multivariate Gaussian

# print(f"eigenvalues: {np.linalg.eig(Sigma)[0]}")

mu = np.array([120, 4, 1, 17, 50, 8]) 
Sigma = np.load("sigma.npy")
# target = lambda x: sp.stats.multivariate_normal(mu, Sigma**0.5).pdf(x)
def target(x):
    # return sp.stats.multivariate_normal(mu, Sigma).pdf(x)
    # dim = x.shape[0]
    # normalization = ((2*np.pi)**(-dim) * np.linalg.det(Sigma))
    # exp = np.exp(-1/2 * (x - mu).T @ np.linalg.inv(Sigma) @ (x - mu))
    # return normalization * exp
    return 0.5

#define the proposal distribution variance
# proposal_sigma_sq = np.array([[0.4, 0.2],[0.2, 0.4]])
proposal_sigma_sq = 0.1*np.eye(6)

#define the proposal distribution sampler
# proposal_distribution = lambda mean: sp.stats.multivariate_normal.rvs(mean, proposal_sigma_sq**0.5, size=1)
def proposal_distribution(mean): 
    return np.random.multivariate_normal(mean, proposal_sigma_sq)

#define the proposal distribution pdf
# proposal_pdf = lambda x, mean: sp.stats.multivariate_normal(mean, proposal_sigma_sq**0.5).pdf(x)
# def proposal_pdf(x, mean): 
#     # return sp.stats.multivariate_normal(mean, proposal_sigma_sq).pdf(x)
#     dim = x.shape[0]
#     normalization = ((2*np.pi)**(-dim) * np.linalg.det(proposal_sigma_sq))
#     exp = np.exp(-1/2 * (x - mean).T @ np.linalg.inv(proposal_sigma_sq) @ (x - mean))
#     return normalization * exp


if __name__ == "__main__":
    start_time = time.time()

    S=100_000
    # init = mu + np.random.randn(6)
    init = np.array([120, 4, 1, 17, 50, 8])
    # print(mu, init)

    samples, log_likelihoods, accept_rates = MH(S, target, init, proposal_distribution, burn=0.2, thin=20, chains=20, n_workers=20)
    # for _ in range(20):
    #     samples, log_likelihoods, accept_rates = _MH(S, target, np.array(init), proposal_distribution, burn=0.2, thin=20)
    # print(accept_rates)

    # with mp.Pool(processes=20) as pool:
    #     results = [
    #         pool.apply_async(_MH, (S, target, init, proposal_distribution, 0.2, 20))
    #         for _ in range(20)
    #     ]
    #     results = [result.get() for result in results]
    # results = list(zip(*results))

    print(f"running time: {time.time() - start_time}")

    # fig, ax = plt.subplots(6, 2, figsize=(12, 15))
    # for i in range(6):
    #     for chain in samples: 
    #         ax[i, 0].plot(np.arange(chain.shape[0]), chain[:, i], alpha=0.5) # plot the trace plot for x_1
    #         ax[i, 0].set_xlabel('n-th sample') # set x-axis label
    #         ax[i, 0].set_ylabel(f'x_{i} value') # set y-axis label
    #         ax[i, 0].set_title(f'traceplot for x_{i}') # set the title

    #         plot_acf(chain[:, i], ax=ax[i, 1])

    # plt.tight_layout() # layout the subplots nicely
    # plt.savefig("diagnostics.pdf")

    # target_sample = sp.stats.multivariate_normal.rvs(mu, Sigma, size=np.vstack(samples).shape[0])
    # fig = corner.corner(target_sample, bins=30, color="red")
    # corner.corner(np.vstack(samples), bins=30, fig=fig)
    # plt.savefig("corner.pdf")