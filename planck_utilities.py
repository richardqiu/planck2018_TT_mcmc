import multiprocessing as mp
import pickle
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import corner

import camb
import clik


def _MH(S, log_target, init, proposal_distribution, burn=0, thin=1):
    samples = []
    log_likelihoods = []
    current = init
    accepts = 0
    current_log_prob = log_target(*current)
    for s in range(S):
        proposal = proposal_distribution(current)
        U = np.random.uniform(0, 1)
        proposal_log_prob = log_target(*proposal)
        mh_prob = np.exp(proposal_log_prob - current_log_prob)
        if U < np.min((1, mh_prob)):
            samples.append(proposal)
            log_likelihoods.append(proposal_log_prob)
            current = proposal
            current_log_prob = proposal_log_prob
            accepts += 1
        else:
            samples.append(current)
            log_likelihoods.append(current_log_prob)
    return np.array(samples)[int(burn * S)::thin], np.array(log_likelihoods)[int(burn * S)::thin], np.array(accepts) * 1. / S


def MH(S, target, init, proposal_distribution, burn=0, thin=1, chains=1, n_workers=1):
    with mp.Pool(processes=n_workers) as pool:
        results = [
            pool.apply_async(_MH, (S, target, init, proposal_distribution, burn, thin))
            for _ in range(chains)
        ]
        results = [result.get() for result in results]
    return list(zip(*results))


CMBlkl = clik.clik("baseline/plc_3.0/hi_l/plik/plik_rd12_HM_v22_TT.clik")
planck_nuisance_params = np.loadtxt("baseline/plc_3.0/hi_l/plik/plik_rd12_HM_v22_TT.clik.cls")[-20:]

def target(ombh2, omch2, cosmomctheta100, tau, ns, ln1e10As):
    pars = camb.CAMBparams()
    pars.set_cosmology(ombh2=ombh2, omch2=omch2, cosmomc_theta=cosmomctheta100/100, tau=tau, omk=0)
    pars.InitPower.set_params(As=np.exp(ln1e10As)/1e10, ns=ns, r=0)
    pars.set_for_lmax(2550, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
    
    TT = powers['total'][:2509, 0]
    log_likelihood = CMBlkl(np.hstack([TT, planck_nuisance_params])) # plik default
    return log_likelihood[0]

#define the proposal distribution variance
# Order: ombh2, omch2, 100*cosmomctheta, tau, ns, ln(1e10*As) 
# proposal_sigma_sq = 0.005*np.square(np.diag([0.00033, 0.0031, 0.00068, 0.038, 0.0094, 0.072]))
with open("planck_chains_run2_run3.pkl", "rb") as f:
    chains = pickle.load(f)["samples"]
proposal_sigma_sq = np.cov(np.vstack(chains).T)

#define the proposal distribution sampler
# proposal_distribution = lambda mean: sp.stats.multivariate_normal.rvs(mean, proposal_sigma_sq**0.5, size=1)
def proposal_distribution(mean): 
    # return sp.stats.multivariate_normal.rvs(mean, proposal_sigma_sq, size=1)
    return np.random.multivariate_normal(mean, proposal_sigma_sq)
