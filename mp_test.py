import multiprocessing as mp
import time

import numpy as np
import scipy as sp
import clik
import camb


Sigma = np.load("sigma.npy") # covariance matrix of the multivariate Gaussian
mu = np.diag(Sigma)
# def target(x):
#     return sp.stats.multivariate_normal(mu, Sigma).pdf(x)

def proposal(x):
    # return sp.stats.multivariate_normal.rvs(mu, Sigma, size=1)
    return np.random.multivariate_normal(mu, Sigma, size=1)


# CMBlkl = clik.clik("baseline/plc_3.0/hi_l/plik/plik_rd12_HM_v22_TT.clik")
planck_nuisance_params = np.loadtxt("baseline/plc_3.0/hi_l/plik/plik_rd12_HM_v22_TT.clik.cls")[-20:]
def get_camb_spectra(ombh2, omch2, cosmomctheta100, tau, ns, ln1e10As): 
    pars = camb.CAMBparams()
    pars.set_cosmology(ombh2=ombh2, omch2=omch2, cosmomc_theta=cosmomctheta100/100, tau=tau, omk=0)
    pars.InitPower.set_params(As=np.exp(ln1e10As)/1e10, ns=ns, r=0)
    pars.set_for_lmax(2550, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
    return np.hstack([powers['total'][:2509, 0], planck_nuisance_params])

# TT = get_camb_spectra(*[0.022068, 0.12029, 1.04122, 0.0925, 0.9624, 3.098])
    
def target(lkl, TT):
    return lkl(TT)[0]

def f(i): 
    # lkl = clik.clik(f"baseline/plc_3.0/hi_l/plik/plik_rd12_HM_v22_TT{i}.clik")
    # planck_nuisance_params = np.loadtxt("baseline/plc_3.0/hi_l/plik/plik_rd12_HM_v22_TT.clik.cls")[-20:]
    
    # def get_camb_spectra(ombh2, omch2, cosmomctheta100, tau, ns, ln1e10As): 
    #     pars = camb.CAMBparams()
    #     pars.set_cosmology(ombh2=ombh2, omch2=omch2, cosmomc_theta=cosmomctheta100/100, tau=tau, omk=0)
    #     pars.InitPower.set_params(As=np.exp(ln1e10As)/1e10, ns=ns, r=0)
    #     pars.set_for_lmax(2550, lens_potential_accuracy=0)
    #     results = camb.get_results(pars)
    #     powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
    #     return np.hstack([powers['total'][:2509, 0], planck_nuisance_params])

    # TT = get_camb_spectra(*[0.022068, 0.12029, 1.04122, 0.0925, 0.9624, 3.098])

    for _ in range(100): 
        # target(np.random.randn(1))
        # proposal(None)
        # target(lkl, TT)
        get_camb_spectra(*[0.022068, 0.12029, 1.04122, 0.0925, 0.9624, 3.098])

if __name__ == "__main__":
    n = 5

    start_time_mp = time.time()
    with mp.Pool(processes=n) as pool: 
        results = [pool.apply_async(f, (i, )) for i in range(n)]
        results = [result.get() for result in results]

    start_time_seq = time.time()
    for _ in range(n):
        f(0)
    print(f"mp time: {start_time_seq - start_time_mp}")
    print(f"seq time: {time.time() - start_time_seq}")
    
