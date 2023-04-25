# MCMC Analysis of the high-l TT data Planck 2018 

Authors: Richard Qiu, Alice Zhang

We have written a Metropolis-Hastings code in Python which analyzes CMB high multipole (l > 30) temperature power spectra data from the Planck Collaboration assuming a Lambda-CDM cosmology. We use the publicly available data release from the Planck collaboration as well as the Planck likelihood code. We use CAMB (Code for Anisotropies in the Microwave Background) as a Boltzmann solver. 


## Installation

Our code assumes the [2018 Planck likelihood code and python wrapper](https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_spectrum_%26_Likelihood_Code#2018_Likelihood) are installed, as well as the following Python libraries: 
- camb
- numpy
- matplotlib
- seaborn
- pandas
- statsmodels
- corner
- jupyter

Parts of the code also utilze the 2018 Planck spectra, which can be found [here](https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_spectrum_%26_Likelihood_Code#2018_CMB_spectra). 


## Running the Code

Our primary Metropolis-Hastings code can be run as follows: 
```py
python planck_mcmc.py
```
By default, we run 5 chains for 10,000 steps each with a burn-rate of 10% and a no thinning, but these parameters are easily changed within `planck_mcmc.py`. The core of the MCMC code can be found in `planck_utilities.py`. In particular, it may be of interest to edit the proposal distribution covariance matrix `proposal_sigma_sq`. 

Analysis of the chains is contained in `Jupyter` notebooks. In particular, `planck_plotting.ipynb` contains code for:
- Running MCMC chain diagnostics (traceplots, autocorrelation plots, computing Gelman-Rubin statistics),
- Thinning and combining chains,
- Creating corner plots,
- Computing the best fit and 68% constraint parameter ranges,
- Plotting the best fit temperature spectra against the binned Planck 2018 data with error bars.  
