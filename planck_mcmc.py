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

from planck_utilities import *

run_name = "run6"

if __name__ == "__main__":
    start_time = time.time()

    S=10_000
    init = [0.022068, 0.12029, 1.04122, 0.0925, 0.9624, 3.098]

    samples, log_likelihoods, accept_rates = MH(S, target, init, proposal_distribution, burn=0.1, thin=1, chains=5, n_workers=5)
    print(f"accept rates: {accept_rates}")

    fig, ax = plt.subplots(6, 2, figsize=(12, 15))
    for i in range(6):
        for chain in samples: 
            ax[i, 0].plot(np.arange(chain.shape[0]), chain[:, i], alpha=0.5) # plot the trace plot for x_1
            ax[i, 0].set_xlabel('n-th sample') # set x-axis label
            ax[i, 0].set_ylabel(f'x_{i} value') # set y-axis label
            ax[i, 0].set_title(f'traceplot for x_{i}') # set the title

            plot_acf(chain[:, i], ax=ax[i, 1])

    plt.tight_layout() # layout the subplots nicely
    plt.savefig(f"planck_diagnostics_{run_name}.pdf")

    corner.corner(np.vstack(samples), bins=30)
    plt.savefig(f"planck_corner_{run_name}.pdf")

    data_to_save = {"samples": samples, "log_likelihoods": log_likelihoods, "accept_rates": accept_rates}
    with open(f"planck_chains_{run_name}.pkl", "wb") as f: 
        pickle.dump(data_to_save, f)
        
    print(f"Total running time: {time.time() - start_time}s")