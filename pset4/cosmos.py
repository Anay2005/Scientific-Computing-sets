import numpy as np
import matplotlib.pyplot as plt
import csv
import os

def load_hubble_data(fn="hubble-data.csv"):
    """
    Loads Hubble parameter measurements from a CSV file.

    The file's first line is a header (ignored). Each subsequent line is:
        z_n, H_n, sigma_n
    where:
      z_n    : Redshift (dimensionless)
      H_n    : Hubble parameter at z_n [km/s/Mpc]
      sigma_n: Uncertainty in H_n [km/s/Mpc]

    Parameters
    ----------
    fn : str
        Filename of the CSV file (default "hubble-data.csv").

    Returns
    -------
    z    : 1D numpy array of redshifts
    H    : 1D numpy array of Hubble parameter measurements [km/s/Mpc]
    sigma: 1D numpy array of uncertainties [km/s/Mpc]
    """
    # Sanity check: Ensure the filename is a string and file exists.
    if not isinstance(fn, str):
        raise ValueError("The filename must be a string.")
    if not os.path.exists(fn):
        raise FileNotFoundError(f"File '{fn}' does not exist.")

    z_list = []
    H_list = []
    sigma_list = []

    with open(fn, "r") as f:
        reader = csv.reader(f)
        first_line = True
        for row in reader:
            if first_line:
                # Ignore the header line
                first_line = False
                continue
            if len(row) < 3:
                # Skip empty lines or malformed lines
                continue
            # Parse z, H, sigma from the CSV
            try:
                z_val     = float(row[0])
                H_val     = float(row[1])
                sigma_val = float(row[2])
            except ValueError:
                # Skip rows with invalid numerical values.
                continue
            z_list.append(z_val)
            H_list.append(H_val)
            sigma_list.append(sigma_val)

    if not z_list or not H_list or not sigma_list:
        raise ValueError("No valid data was loaded from the file.")

    z = np.array(z_list)
    H = np.array(H_list)
    sigma = np.array(sigma_list)
    return z, H, sigma


def Hubble_LCDM(z, H0, Om, OL):
    """
    The theoretical Hubble parameter H(z) under the Lambda-CDM model.

    H(z) = H0 * sqrt( Om*(1+z)^3 + OL + (1 - Om - OL)*(1+z)^2 )

    where:
      H0: Hubble constant [km/s/Mpc]
      Om: Matter density parameter, Omega_m
      OL: Dark energy density parameter, Omega_Lambda

    z is dimensionless redshift.
    """
    # Sanity checks for inputs.
    if not isinstance(z, np.ndarray):
        try:
            z = np.array(z)
        except Exception as e:
            raise ValueError("z must be convertible to a numpy array.") from e
    if not isinstance(H0, (int, float)) or H0 <= 0:
        raise ValueError("H0 must be a positive number.")
    if not isinstance(Om, (int, float)):
        raise ValueError("Om must be a number.")
    if not isinstance(OL, (int, float)):
        raise ValueError("OL must be a number.")

    # Compute the dimensionless Hubble parameter squared.
    Ez_sq = Om*(1+z)**3 + OL + (1 - Om - OL)*(1+z)**2
    # Sanity check: Ensure the argument of the square root is non-negative.
    if np.any(Ez_sq < 0):
        raise ValueError("Encountered negative values under the square root; check parameter choices.")
    return H0 * np.sqrt(Ez_sq)


def log_likelihood(z, H_data, sigma_data, H0, Om, OL):
    """
    Computes the log-likelihood for the data given (H0, Om, OL).

    We assume Gaussian errors:
        L = product over n of [ 1/(sqrt(2*pi)*sigma_n) * exp( - (H(z_n) - H_n)^2 / (2 sigma_n^2) ) ]

    so log L = - sum over n of [ (H(z_n) - H_n)^2 / (2 sigma_n^2 ) + log(sigma_n) + const ]

    We'll drop constants that don't depend on parameters, so:
    log L ~ - sum [ (H(z_n) - H_n)^2 / (2 sigma_n^2 ) + log(sigma_n) ]

    but for Metropolis, the difference matters, so we can keep terms that matter or just keep
    the main sum. We'll keep a version ignoring constants that do not depend on (H0, Om, OL).
    """
    # Sanity checks for input data arrays.
    if not (len(z) == len(H_data) == len(sigma_data)):
        raise ValueError("Input arrays z, H_data, and sigma_data must have the same length.")
    if np.any(np.array(sigma_data) <= 0):
        raise ValueError("All uncertainty values (sigma_data) must be positive.")

    H_model = Hubble_LCDM(z, H0, Om, OL)
    # Chi-square like term
    chi2 = np.sum(((H_model - H_data)**2) / (2.0 * sigma_data**2))
    # Also include the log(sigma_n) terms for completeness
    ln_sig = np.sum(np.log(sigma_data))

    # We'll define log likelihood as negative of the "energy" function used in Metropolis:
    # E = chi2 + ln_sig.  The actual log-likelihood ~ -(chi2 + ln_sig).
    # We can skip constants that don't depend on parameters.
    return -(chi2 + ln_sig)


def log_prior(H0, Om, OL):
    """
    A simple log-prior for (H0, Om, OL).

    We'll assume:
      - H0 is in [20, 150], uniform
      - Om is in [0, 1], uniform
      - OL is in [0, 1], uniform
      - Additionally, we might want Om + OL <= 1.3 or something like that to allow for curvature
        but not too large. This is optional.
    """
    # Sanity checks for parameter types.
    if not isinstance(H0, (int, float)):
        raise ValueError("H0 must be a number.")
    if not isinstance(Om, (int, float)):
        raise ValueError("Om must be a number.")
    if not isinstance(OL, (int, float)):
        raise ValueError("OL must be a number.")

    if (20 <= H0 <= 150) and (0 <= Om <= 1) and (0 <= OL <= 1):
        # Flat prior in that region
        return 0.0  # log(1)
    # Otherwise, prior is zero => log-prior = -inf
    return -np.inf


def LCDM_mc(z, H_data, sigma_data, nsteps=20000):
    """
    Performs a Markov Chain Monte Carlo run for the Lambda-CDM parameters:
      X = (H0, Om, OL).

    Parameters
    ----------
    z, H_data, sigma_data : array
        Observational data.
    nsteps : int
        Number of MCMC steps.

    Returns
    -------
    chain_H0 : 1D array of size nsteps
    chain_Om : 1D array of size nsteps
    chain_OL : 1D array of size nsteps
    """
    # Sanity checks for input data and parameters.
    if not (len(z) == len(H_data) == len(sigma_data)):
        raise ValueError("Input arrays z, H_data, and sigma_data must have the same length.")
    if not isinstance(nsteps, int) or nsteps <= 0:
        raise ValueError("nsteps must be a positive integer.")

    # Initialize the chain arrays.
    chain_H0 = np.zeros(nsteps)
    chain_Om = np.zeros(nsteps)
    chain_OL = np.zeros(nsteps)

    # Choose an initial guess for (H0, Om, OL).
    # e.g. H0=70, Om=0.3, OL=0.7, which is a typical guess
    H0_curr = 70.0
    Om_curr = 0.3
    OL_curr = 0.7

    # Evaluate log-posterior = log-likelihood + log-prior
    logL_curr = log_likelihood(z, H_data, sigma_data, H0_curr, Om_curr, OL_curr)
    logP_curr = logL_curr + log_prior(H0_curr, Om_curr, OL_curr)
    if not np.isfinite(logP_curr):
        # If this is infinite, we might pick new initial guess
        H0_curr, Om_curr, OL_curr = 70.0, 0.3, 0.6
        logL_curr = log_likelihood(z, H_data, sigma_data, H0_curr, Om_curr, OL_curr)
        logP_curr = logL_curr + log_prior(H0_curr, Om_curr, OL_curr)

    # For each step, propose a new set of parameters from a small random jump.
    for i in range(nsteps):
        chain_H0[i] = H0_curr
        chain_Om[i] = Om_curr
        chain_OL[i] = OL_curr

        # Propose new parameters by random walk
        # Adjust step sizes as needed for decent acceptance rate
        H0_new = H0_curr + np.random.normal(0, 1.0)  # e.g. stdev=1
        Om_new = Om_curr + np.random.normal(0, 0.02)
        OL_new = OL_curr + np.random.normal(0, 0.02)

        # Compute new log-posterior
        logL_new = log_likelihood(z, H_data, sigma_data, H0_new, Om_new, OL_new)
        logP_new = logL_new + log_prior(H0_new, Om_new, OL_new)

        # Metropolis acceptance
        dlogP = logP_new - logP_curr
        if np.isfinite(logP_new) and (dlogP >= 0 or np.random.rand() < np.exp(dlogP)):
            # Accept
            H0_curr, Om_curr, OL_curr = H0_new, Om_new, OL_new
            logL_curr, logP_curr = logL_new, logP_new
        else:
            # Reject, remain at old values
            pass

    return chain_H0, chain_Om, chain_OL


def LCDM_estimates():
    """
    Loads the data, runs MCMC, and produces:
      - Histograms of H0, Om, OL
      - 2D heat maps (joint distributions) of (Om,OL) and (H0,Om).

    Also prints out best-fit estimates (mean +/- std).
    """
    # Load data
    z, H_data, sigma_data = load_hubble_data("hubble-data.csv")
    # Sanity check: Ensure that data arrays are not empty.
    if len(z) == 0 or len(H_data) == 0 or len(sigma_data) == 0:
        raise ValueError("Loaded data arrays are empty. Check the input file.")

    # Run MCMC
    nsteps = 30000
    chain_H0, chain_Om, chain_OL = LCDM_mc(z, H_data, sigma_data, nsteps=nsteps)

    # Discard "burn-in"
    burnin = 5000
    H0_chain = chain_H0[burnin:]
    Om_chain = chain_Om[burnin:]
    OL_chain = chain_OL[burnin:]

    # Print estimates (means and stdev)
    H0_mean = np.mean(H0_chain)
    H0_std  = np.std(H0_chain)
    Om_mean = np.mean(Om_chain)
    Om_std  = np.std(Om_chain)
    OL_mean = np.mean(OL_chain)
    OL_std  = np.std(OL_chain)

    print(f"Estimated H0 = {H0_mean:.2f} +/- {H0_std:.2f}  (km/s/Mpc)")
    print(f"Estimated Om = {Om_mean:.3f} +/- {Om_std:.3f}")
    print(f"Estimated OL = {OL_mean:.3f} +/- {OL_std:.3f}")

    # Histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # H0
    axes[0].hist(H0_chain, bins=30, color='b', alpha=0.7)
    axes[0].set_xlabel("H0 (km/s/Mpc)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"H0 Distribution\nmean={H0_mean:.2f}, std={H0_std:.2f}")
 
    # Om
    axes[1].hist(Om_chain, bins=30, color='g', alpha=0.7)
    axes[1].set_xlabel("Omega_m")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Omega_m Distribution\nmean={Om_mean:.3f}, std={Om_std:.3f}")
    # OL
    axes[2].hist(OL_chain, bins=30, color='r', alpha=0.7)
    axes[2].set_xlabel("Omega_Lambda")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title(f"Omega_Lambda Distribution\nmean={OL_mean:.3f}, std={OL_std:.3f}")

    plt.tight_layout()
    plt.show()

    # 2D Heat Maps
    # (Om, OL)
    plt.figure(figsize=(5, 4))
    plt.hist2d(Om_chain, OL_chain, bins=40, cmap='plasma')
    plt.colorbar(label="Counts")
    plt.xlabel("Omega_m")
    plt.ylabel("Omega_Lambda")
    plt.title("Joint distribution: (Omega_m, Omega_Lambda)")
    plt.tight_layout()
    plt.show()

    # (H0, Om)
    plt.figure(figsize=(5, 4))
    plt.hist2d(H0_chain, Om_chain, bins=40, cmap='plasma')
    plt.colorbar(label="Counts")
    plt.xlabel("H0 (km/s/Mpc)")
    plt.ylabel("Omega_m")
    plt.title("Joint distribution: (H0, Omega_m)")
    plt.tight_layout()
    plt.show()


def main():
    # Sanity check: Ensure the required data file exists before proceeding.
    if not os.path.exists("hubble-data.csv"):
        raise FileNotFoundError("The file 'hubble-data.csv' does not exist in the working directory.")
    LCDM_estimates()


if __name__ == "__main__":
    main()
