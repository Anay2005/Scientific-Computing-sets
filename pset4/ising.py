import numpy as np
import matplotlib.pyplot as plt

def ising_mc(J=1.0, Nx=16, Ny=16, nsteps=50000):
    """
    Monte Carlo simulation of the 2D Ising model using the Metropolis algorithm.
    
    Parameters
    ----------
    J : float
        Coupling constant (should be > 0).
    Nx, Ny : int
        Number of sites in the x and y directions.
    nsteps : int
        Total number of Monte Carlo steps per temperature.
        
    The simulation runs over a range of temperatures T, from T_min = 1/J to T_max = 3.5/J.
    The Boltzmann constant k_B is set to 1.
    
    The function produces two plots:
      1. Average energy per spin vs Temperature.
      2. Average absolute magnetization per spin vs Temperature.
    
    Returns
    -------
    Ts : np.ndarray
        Array of temperatures.
    energy_avgs : np.ndarray
        Average energy per spin at each temperature.
    mag_avgs : np.ndarray
        Average absolute magnetization per spin at each temperature.
    """
    
    # --- Sanity Checks ---
    if not (isinstance(J, (int, float)) and J > 0):
        raise ValueError("J must be a positive number.")
    if not (isinstance(Nx, int) and Nx > 0):
        raise ValueError("Nx must be a positive integer.")
    if not (isinstance(Ny, int) and Ny > 0):
        raise ValueError("Ny must be a positive integer.")
    if not (isinstance(nsteps, int) and nsteps > 0):
        raise ValueError("nsteps must be a positive integer.")
    
    # Temperature Range
    T_min = 1.0 / J
    T_max = 3.5 / J
    nT = 30  # Number of temperature points.
    Ts = np.linspace(T_min, T_max, nT)
    
    # Arrays to store average energy and magnetization ---
    energy_avgs = np.zeros(nT)
    mag_avgs = np.zeros(nT)
    
    N = Nx * Ny  # Total number of spins.
    
    def compute_energy(spins):
        # Compute total energy: E = -J sum_<ij> S_i S_j (each neighbor counted once)
        E = 0.0
        E -= J * np.sum(spins * np.roll(spins, 1, axis=0))  # vertical neighbors
        E -= J * np.sum(spins * np.roll(spins, 1, axis=1))  # horizontal neighbors
        return E

    def compute_magnetization(spins):
        # Total magnetization per spin.
        return np.sum(spins) / N
    
    # --- Loop over Temperatures ---
    for idx, T in enumerate(Ts):
        beta = 1.0 / T
        
        # Initialize spins randomly to +1 or -1.
        spins = np.random.choice([1, -1], size=(Nx, Ny))
        
        # Equilibration phase (first 20% of steps)
        n_eq = int(0.2 * nsteps)
        for step in range(n_eq):
            i = np.random.randint(0, Nx)
            j = np.random.randint(0, Ny)
            s = spins[i, j]
            # Periodic boundary conditions: use np.roll (here, done manually)
            s_up    = spins[(i-1) % Nx, j]
            s_down  = spins[(i+1) % Nx, j]
            s_left  = spins[i, (j-1) % Ny]
            s_right = spins[i, (j+1) % Ny]
            dE = 2 * J * s * (s_up + s_down + s_left + s_right)
            if dE < 0 or np.random.rand() < np.exp(-beta * dE):
                spins[i, j] = -s
        
        # Measurement phase
        E_sum = 0.0
        M_sum = 0.0
        n_measure = 0
        
        for step in range(n_eq, nsteps):
            i = np.random.randint(0, Nx)
            j = np.random.randint(0, Ny)
            s = spins[i, j]
            s_up    = spins[(i-1) % Nx, j]
            s_down  = spins[(i+1) % Nx, j]
            s_left  = spins[i, (j-1) % Ny]
            s_right = spins[i, (j+1) % Ny]
            dE = 2 * J * s * (s_up + s_down + s_left + s_right)
            if dE < 0 or np.random.rand() < np.exp(-beta * dE):
                spins[i, j] = -s
            
            # Sample one measurement every N spin flips (~one sweep)
            if step % N == 0:
                E_sum += compute_energy(spins)
                M_sum += compute_magnetization(spins)
                n_measure += 1
        
        # Additional sanity check: ensure that at least one measurement was taken.
        if n_measure == 0:
            raise RuntimeError("No measurements were recorded; consider increasing nsteps.")
        
        # Average energy and magnetization per spin.
        energy_avgs[idx] = E_sum / (n_measure * N)
        mag_avgs[idx] = np.abs(M_sum / n_measure)
    
    # --- Plot Results ---
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(Ts, energy_avgs, 'o-', color='b')
    plt.xlabel("Temperature T")
    plt.ylabel("Average energy per spin")
    plt.title("Energy vs Temperature")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(Ts, mag_avgs, 'o-', color='r')
    plt.xlabel("Temperature T")
    plt.ylabel("Average |magnetization|")
    plt.title("Magnetization vs Temperature")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return Ts, energy_avgs, mag_avgs


def ising_mc_reweight(J=1.0, Nx=16, Ny=16, nsteps=50000, T0=2.4):
    """
    Monte Carlo simulation of the 2D Ising model using the Metropolis algorithm,
    with re-weighting to estimate observables (e.g. heat capacity) at various temperatures T1
    from a single Monte Carlo run performed at T0.
    
    Parameters
    ----------
    J : float
        Coupling constant (in energy units). (J > 0 means ferromagnetic coupling.)
    Nx, Ny : int
        Number of sites in x and y directions, respectively.
    nsteps : int
        Total number of Monte Carlo steps.
    T0 : float
        The simulation temperature (in energy units, with k_B = 1). Typical choice is around 2.4.
        
    Returns
    -------
    Ts : np.ndarray
        Array of temperatures (T1 values) over which re-weighting is performed.
    C_reweighted : np.ndarray
        Re-weighted heat capacity per spin at each temperature T1.
    """
    # --- Sanity Checks ---
    if not (J > 0):
        raise ValueError("J must be positive.")
    if not (isinstance(Nx, int) and Nx > 0 and isinstance(Ny, int) and Ny > 0):
        raise ValueError("Nx and Ny must be positive integers.")
    if not (isinstance(nsteps, int) and nsteps > 0):
        raise ValueError("nsteps must be a positive integer.")
    if T0 <= 0:
        raise ValueError("T0 must be positive.")

    N = Nx * Ny  # Total number of spins
    beta0 = 1.0 / T0

    # --- Temperature range for reweighting: T1 from T_min to T_max ---
    T_min = 1.0 / J
    T_max = 3.5 / J
    nT = 40  # number of temperature points for reweighting
    Ts = np.linspace(T_min, T_max, nT)
    
    # Array to store measured energies ---
    energies = []  # total energy at each measurement

    def compute_energy(spins):
        """Compute total energy of the configuration using periodic BC."""
        E = 0.0
        E -= J * np.sum(spins * np.roll(spins, 1, axis=0))  # vertical bonds
        E -= J * np.sum(spins * np.roll(spins, 1, axis=1))  # horizontal bonds
        return E

    # Initialize spins randomly: S = +1 or -1 ---
    spins = np.random.choice([1, -1], size=(Nx, Ny))

    # --- Equilibration Phase: first 20% of steps ---
    n_eq = int(0.2 * nsteps)
    for step in range(n_eq):
        i = np.random.randint(0, Nx)
        j = np.random.randint(0, Ny)
        s = spins[i, j]
        s_up    = spins[(i-1) % Nx, j]
        s_down  = spins[(i+1) % Nx, j]
        s_left  = spins[i, (j-1) % Ny]
        s_right = spins[i, (j+1) % Ny]
        dE = 2 * J * s * (s_up + s_down + s_left + s_right)
        if dE < 0 or np.random.rand() < np.exp(-beta0 * dE):
            spins[i, j] = -s

    # Measurement Phase 
    n_measure = 0
    for step in range(n_eq, nsteps):
        i = np.random.randint(0, Nx)
        j = np.random.randint(0, Ny)
        s = spins[i, j]
        s_up    = spins[(i-1) % Nx, j]
        s_down  = spins[(i+1) % Nx, j]
        s_left  = spins[i, (j-1) % Ny]
        s_right = spins[i, (j+1) % Ny]
        dE = 2 * J * s * (s_up + s_down + s_left + s_right)
        if dE < 0 or np.random.rand() < np.exp(-beta0 * dE):
            spins[i, j] = -s

        # Measure energy every N spin flips (one sweep)
        if step % N == 0:
            E = compute_energy(spins)
            energies.append(E)
            n_measure += 1

    # Additional sanity check: ensure that measurements were recorded.
    if n_measure == 0:
        raise RuntimeError("No measurements were recorded; consider increasing nsteps.")
    
    energies = np.array(energies)  # convert to numpy array

    # Reweighting to compute observables at various T1 ---
    C_reweighted = np.zeros(nT)  # heat capacity per spin at T1
    for idx, T1 in enumerate(Ts):
        beta1 = 1.0 / T1
        # Weight factor for each measurement: exp[-(beta1 - beta0) * E]
        weights = np.exp(-(beta1 - beta0) * energies)
        
        # Sanity check: Ensure the sum of weights is nonzero to avoid division by zero.
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            raise RuntimeError(f"Sum of weights is zero at T1 = {T1}; check simulation parameters.")
        
        # Reweighted averages at T1
        E_avg = np.sum(energies * weights) / weight_sum
        E2_avg = np.sum(energies**2 * weights) / weight_sum
        # Heat capacity per spin: C = (⟨E²⟩ - ⟨E⟩²) / (T1² * N)
        C_reweighted[idx] = (E2_avg - E_avg**2) / (T1**2 * N)
    
    # Plot the Reweighted Heat Capacity 
    plt.figure(figsize=(6, 5))
    plt.plot(Ts, C_reweighted, 'o-', color='m', label="C$_B$ via re-weighting")
    plt.xlabel("Temperature T")
    plt.ylabel("Heat Capacity per Spin, C$_B$")
    plt.title("Reweighted Heat Capacity vs Temperature")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return Ts, C_reweighted

def main():
    # Additional sanity check for main: Ensure that nsteps is sufficiently large for meaningful statistics.
    nsteps = 50000
    if nsteps < 1000:
        raise ValueError("nsteps is too small for a reliable simulation; please use a larger value.")
    
    ising_mc(J=1.0, Nx=16, Ny=16, nsteps=nsteps)
    ising_mc_reweight(J=1.0, Nx=16, Ny=16, nsteps=nsteps, T0=2.4)

if __name__ == "__main__":
    main()
