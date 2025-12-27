import numpy as np
import matplotlib.pyplot as plt


# seed for reproducibility
np.random.seed(42)
# Fix the number of electrons
N = 100

def chain_hamiltonian(epsn, t):
    """
    Construct the Hamiltonian matrix for a 1D tight-binding chain.
    
    Parameters:
    epsn (1D array): Real on-site energies.
    t (1D array): Nearest-neighbor hopping terms (real or complex).
    
    Returns:
    H (2D array): Complex Hermitian matrix representing the Hamiltonian.
    
    Raises:
    ValueError: If input lengths are incompatible or epsn is not real.
    """
    # Sanity checks
    if len(epsn) != len(t) + 1:
        raise ValueError("Length of epsn must be len(t) + 1")
    if not np.isrealobj(epsn):
        raise ValueError("epsn must be a real array")
    
    N = len(epsn)
    H = np.zeros((N, N), dtype=complex)
    
    # Fill diagonal with on-site energies
    np.fill_diagonal(H, epsn)
    
    # Fill off-diagonals with hopping terms
    for i in range(N-1):
        H[i, i+1] = t[i]
        H[i+1, i] = np.conj(t[i])
    
    return H

def random_chain_study():
    """
        1) We create a chain of N sites.
        2) Each site has an energy 'eps_n', which is set to epsilon_0 * chi[n].
        - chi is a set of random numbers (from a normal distribution).
        - epsilon_0 changes from -5 to +5 to see different scenarios.

        3) We measure the eigenvalues (energy levels) of the Hamiltonian and plot them.
        4) Then we pick three typical values of epsilon_0 (very small, very large, and about 1)
        and plot the ground-state probability distribution |psi_n|^2. 
        This tells us how likely it is to find the electron on each site n.
    """

    # draw chi_array of size N from a normal distribution
    chi_array = np.random.randn(N)

    # generate random on-site energies for the x-axis
    eps0 = np.linspace(-100, 100, N)

    # fixed hopping term with t_0 = 1eV
    t_n = np.full((N-1,), 1, dtype=complex)

    # place to store the eigenvalues
    eigenvalues = []

    # loop over the on-site energies
    for eps in eps0:
        # construct the Hamiltonian
        H = chain_hamiltonian(eps * chi_array, t_n)
        # compute the eigenvalues
        eigvals = np.linalg.eigvalsh(H)
        # append the eigenvalues to the list
        eigenvalues.append(eigvals)

    # convert the eigenvalues to a numpy array
    eigenvalues = np.array(eigenvalues)
    
    # Use the matplotlib library to plot the eigenvalues
    plt.figure()
    plt.plot(eps0, eigenvalues, 'o')
    plt.xlabel(r'$\epsilon_0$')
    plt.ylabel('Eigenvalues')
    plt.title('Eigenvalues vs energy scale')
    plt.show()

    # Now we will plot several mode distributions, \psi_n^2 vs n

    # 1st regime epsilon_0 << t_0
    eps0 = 0.001
    # calculate on-site energies
    epsn = eps0 * chi_array
    # construct the Hamiltonian
    H = chain_hamiltonian(epsn, t_n)
    # compute the eigenvectors
    eigvals, eigvecs = np.linalg.eigh(H)
    # plot the mode distributions
    plt.figure()
    plt.plot(np.abs(eigvecs[:, 0])**2, marker=None, label='Ground state')
    plt.xlabel('Site index')
    plt.ylabel(r'$|\psi_n|^2$')
    plt.title(r'$\epsilon_0 << t_0$')
    plt.legend()
    plt.show()

    # 2nd regime epsilon_0 >> t_0
    eps0 = 1000
    # calculate on-site energies
    epsn = eps0 * chi_array
    # construct the Hamiltonian
    H = chain_hamiltonian(epsn, t_n)
    # compute the eigenvectors
    eigvals, eigvecs = np.linalg.eigh(H)
    # plot the mode distributions
    plt.figure()
    plt.plot(np.abs(eigvecs[:, 0])**2, marker=None, label='Ground state')
    plt.xlabel('Site index')
    plt.ylabel(r'$|\psi_n|^2$')
    plt.title(r'$\epsilon_0 >> t_0$')
    plt.legend()
    plt.show()

    # 3rd regime epsilon_0 ~ t_0
    eps0 = 1
    # calculate on-site energies
    epsn = eps0 * chi_array
    # construct the Hamiltonian
    H = chain_hamiltonian(epsn, t_n)
    # compute the eigenvectors
    eigvals, eigvecs = np.linalg.eigh(H)
    # plot the mode distributions
    plt.figure()
    plt.plot(np.abs(eigvecs[:, 0])**2, marker=None, label='Ground state')
    plt.xlabel('Site index')
    plt.ylabel(r'$|\psi_n|^2$')
    plt.title(r'$\epsilon_0 \approx t_0$')
    plt.legend()
    plt.show()
    """
        Random Chain
            • When \epsilon_0 is small, all the on-site energies are nearly the same, so the electron’s probability of being on each site is more spread out. We get a Gaussian-like distribution but with a low peak probability.
            • When \epsilon_0 is large, each site energy can be very different. The electron will likely stay on a few sites where the energy is most favorable, making the probability distribution look peaked at certain sites and nearly zero at others which is what was observed.
            • When \epsilon_0 is about the same as t_0, the probabilty of observing an electron was peaked near lower energy sites and almost zero at higher energy sites.    
            
            For Band Diagrams:
            • As \epsilon_0 changes from negative to positive, the energy levels “fan out” more if |\epsilon_0| is large. I saw a bow‐tie shape where levels converge near \epsilon_0 = 0 but spread out for big |\epsilon_0|.
        
    """


def ssh_chain_study():
    """
    We now study the SSH model:
      - On-site energies = 0
      - Hoppings alternate between t0 and t1 along the chain
      - We first do a large "band diagram" for both even and odd N.
      - Then we switch to a smaller chain to see the probability (|psi_n|^2)
        for selected states: ground state, near-zero state, highest-energy state.

    Probability Interpretation:
      - If a wavefunction is "localized" near the chain edges, we'll see that
        |psi_n|^2 is large at those edge sites and small elsewhere.
      - If a wavefunction is "spread out," then |psi_n|^2 is more uniform
        across the chain.
    """

    # Band diagrams showing the variation of the eigen-energies E0,...,EN versus t0, for t1 fixed.
    # Show two cases: even N and odd N.
    
    # Even N
    num_params = 50
    t0_values = np.linspace(-2, 2, num_params)

    # fix N=10 sites, on-site energies = 0, and t1=1 for the SSH alternation
    N = 50
    t1 = 1.0
    epsn = np.zeros(N, dtype=float)

    # Allocate a 2D array to store all eigenvalues
    # shape = (num_params, N) because for each t0, we get N eigenvalues
    all_eigvals = np.zeros((num_params, N), dtype=float)

    for i, t0 in enumerate(t0_values):
        # Build the hopping array for the SSH-like model:
        t_n = np.full(N-1, t0, dtype=complex)
        # Let’s say we want t1 on every even bond index, just as an example
        t_n[::2] = t1

        # Construct Hamiltonian
        H = chain_hamiltonian(epsn, t_n)
        # Diagonalize
        eigvals, _ = np.linalg.eigh(H)
        
        # np.linalg.eigh returns the eigenvalues in ascending order
        eigvals = eigvals[::-1]
        
        all_eigvals[i, :] = eigvals  # store in row i

    # Now plot each eigenvalue "band" vs t0
    plt.figure(figsize=(7,5))

    # all_eigvals has shape (num_params, N)
    # so all_eigvals[:, j] is the j-th eigenvalue band as a function of t0_values
    for j in range(N):
        plt.plot(t0_values, all_eigvals[:, j], '-',
                color='blue', alpha=0.8) 

    plt.xlabel(r'$t_0$')
    plt.ylabel('Energy')
    plt.title('Band structure as lines vs $t_0$ for even N')
    plt.grid(True)
    plt.show()


    # Odd N
    N = 49  
    epsn = np.zeros(N, dtype=float)  # on-site energies = 0
    
    # Range of t0 values (or any other parameter you want to scan)
    num_params = 50
    t0_values = np.linspace(-2, 2, num_params)
    
    # We'll define t1 = 1.0 and alternate t0, t1 in the hopping array
    t1 = 1.0
    
    # Array to store the eigenvalues for each t0
    all_eigvals = np.zeros((num_params, N), dtype=float)
    
    for i, t0 in enumerate(t0_values):
        # Build the (N-1)-dimensional hopping array
        t_n = np.full(N - 1, t0, dtype=complex)
        
        # Set every other bond to t1
        t_n[::2] = t1  # i.e. t_n[0], t_n[2], ...
        
        # Construct the Hamiltonian
        H = chain_hamiltonian(epsn, t_n)
        
        # Diagonalize. np.linalg.eigh returns sorted eigenvalues (ascending).
        eigvals, _ = np.linalg.eigh(H)
        
        # Store them. If you want descending order, you could reverse: eigvals[::-1]
        all_eigvals[i, :] = eigvals
    
    # Now plot each "band" (each eigenvalue index) vs t0
    plt.figure(figsize=(7,5))
    
    for band_idx in range(N):
        plt.plot(t0_values, all_eigvals[:, band_idx], '-',
                color='blue', alpha=0.8)
    
    plt.xlabel(r'$t_0$')
    plt.ylabel('Energy (Eigenvalues)')
    plt.title(f'Band structure (N={N} is odd) as lines vs $t_0$')
    plt.grid(True)
    plt.show()

    # We will now plot the mode distributions psi_n^2 vs n for the ground state and first excited state
    # We will show results for various values of t0 and t1 and for even and odd N.

    
    # Even N
    N_even = 50
    
    # We'll illustrate two parameter sets:
    #   (t0 < t1) and (t0 > t1)
    param_pairs_even = [
        (0.5, 1.0),  # t0 < t1
        (1.5, 1.0)   # t0 > t1
    ]
    
    for (t0_val, t1_val) in param_pairs_even:
        
        # Build the hopping array of length N_even-1
        # Start by filling everything with t0_val
        t_n = np.full(N_even - 1, t0_val, dtype=complex)
        # Now set every other bond to t1_val
        t_n[::2] = t1_val  # i.e. t_n[0], t_n[2], ...
        
        # On-site energies are zero
        epsn = np.zeros(N_even, dtype=float)
        
        # Construct and diagonalize the Hamiltonian
        H = chain_hamiltonian(epsn, t_n)
        eigvals, eigvecs = np.linalg.eigh(H)  # sorted eigenvalues
        
        # Index of eigenvalue closest to zero
        idx_zero = np.argmin(np.abs(eigvals))
        
        # --- Ground State (lowest-energy) ---
        plt.figure()
        plt.plot(np.abs(eigvecs[:, 0])**2, 'o', label='Ground state')
        plt.title(
            f'SSH (N={N_even}), t0={t0_val}, t1={t1_val}\n'
            f'Ground State E={eigvals[0]:.3f}'
        )
        plt.xlabel('Site index n')
        plt.ylabel(r'$|\psi_n|^2$')
        plt.legend()
        plt.show()
        
        # --- Zero-Mode Candidate (nearest E=0) ---
        plt.figure()
        plt.plot(np.abs(eigvecs[:, idx_zero])**2, 'o', label='Near-zero mode')
        plt.title(
            f'SSH (N={N_even}), t0={t0_val}, t1={t1_val}\n'
            f'Near-zero E={eigvals[idx_zero]:.3f}'
        )
        plt.xlabel('Site index n')
        plt.ylabel(r'$|\psi_n|^2$')
        plt.legend()
        plt.show()
        
        # --- Highest-Energy State ---
        plt.figure()
        plt.plot(np.abs(eigvecs[:, -1])**2, 'o', label='Highest-energy state')
        plt.title(
            f'SSH (N={N_even}), t0={t0_val}, t1={t1_val}\n'
            f'Highest State E={eigvals[-1]:.3f}'
        )
        plt.xlabel('Site index n')
        plt.ylabel(r'$|\psi_n|^2$')
        plt.legend()
        plt.show()

    # Odd N
    N_odd = 49
    
    # Same idea: pick two parameter sets
    param_pairs_odd = [
        (0.5, 1.0),  # t0 < t1
        (1.5, 1.0)   # t0 > t1
    ]
    
    for (t0_val, t1_val) in param_pairs_odd:
        
        # Build the hopping array
        t_n = np.full(N_odd - 1, t0_val, dtype=complex)
        t_n[::2] = t1_val
        
        epsn = np.zeros(N_odd, dtype=float)
        
        # Construct and diagonalize
        H = chain_hamiltonian(epsn, t_n)
        eigvals, eigvecs = np.linalg.eigh(H)
        
        idx_zero = np.argmin(np.abs(eigvals))
        
        # --- Ground State ---
        plt.figure()
        plt.plot(np.abs(eigvecs[:, 0])**2, 'o', label='Ground state')
        plt.title(
            f'SSH (N={N_odd}), t0={t0_val}, t1={t1_val}\n'
            f'Ground State E={eigvals[0]:.3f}'
        )
        plt.xlabel('Site index n')
        plt.ylabel(r'$|\psi_n|^2$')
        plt.legend()
        plt.show()
        
        # --- Zero-Mode Candidate ---
        plt.figure()
        plt.plot(np.abs(eigvecs[:, idx_zero])**2, 'o', label='Near-zero mode')
        plt.title(
            f'SSH (N={N_odd}), t0={t0_val}, t1={t1_val}\n'
            f'Near-zero E={eigvals[idx_zero]:.3f}'
        )
        plt.xlabel('Site index n')
        plt.ylabel(r'$|\psi_n|^2$')
        plt.legend()
        plt.show()
        
        # --- Highest-Energy State ---
        plt.figure()
        plt.plot(np.abs(eigvecs[:, -1])**2, 'o', label='Highest-energy state')
        plt.title(
            f'SSH (N={N_odd}), t0={t0_val}, t1={t1_val}\n'
            f'Highest State E={eigvals[-1]:.3f}'
        )
        plt.xlabel('Site index n')
        plt.ylabel(r'$|\psi_n|^2$')
        plt.legend()
        plt.show()

        """
            For the band diagrams:
             • For even N, If t0 and t1 differ a lot, the band gap is large. If they are similar, the gap is small.
               For odd N, one energy level was found to be near zero. In the standard real SSH model, that often means there is a “zero mode” in the gap that localizes near the edges of the chain.

            For the mode distributions:
                - The probability |\psi_n|^2 formed a Gaussian-like distribution for the ground state, with a peak near the center of the chain.
                - The probaility |\psi_n|^2 for the near-zero mode was peaked at the edges of the chain mostly but in one case, it was peaked at the center.
                - The probability |\psi_n|^2 for the highest-energy state was peaked at the center of the chain forming a Gaussian-like distribution.

        """
 






def main():
    random_chain_study()
    ssh_chain_study()

if __name__ == "__main__":
    main()
