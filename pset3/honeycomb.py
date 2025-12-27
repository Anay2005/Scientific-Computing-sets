import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

def honeycomb_model(a, L1, L2, t, Afun, args=()):
    """
    Construct site positions and Hamiltonian for a 2D honeycomb lattice
    in a parallelogram of L1 x L2 unit cells, with Peierls phases from
    a vector potential A.
    
    Sanity checks:
      - a > 0
      - L1 and L2 are positive integers
      - t is a numeric type
      - Afun is callable and returns a sequence of two numbers

    Parameters
    ----------
    a : float
        Nearest-neighbor distance (lattice spacing).
    L1, L2 : int
        Number of unit-cell spacings along each parallelogram direction.
    t : float
        Hopping amplitude (real).
    Afun : callable
        Function specifying the vector potential, called as Afun(x, y, *args)
        and returning [Ax, Ay].
    args : tuple
        Additional parameters to be passed to Afun(...).

    Returns
    -------
    sites : np.ndarray, shape (N, 2)
        Array of site positions (x, y).  N = 2 * L1 * L2 (A,B sublattices).
    H : scipy.sparse.csr_matrix
        The tight-binding Hamiltonian (N x N), with Peierls phases included.
        H_{ij} = t * exp(i * A(r_mid) · (r_i - r_j)) for nearest neighbors.
    """
    # --- Sanity Checks ---
    if not (isinstance(a, (int, float)) and a > 0):
        raise ValueError("Parameter 'a' must be a positive number.")
    if not (isinstance(L1, int) and L1 > 0):
        raise ValueError("L1 must be a positive integer.")
    if not (isinstance(L2, int) and L2 > 0):
        raise ValueError("L2 must be a positive integer.")
    if not isinstance(t, (int, float)):
        raise ValueError("Hopping amplitude 't' must be numeric.")
    if not callable(Afun):
        raise ValueError("Afun must be callable.")
    try:
        A_test = Afun(0, 0, *args)
        if not (hasattr(A_test, '__len__') and len(A_test) == 2):
            raise ValueError("Afun must return a sequence of two numbers.")
    except Exception as e:
        raise ValueError("Afun did not execute properly at (0,0): " + str(e))

    # Number of sites: 2 sites per unit cell (A and B)
    N = 2 * L1 * L2

    # Prepare array to store site positions
    sites = np.zeros((N, 2), dtype=float)

    # --- Lattice vectors for honeycomb (parallelogram) ---
    a1 = np.array([1.5 * a,  0.5 * np.sqrt(3) * a])  # (3/2 a, +sqrt(3)/2 a)
    a2 = np.array([1.5 * a, -0.5 * np.sqrt(3) * a])  # (3/2 a, -sqrt(3)/2 a)

    # Sublattice offsets (A, B)
    tauA = np.array([0.0,             0.0])
    tauB = np.array([0.5 * a, 0.5 * np.sqrt(3) * a])

    def idx(m, n, sublatt):
        """Return site index for sublattice in cell (m, n). sublatt = 0 for A, 1 for B."""
        return 2 * (m + n * L1) + sublatt

    # Fill in site positions
    for n in range(L2):
        for m in range(L1):
            iA = idx(m, n, 0)
            iB = idx(m, n, 1)
            Rmn = m * a1 + n * a2  # Origin of cell (m, n)
            sites[iA] = Rmn + tauA
            sites[iB] = Rmn + tauB

    # --- Build the Hamiltonian ---
    H = lil_matrix((N, N), dtype=complex)
    neighbor_offsets = [
        np.array([ 0.5 * a,         0.5 * np.sqrt(3) * a]),
        np.array([ 0.5 * a,        -0.5 * np.sqrt(3) * a]),
        np.array([-1.0 * a,         0.0])
    ]

    for n in range(L2):
        for m in range(L1):
            iA = idx(m, n, 0)
            rA = sites[iA]
            for d in neighbor_offsets:
                rB = rA + d
                rBminusTauB = rB - tauB
                A_mat = np.column_stack((a1, a2))
                try:
                    xy = np.linalg.solve(A_mat, rBminusTauB)
                except np.linalg.LinAlgError as e:
                    raise ValueError("Error solving linear system for neighbor cell indices: " + str(e))
                mprime, nprime = xy
                mprime_int = int(round(mprime))
                nprime_int = int(round(nprime))
                if (abs(mprime - mprime_int) < 1e-7 and
                    abs(nprime - nprime_int) < 1e-7 and
                    0 <= mprime_int < L1 and 0 <= nprime_int < L2):
                    iB = idx(mprime_int, nprime_int, 1)
                    rB_exact = sites[iB]
                    r_mid = 0.5 * (rA + rB_exact)
                    Ax, Ay = Afun(r_mid[0], r_mid[1], *args)
                    Avec = np.array([Ax, Ay])
                    dr = rA - rB_exact  # Vector from B to A
                    phase = np.exp(1j * np.dot(Avec, dr))
                    H[iA, iB] = t * phase
                    H[iB, iA] = t * np.conjugate(phase)
    return sites, H.tocsr()

def honeycomb_demo():
    """
    Demonstrate the honeycomb-lattice tight-binding model with a uniform
    magnetic field, focusing on eigenstates near E=0.
    
    Sanity checks:
      - Validates lattice parameters and number of eigenstates.
    
    We use the vector potential:
        A(x, y) = (B0/2) * [-y, x]
    
    This corresponds to a constant magnetic field B0 in the z-direction.
    The demo diagonalizes a portion of the Hamiltonian around E=0 and produces:
      1) A plot of the eigenvalues near E=0.
      2) pcolormesh-style (tripcolor) plots of the probability density |psi|^2
         for the eigenstates.
    """
    # --- Sanity Checks for demo parameters ---
    a = 1.0        # Must be > 0.
    L1 = 20        # Positive integer.
    L2 = 20        # Positive integer.
    t = 1.0        # Numeric.
    B0 = 1.0       # Numeric.
    k = 6          # Positive integer.
    if not (isinstance(L1, int) and L1 > 0):
        raise ValueError("L1 must be a positive integer.")
    if not (isinstance(L2, int) and L2 > 0):
        raise ValueError("L2 must be a positive integer.")
    if not (isinstance(k, int) and k > 0):
        raise ValueError("k must be a positive integer.")

    def uniform_magnetic_A(x, y, B):
        # A(x, y) = (B/2)*[-y, x]
        return [-(B/2) * y, (B/2) * x]

    # --- Build the Hamiltonian ---
    sites, H = honeycomb_model(a, L1, L2, t, uniform_magnetic_A, args=(B0,))
    print(f"Number of sites = {H.shape[0]}")
    print("Hamiltonian built successfully. Now diagonalizing near E=0...")

    try:
        vals, vecs = eigsh(H, k=k, sigma=0.0, which='LM')
    except Exception as e:
        raise RuntimeError("Error during eigenvalue decomposition: " + str(e))
    
    idx_sort = np.argsort(vals)
    vals = vals[idx_sort]
    vecs = vecs[:, idx_sort]

    # --- Plot eigenvalues ---
    plt.figure(figsize=(5, 4))
    plt.plot(range(k), vals, 'o-')
    plt.title("Eigenvalues near E=0")
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Energy E")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Plot eigenstates using tripcolor (a pcolormesh-like plot for unstructured grids) ---
    xcoords = sites[:, 0]
    ycoords = sites[:, 1]
    # Create a triangulation for the unstructured grid.
    triang = Triangulation(xcoords, ycoords)

    for i in range(k):
        psi = vecs[:, i]
        prob_density = np.abs(psi) ** 2
        plt.figure(figsize=(5, 4))
        # tripcolor creates a pseudocolor plot on an unstructured triangular grid.
        plt.tripcolor(triang, prob_density, shading='flat', cmap='plasma')
        plt.colorbar(label="|psi|^2")
        plt.title(f"Eigenstate {i+1} (E = {vals[i]:.4f})")
        plt.xlabel("x coordinate")
        plt.ylabel("y coordinate")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    """
    The first figure shows that there is a small set of eigenstates straddling zero energy—some just below and some just above—reflecting the bipartite nature of the honeycomb lattice and possibly the Dirac cone near 
    E=0
    The first two (negative-energy) states are almost degenerate in energy and look similar, 
    but with slightly shifted max amplitude of wavefunction.

    Some bright spots appear near the edges or corners of the parallelogram. This can happen in finite samples where wavefunctions favor boundary sites, 
    especially in the presence of a magnetic field or other perturbations.
    
    """
def main():
    honeycomb_demo()
if __name__ == "__main__":
    main()
