import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def schrodinger_solver_2d(Lx, Ly, Nx, Ny, Vfun, neigs, E0, args=()):
    """
    Solves the 2D time-independent Schrödinger equation 
      (-1/2 * ∇² + V(x,y)) ψ(x,y) = E ψ(x,y)
    on a rectangular domain of size Lx x Ly (centered at the origin) with Dirichlet
    boundary conditions ψ = 0 at x = ±Lx/2 and y = ±Ly/2.
    
    Parameters
    ----------
    Lx, Ly : floats
        Domain lengths in x and y (must be positive).
    Nx, Ny : ints
        Number of *interior* discretization points in x and y directions (must be >= 1).
    Vfun : callable
        Function specifying the potential, called as Vfun(X, Y, *args).
        X, Y are 2D arrays of shape (Nx, Ny).
    neigs : int
        Number of eigenstates to compute.
    E0 : float
        Target energy around which to search for eigenvalues.
    args : tuple, optional
        Additional parameters passed to Vfun.
    
    Returns
    -------
    E : 1D array
        Array of eigenvalues (length = neigs).
    psi : 3D array
        psi[n, i, j] = ψₙ(x[i], y[j]) (shape = (neigs, Nx, Ny)).
    x : 1D array
        x coordinates of interior points (length = Nx).
    y : 1D array
        y coordinates of interior points (length = Ny).
    """
    # sanity checks
    if Lx <= 0:
        raise ValueError("Lx must be positive.")
    if Ly <= 0:
        raise ValueError("Ly must be positive.")
    if Nx < 1 or Ny < 1:
        raise ValueError("Nx, Ny must be >= 1.")
    if neigs < 1:
        raise ValueError("neigs must be >= 1.")

    # grid setup
    dx = Lx / (Nx + 1)
    dy = Ly / (Ny + 1)
    
    x = np.linspace(-Lx/2 + dx, Lx/2 - dx, Nx)
    y = np.linspace(-Ly/2 + dy, Ly/2 - dy, Ny)
    
    # 2D mesh
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # array to evaluate the potential
    V = Vfun(X, Y, *args)  # shape (Nx, Ny)
    if V.shape != (Nx, Ny):
        raise ValueError("Vfun must return an array of shape (Nx, Ny).")
    
    # Flatten the data for operator
    N_total = Nx * Ny
    V_flat = V.ravel(order='C')
    
    # Builc finite difference Laplacian operator
    lap_main = (-2.0/dx**2 - 2.0/dy**2) * np.ones(N_total)
    lap_off_y = 1.0/dy**2 * np.ones(N_total - 1)
    lap_off_x = 1.0/dx**2 * np.ones(N_total - Ny)
    
    # Remove wrap-around in y-direction
    for i in range(1, Nx):
        idx = i*Ny - 1
        lap_off_y[idx] = 0.0
    
    diagonals = [lap_main, lap_off_y, lap_off_y, lap_off_x, lap_off_x]
    offsets = [0, 1, -1, Ny, -Ny]
    L = sp.diags(diagonals, offsets, shape=(N_total, N_total), format='csr')
    

    # Hamiltonian: H = -1/2 L + V
    H = -0.5 * L + sp.diags(V_flat, 0, format='csr')
    
    # solve the Eigenvalue problem
    # We find 'neigs' eigenvalues closest to E0 using shift-invert (sigma=E0).
    E, psi_flat = spla.eigsh(H, k=neigs, sigma=E0, which='LM')
    
    # Reshape and normalize eigenfunctions
    psi = np.empty((neigs, Nx, Ny), dtype=complex)
    for n in range(neigs):
        psi_n = psi_flat[:, n].reshape((Nx, Ny), order='C')
        norm_val = np.sqrt(np.sum(np.abs(psi_n)**2) * dx * dy)
        psi[n, :, :] = psi_n / norm_val
    
    return E, psi, x, y


def schrodinger_2d_error_demo():
    """
    Demonstrates the validity of the schrodinger_solver_2d by:
      (i) Showing pseudocolor plots for some eigenfunctions (e.g., ground state, first excited state).
      (ii) Computing numerical eigenvalues for a 2D infinite well at various grid resolutions N,
          and comparing to exact analytic eigenvalues to examine how the error scales.
      (iii) Estimating the power law of the error vs. N (like ~ 1/N^2).
    """
    # Sanity check for domain
    Lx = 1.0
    Ly = 1.0
    if Lx <= 0 or Ly <= 0:
        raise ValueError("Box dimensions must be positive in the error demo.")
    
    def exact_energy(nx, ny, Lx=1.0, Ly=1.0):
        # E_{nx, ny} = (pi^2 / 2) * (nx^2 / Lx^2 + ny^2 / Ly^2)
        return (np.pi**2 / 2.0) * ((nx**2 / Lx**2) + (ny**2 / Ly**2))

    def zero_potential(x, y):
        """Inside an infinite well (with Dirichlet BC at boundaries), we take V=0."""
        return np.zeros_like(x)
    
    # setup demo states
    Nx_demo = 40
    Ny_demo = 40
    neigs_demo = 6
    E0_demo = 10.0  # near the ground+first few states

    E_demo, psi_demo, xgrid, ygrid = schrodinger_solver_2d(
        Lx, Ly, Nx_demo, Ny_demo,
        zero_potential, neigs_demo, E0_demo
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    X2D, Y2D = np.meshgrid(xgrid, ygrid, indexing='ij')
    
    for i in range(4):
        pc = axes[i].pcolormesh(X2D, Y2D, np.real(psi_demo[i]), shading='auto')
        axes[i].set_aspect('equal')
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
        axes[i].set_title(f"State {i}, E ~ {E_demo[i]:.3f}")
        fig.colorbar(pc, ax=axes[i])
    
    fig.suptitle("Pseudocolor: 2D infinite well (numerical eigenfunctions) real part of wavefuntion")
    plt.tight_layout()
    plt.show()
    
    # Numerical vs exact energy comparison
    reference_states = [(1,1), (2,1), (2,2)]
    exact_vals = [exact_energy(nx, ny, Lx, Ly) for (nx, ny) in reference_states]
    
    N_list = [10, 20, 30, 40, 60, 80]
    errors = [[] for _ in range(len(reference_states))]
    
    for N in N_list:
        E_num, _, _, _ = schrodinger_solver_2d(
            Lx, Ly, N, N, zero_potential, neigs=9, E0=20.0
        )
        sorted_inds = np.argsort(E_num)
        E_sorted = E_num[sorted_inds]
        
        # Match each reference state to the closest numerical value
        for i, E_exact in enumerate(exact_vals):
            idx_closest = np.argmin(np.abs(E_sorted - E_exact))
            E_found = E_sorted[idx_closest]
            rel_err = abs(E_found - E_exact)/abs(E_exact)
            errors[i].append(rel_err)
    
    # Plot error vs N (linear scale)
    fig2, ax2 = plt.subplots(figsize=(8,6))
    for i, (nx, ny) in enumerate(reference_states):
        ax2.plot(N_list, errors[i], 'o-', label=f"(nx={nx}, ny={ny})")
    ax2.set_xlabel("N (grid points in each direction)")
    ax2.set_ylabel("Relative error in energy")
    ax2.set_title("Error vs. Discretization for 2D Box")
    ax2.legend()
    ax2.grid(True)
    plt.show()

    # power law scaling
    fig3, ax3 = plt.subplots(figsize=(8,6))
    for i, (nx, ny) in enumerate(reference_states):
        logN = np.log(N_list)
        logErr = np.log(errors[i])
        # Fit: logErr ~ A + p*logN
        p, A = np.polyfit(logN, logErr, 1)
        ax3.loglog(N_list, errors[i], 'o-', label=f"(nx={nx}, ny={ny}), slope={p:.2f}")
    
    ax3.set_xlabel("N (log scale)")
    ax3.set_ylabel("Relative error (log scale)")
    ax3.set_title("Log-Log Error Scaling: 2D Box")
    ax3.legend()
    ax3.grid(True, which='both')
    plt.show()
    
    # Print power-law summary
    for i, (nx, ny) in enumerate(reference_states):
        logN = np.log(N_list)
        logErr = np.log(errors[i])
        p, A = np.polyfit(logN, logErr, 1)
        print(f"State (nx={nx}, ny={ny}), best-fit exponent p ~ {p:.3f}")
    """
        Relative error in energy was higher for higher energy states and even higher for lesser number of grid points
        The behaviour is observed to mimic exponential decay with increasing number of grid points
        The power law scaling is observed and the slope is the same for all the states
    """
def stadium_demo():
    """
    Demonstrates eigenstates in a stadium-shaped "infinite well":
      - The stadium region is formed by two semicircles of radius R
        joined by a rectangle of length L.
      - We set V=0 inside the stadium and a very large value (V0=1e6) outside
        (within a big bounding rectangle), so that the wavefunction is effectively
        forced to be zero outside the stadium shape.
      - Dirichlet boundary conditions also apply on the bounding rectangle edges.
      
    This function plots a few representative eigenstates for chosen parameters
    (R, L, energies ~ E0).
    """
    # define stadium shape
    R = 10.0  # radius of semicircles
    L = 1.0   # length of the rectangle connecting the semicircles
    E0 = 20.0  
    neigs = 6  
    
    # bounding box: x from -(R+L/2) to +(R+L/2), y from -R to +R
    Lx = 2.0*(R + L/2.0)
    Ly = 2.0*R
    Nx = 200
    Ny = 200
    
    if R <= 0 or L < 0:
        raise ValueError("Invalid stadium dimensions: R>0, L>=0.")
    if Nx < 1 or Ny < 1:
        raise ValueError("Nx, Ny must be >= 1 for stadium demo.")
    
    def stadium_potential(x, y, R, L, V0=1e6):
        """
        V=0 inside the stadium, V=V0 outside.
        
        Stadium geometry:
          - For x between [-L/2, +L/2], inside if |y| <= R.
          - For x > +L/2, inside if (x - L/2)^2 + y^2 <= R^2.
          - For x < -L/2, inside if (x + L/2)^2 + y^2 <= R^2.
        """
        V = V0 * np.ones_like(x)
        
        inside_rect = ((x >= -L/2) & (x <= L/2) & (np.abs(y) <= R))
        inside_right_circle = ((x - L/2)**2 + y**2 <= R**2)
        inside_left_circle = ((x + L/2)**2 + y**2 <= R**2)
        inside_stadium = inside_rect | inside_right_circle | inside_left_circle
        
        V[inside_stadium] = 0.0
        return V
    
    # wrap potential to pass to solver
    def Vfun_stadium(X, Y):
        return stadium_potential(X, Y, R, L, 1e6)
    
    # Solve
    E, psi, xgrid, ygrid = schrodinger_solver_2d(
        Lx, Ly, Nx, Ny,
        Vfun_stadium, neigs, E0
    )
    
    # plot representative states
    X2D, Y2D = np.meshgrid(xgrid, ygrid, indexing='ij')
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    
    for i in range(neigs):
        c = axes[i].pcolormesh(X2D, Y2D, np.abs(psi[i]), shading='auto')
        axes[i].set_title(f"Mode {i}, E={E[i]:.3f}")
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
        axes[i].set_aspect('equal')
        fig.colorbar(c, ax=axes[i])
    
    plt.suptitle(f"Stadium Billiard: R={R}, L={L}, E0={E0}, Nx={Nx}, Ny={Ny}")
    plt.tight_layout()
    plt.show()

    """
    Bright (or warm) colors in the plot typically indicate larger wavefunction amplitude, while dark (or cool) colors indicate small amplitude (close to zero).
    The color bar on the right of each subplot shows the numerical range of those amplitudes. In other words, the bottom of the color bar corresponds to 
    ∣ψ∣≈0, and the top of the color bar corresponds to the maximum amplitude of that particular mode. 
    A higher amplitude in the wavefunction means that the probability of finding the particle at that location is higher. 
    In quantum mechanics, the square of the wavefunction's amplitude (|ψ(x, y)|²) gives you the probability density. 
    So, if you see bright, high-amplitude regions in the plot, it indicates where the particle is most likely to be found.

    The patterns you see, like rings or lines, are the “footprints” of the wave. For example, you might see concentric circles (like ripples in water) or lines that go through the center.
    Because the stadium shape (with two big curved parts and a short straight middle) is almost circular, many of the wave patterns look nearly circular too.

    The numbers next to the plots are the energies of each wave mode. When these numbers are very close together, 
    it means the waves are almost the same in energy—a sign of the shape’s symmetry.
    In a perfectly circular area, some waves would have exactly the same energy, but here 
    they’re just very close because the stadium isn’t a perfect circle.

    While most of the waves look circular, you can notice small distortions. That’s because the tiny straight section connecting the curves slightly changes the symmetry, altering the pattern a bit.


    """

def main():
    # Run demos
    schrodinger_2d_error_demo()
    stadium_demo()

if __name__ == "__main__":
    main()
