import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Callable, Any, Optional

@dataclass(frozen=True)
class SimulationConfig:
    """
    Immutable configuration for the 2D Schrödinger solver.
    Keeps physical parameters and grid settings safe and organized.
    """
    Lx: float
    Ly: float
    Nx: int
    Ny: int
    neigs: int
    E0: float

    def __post_init__(self):
        """Sanity checks ran automatically upon creation."""
        if self.Lx <= 0 or self.Ly <= 0:
            raise ValueError("Dimensions Lx, Ly must be positive.")
        if self.Nx < 1 or self.Ny < 1:
            raise ValueError("Grid points Nx, Ny must be >= 1.")
        if self.neigs < 1:
            raise ValueError("Number of eigenvalues must be >= 1.")

    @property
    def dx(self) -> float:
        return self.Lx / (self.Nx + 1)

    @property
    def dy(self) -> float:
        return self.Ly / (self.Ny + 1)

    @property
    def x_grid(self) -> np.ndarray:
        return np.linspace(-self.Lx/2 + self.dx, self.Lx/2 - self.dx, self.Nx)

    @property
    def y_grid(self) -> np.ndarray:
        return np.linspace(-self.Ly/2 + self.dy, self.Ly/2 - self.dy, self.Ny)


def build_1d_laplacian(size: int, step_size: float) -> sp.csr_matrix:
    """
    Constructs a 1D sparse Laplacian matrix with Dirichlet boundary conditions.
    """
    diagonals = [np.ones(size - 1), -2 * np.ones(size), np.ones(size - 1)]
    offsets = [-1, 0, 1]
    laplacian_1d = sp.diags(diagonals, offsets, shape=(size, size), format='csr')
    return laplacian_1d / (step_size ** 2)


def schrodinger_solver_2d(
    config: SimulationConfig,
    potential_func: Callable[[np.ndarray, np.ndarray, Any], np.ndarray],
    args: tuple = ()
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves the 2D time-independent Schrödinger equation.
    
    Parameters
    ----------
    config : SimulationConfig
        The physical and grid configuration object.
    potential_func : callable
        Function V(X, Y, *args) returning the potential energy array.
    args : tuple, optional
        Extra arguments passed to potential_func.

    Returns
    -------
    eigenvalues : np.ndarray
        Sorted eigenvalues.
    eigenvectors : np.ndarray
        Normalized wavefunctions, shape (neigs, Nx, Ny).
    """
    
    # 1. Grid Setup (using properties from dataclass)
    X, Y = np.meshgrid(config.x_grid, config.y_grid, indexing='ij')

    # 2. Potential Energy
    V = potential_func(X, Y, *args)
    if V.shape != (config.Nx, config.Ny):
        raise ValueError(f"Potential shape mismatch: expected {(config.Nx, config.Ny)}, got {V.shape}")
    
    # Flatten using Row-Major (C-style) order
    V_diagonal = sp.diags(V.ravel(order='C'), 0, format='csr')

    # 3. Kinetic Energy (using Kronecker Product)
    D_1d_x = build_1d_laplacian(config.Nx, config.dx)
    D_1d_y = build_1d_laplacian(config.Ny, config.dy)
    
    I_x = sp.eye(config.Nx)
    I_y = sp.eye(config.Ny)

    # D_xx = D_x (x) I_y  (Acting on slow index X)
    # D_yy = I_x (x) D_y  (Acting on fast index Y)
    laplacian_x = sp.kron(D_1d_x, I_y, format='csr') 
    laplacian_y = sp.kron(I_x, D_1d_y, format='csr')
    
    T_mat = -0.5 * (laplacian_x + laplacian_y)

    # 4. Hamiltonian & Solver
    H = T_mat + V_diagonal

    vals, vecs = spla.eigsh(H, k=config.neigs, sigma=config.E0, which='LM')

    # 5. Post-Processing
    psi_out = np.empty((config.neigs, config.Nx, config.Ny), dtype=vals.dtype)
    
    for k in range(config.neigs):
        # Reshape the flattened eigenvector back to 2D grid
        grid_state = vecs[:, k].reshape((config.Nx, config.Ny), order='C')
        
        # Normalize: Integral |psi|^2 dA = 1
        norm_factor = np.sqrt(np.sum(np.abs(grid_state)**2) * config.dx * config.dy)
        psi_out[k, :, :] = grid_state / norm_factor

    # Sort eigenvalues
    sort_idx = np.argsort(vals)
    return vals[sort_idx], psi_out[sort_idx]


# --- Helper Functions for Demos ---

def potential_infinite_well(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.zeros_like(X)

def potential_stadium(X: np.ndarray, Y: np.ndarray, radius: float, length: float, v_wall: float = 1e6) -> np.ndarray:
    V = np.full_like(X, v_wall)
    inside_rect = (X >= -length/2) & (X <= length/2) & (np.abs(Y) <= radius)
    inside_right_circle = ((X - length/2)**2 + Y**2 <= radius**2)
    inside_left_circle  = ((X + length/2)**2 + Y**2 <= radius**2)
    
    mask_inside = inside_rect | inside_right_circle | inside_left_circle
    V[mask_inside] = 0.0
    return V


# --- Demos ---

def run_stadium_demo():
    print("\n--- Running Stadium Billiard Demo with Dataclass ---")
    
    radius = 10.0
    length = 5.0
    
    # Create the immutable configuration object
    sim_config = SimulationConfig(
        Lx = 2.0 * (radius + length + 2.0),
        Ly = 2.0 * (radius + 2.0),
        Nx = 150,
        Ny = 150,
        neigs = 6,
        E0 = 2.0
    )
    
    # Pass the config object instead of loose variables
    energies, psi = schrodinger_solver_2d(
        sim_config, 
        potential_stadium, 
        args=(radius, length)
    )

    X2D, Y2D = np.meshgrid(sim_config.x_grid, sim_config.y_grid, indexing='ij')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i in range(sim_config.neigs):
        im = axes[i].pcolormesh(X2D, Y2D, np.abs(psi[i]), shading='auto', cmap='inferno')
        axes[i].set_title(f"State {i}: E = {energies[i]:.4f}")
        axes[i].set_aspect('equal')
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Outline overlay
        axes[i].contour(X2D, Y2D, potential_stadium(X2D, Y2D, radius, length), levels=[1], colors='white', linewidths=0.5)

    plt.suptitle(f"Stadium Billiard Eigenstates (R={radius}, L={length})")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_stadium_demo()