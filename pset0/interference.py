import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation




# Specify the number of dimensions
# remember sources*dimensions = no.of x,y,z for sources
dimensions  = 3





def wavefunction(r, src_a, src_r, k):

    """
    Compute the wavefunction at each point in r due to multiple sources in src_r using broadcasting.
    
    Parameters:
      - r: Array of shape (P, 3) for P evaluation points.
      - src_a: Array of shape (N,) for amplitudes of N sources.
      - src_r: Array of shape (N, 3) for positions of N sources.
      - k: Wavenumber.
    
    Returns:
      - psi: Array of shape (P,) containing the computed wavefunction at each evaluation point.
    """
    # Validate input dimensions
    assert src_r.shape[0] == len(src_a), "Mismatch between number of sources and number of amplitudes"
    assert r.shape[1] == dimensions, f"Each evaluation point must have {dimensions} coordinates"
    assert src_r.shape[1] == dimensions, f"Each source position must have {dimensions} coordinates"
     # Use broadcasting to compute differences between each evaluation point and each source
    # r[:, np.newaxis, :] has shape (P, 1, 3)
    # src_r[np.newaxis, :, :] has shape (1, N, 3)
    # The result delta_r will have shape (P, N, 3)
    # Calculate the magnitude of each vector difference along the spatial dimensions (axis=2)
    # This yields an array of shape (P, N) with distances from each point to each source.
    mag_delta_r = np.linalg.norm(r[:, np.newaxis, :] - src_r[np.newaxis, :, :], axis=2)

    # Avoid division by zero: replace zeros with a small epsilon
    mag_delta_r = np.where(mag_delta_r == 0, 1e-10, mag_delta_r)

    # Now compute the wavefunction contributions from each source at each point
    # For each point i and source j, calculate psi_ij = a_j * exp(i*k*r_ij) / r_ij
    # psi_expanded will have shape (P, N)
    psi_expanded = src_a * np.exp(1j * k * mag_delta_r) / mag_delta_r

    # Sum contributions from all sources for each evaluation point
    # Sum over axis=1 (the source axis) to get shape (P,)
    psi = np.sum(psi_expanded, axis=1)

    return psi



def interference_plot(xspan, yspan, z, src_a, src_r, k):

    # unpack tuples, the point is to use xmin,xmax... as normal variables
    # assert the structure of span
    assert len(xspan) == 3, "xspan must be a tuple of (xmin, xmax, M)"
    assert len(yspan) == 3, "yspan must be a tuple of (ymin, ymax, N)"
    xmin, xmax, M = xspan
    ymin, ymax, N = yspan

    # Validate source arrays
    # ndim is for the dimension of array(1D, 2D...)
    assert src_r.ndim == 2 and src_r.shape[1] == dimensions, f"src_r must be of shape (N, {dimensions})"
    assert src_a.ndim == 1 and src_a.shape[0] == src_r.shape[0], "src_a length must match number of rows in src_r"

    # generate the 'x' and 'y' coor
    x = np.linspace(xmin, xmax, M)
    y = np.linspace(ymin, ymax, N)
    # generate a grid of shape M*N
    X,Y = np.meshgrid(x,y)
   

    # Create an array of shape (N*M, 3) for the points (x, y, z)
    # so we can feed them into a wavefunction-like function.
    R = np.zeros((M*N, dimensions))
    # assert(R.shape == src_r.shape)
    R[:, 0] = X.ravel()       # x-coordinates
    R[:, 1] = Y.ravel()       # y-coordinates
    R[:, 2] = z               # all z = constant plane
    # Compute the wavefunction values for each point
    psi_arr = wavefunction(R, src_a, src_r, k)

    # Compute intensity = |psi|^2
    intensity = np.abs(psi_arr)**2
    

    # Reshape intensity back to (N, M) to match (X, Y)
    I_2D = intensity.reshape(N, M)
    


    # Plot using pcolormesh
    plt.pcolormesh(X, Y, I_2D, cmap='plasma', 
               norm=LogNorm(vmin=I_2D.min(), vmax=I_2D.max()))
    plt.colorbar(label="Intensity")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Interference Pattern at z={z}")
    plt.show()
    
def interference_demo():
     

        """
        Demonstrates wave interference patterns using multiple sources with different configurations.
        
        This demo sets up two distinct configurations:
        1. **Symmetric Five-Point Configuration**: Five sources placed symmetrically around the origin.
        - **Purpose**: To visualize complex interference patterns resulting from multiple equidistant sources.
        2. **Linear Double-Slit Configuration**: Two sources aligned along the x-axis.
        - **Purpose**: To simulate classic double-slit interference, highlighting constructive and destructive interference.
        """
        # Common parameters
        k = 5.0               # Wavenumber; higher k leads to more oscillations per unit distance
        z = 0.0               # Detection plane at z=0 for simplicity

        # ----------------------------
        # Configuration 1: Symmetric Five-Point Configuration
        # ----------------------------
        # Five sources placed symmetrically on the xy-plane at equal distances from the origin
        src_r_sym = np.array([
            [0, 0, 0],              # Center source
            [7.5, 0, 0],            # Right
            [0, 7.5, 0],            # Top
            [-7.5, 0, 0],           # Left
            [0, -7.5, 0]            # Bottom
        ], dtype=np.float64)

        # All sources have the same amplitude
        src_a_sym = np.array([1+0j] * src_r_sym.shape[0])

        # Define the spatial grid
        xspan_sym = (-15, 15, 500)  # Wider range for better visibility of interference
        yspan_sym = (-15, 15, 500)

        print("Plotting Symmetric Five-Point Interference Pattern...")
        interference_plot(xspan_sym, yspan_sym, z, src_a_sym, src_r_sym, k)

        # animate the wavefunction for this configuration
        wavefunction_animate(xspan_sym, yspan_sym, z, src_a_sym, src_r_sym, k)

        # ----------------------------
        # Configuration 2: Linear Double-Slit Configuration
        # ----------------------------
        # Two sources placed along the x-axis, simulating a double-slit experiment
        src_r_double = np.array([
            [-5, 0, 0],             # Left slit
            [5, 0, 0]               # Right slit
        ], dtype=np.float64)

        # Both slits have the same amplitude
        src_a_double = np.array([1+0j, 1+0j])

        # Define the spatial grid
        xspan_double = (-20, 20, 800)  # Extended range to capture multiple interference fringes
        yspan_double = (-10, 10, 400)

        print("Plotting Linear Double-Slit Interference Pattern...")
        interference_plot(xspan_double, yspan_double, z, src_a_double, src_r_double, k)

        # animate the wavefunction for this configuration
        wavefunction_animate(xspan_double, yspan_double, z, src_a_double, src_r_double, k)

        # ----------------------------
        # Configuration 3: Circular Array of Sources
        # ----------------------------
        # Ten sources placed uniformly on a circle
        num_sources = 10
        radius = 10
        angles = np.linspace(0, 2 * np.pi, num_sources, endpoint=False)
        src_r_circle = np.stack((radius * np.cos(angles), radius * np.sin(angles), np.zeros(num_sources)), axis=-1)

        # All sources have the same amplitude for simplicity
        src_a_circle = np.array([1+0j] * num_sources)

        # Define the spatial grid
        xspan_circle = (-20, 20, 600)
        yspan_circle = (-20, 20, 600)

        print("Plotting Circular Array Interference Pattern...")
        interference_plot(xspan_circle, yspan_circle, z, src_a_circle, src_r_circle, k)

        # animate the wavefunction for this configuration
        wavefunction_animate(xspan_circle, yspan_circle, z, src_a_circle, src_r_circle, k)

        # ----------------------------
        # Interpretation of Results
        # ----------------------------
        """
        **Symmetric Five-Point Configuration**:
        - **Parameter Choices**:
            - **Number of Sources**: Five sources because there are 4 quadrants and an origin to ensure symmetry.
            - **Positions**: Placing sources symmetrically around the origin ensures that interference patterns are balanced and aesthetically pleasing.
            - **Wavenumber (k=5.0)**: A higher wavenumber results in more oscillations, making interference fringes more distinct.
        - **Results**:
            - The central source combines with the surrounding sources to create a complex interference pattern with multiple bright and dark regions.
            - Placing the sources symmetrically was the most interesting part, the physical wavefunction was like water waves interfering

        **Linear Double-Slit Configuration**:
        - **Parameter Choices**:
            - **Number of Sources**: Two sources mimic the classic double-slit experiment, a fundamental setup in wave interference studies.
            - **Positions**: Aligning sources along the x-axis creates a straightforward interference pattern with clear fringes.
            - **Wavenumber (k=5.0)**: A higher wavenumber results in more oscillations, making interference fringes more distinct.
        - **Results**:
            - Alternating bright and dark fringes perpendicular to the line joining the two slits.
            - Near the sources dark fringes pattern was observed to be like a comet coming near to a planet(hyperbolic trajectory)

        **Circular Array of Sources**:
        - **Parameter Choices**:
            - **Number of Sources**: Ten sources uniformly placed on a circle to observe more interference patterns.
            - **Positions**: Uniform distribution ensures that each source contributes equally to the overall pattern.
            - **Wavenumber (k=5.0)**: A higher wavenumber results in more oscillations, making interference fringes more distinct.
        - **Results**:
            - Circular symmetry in the interference pattern.
            - Circular sources produced a pattern which looked like a flower, it is hard to describe the pattern in words, see it!

        **General observations**:
            - The intensity was the highest near the sources
            - For a single source, the intensity decreased rapidly moving away from the sources
            - Placing a single source on edge of the grid produced 0 intensity towards the centre
            - The interference patterns result from the superposition of waves emanating from each source.
            - Regions where waves constructively interfere appear brighter, while regions of destructive interference appear darker.

        """

def wavefunction_animate(xspan, yspan, z, src_a, src_r, k):
    # Setup grid and initial computations similar to interference_plot:
    xmin, xmax, M = xspan
    ymin, ymax, N = yspan
    x = np.linspace(xmin, xmax, M)
    y = np.linspace(ymin, ymax, N)
    X, Y = np.meshgrid(x, y)

    R = np.zeros((M * N, 3))
    R[:, 0] = X.ravel()
    R[:, 1] = Y.ravel()
    R[:, 2] = z
    psi_arr = wavefunction(R, src_a, src_r, k)
    # Reshape psi_arr back to (N, M) for spatial grid:
    psi_arr = psi_arr.reshape(N, M)

    # Setup the figure and pcolormesh:
    fig, ax = plt.subplots()
    # Initialize plot with zeros for t=0
    initial_wave = np.real(psi_arr * np.exp(-1j * k * 0))  # t=0
    mesh = ax.pcolormesh(X, Y, initial_wave, cmap='plasma')
    fig.colorbar(mesh, ax=ax, label="Wave Amplitude")
    ax.set_title(f"Physical Wavefunction at z={z}")

    # Time period given by T = 2π / ω, here ω=k
    T = 2 * np.pi / k

    def update(frame):
        # frame will be a time value or an index we map to time
        # Calculate time t for current frame; mapping frame index to time
        # Here we assume frames correspond to fractions of the period
        t = (frame / 20) * T  # Adjust denominator for desired speed/resolution

        # Compute the physical wavefunction Ψ at time t
        current_wave = np.real(psi_arr * np.exp(-1j * k * t))
        # Update the pcolormesh with new data
        mesh.set_array(current_wave.ravel())
        ax.set_title(f"Physical Wavefunction at z={z}")
        return mesh,

    # Create animation: frames can be a sequence or a number of frames,
    # interval is in milliseconds, blit=True for performance
    anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
    plt.show()

    
  
def main():
    # This function would plot the interference pattern for the given sources and also perfom the animation of the wavefunction
    interference_demo()

    
    
   
       
if __name__ == "__main__":
    main()