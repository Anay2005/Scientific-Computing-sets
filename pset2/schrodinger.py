import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def schrodinger1d_integrate(psifun, Vfun, nt, dt, N, L, output_step):
    """
    Integrates the 1D time-dependent Schrödinger equation using the split-step Fourier method.

    Inputs:
      psifun      : Function specifying the initial wavefunction. When called with an array x,
                    it returns an equal-sized array containing ψ(x,0).
      Vfun        : Function specifying the potential. When called with an array x,
                    it returns an equal-sized array containing V(x).
      nt          : Number of time steps to take (integer).
      dt          : Length of each time step (float).
      N           : Number of spatial discretization points (integer).
      L           : Total length of the computational domain (float). The domain spans -L/2 ≤ x < L/2.
      output_step : Number of time steps between each output record (integer).

    Return values:
      x   : 1D numpy array of size N, containing the spatial discretization points.
      t   : 1D numpy array specifying the times at which the wavefunction was recorded.
            The first element is 0 (the initial time).
      psi : 2D numpy array recording the wavefunction, where psi[i, j] ≡ ψ(x_j, t_i).
            The first row (psi[0, :]) is the initial wavefunction.
    """
    # Constants (using natural units: hbar = 1, m = 1)
    hbar = 1.0
    m = 1.0

    # Define the spatial grid: x in [-L/2, L/2)
    x = np.linspace(-L/2, L/2, N, endpoint=False)

    # Define the corresponding momentum grid.
    # np.fft.fftfreq returns frequencies in cycles/unit; multiplying by 2π gives angular frequencies.
    k = np.fft.fftfreq(N, d=L/N) * 2 * np.pi

    # Initial wavefunction (ensure it's a complex array)
    psi = np.array(psifun(x), dtype=complex)

    # Evaluate the potential on the grid
    V = Vfun(x)

    # Precompute the phase factors:
    # Half-step potential evolution operator: exp(-i V dt/(2*hbar))
    expV_half = np.exp(-1j * V * dt / (2 * hbar))
    # Full-step kinetic evolution operator in momentum space: exp(-i T dt/hbar)
    # Here, T = p^2/(2m) and in momentum space p = hbar * k, so T = hbar^2 k^2/(2m).
    expT = np.exp(-1j * (hbar * k)**2 / (2 * m) * dt / hbar)

    # Prepare arrays to record output
    n_output = nt // output_step + 1  # number of output records
    psi_rec = np.empty((n_output, N), dtype=complex)
    t_rec = np.empty(n_output)

    # Record the initial state
    psi_rec[0, :] = psi
    t_rec[0] = 0.0

    out_index = 1  # index for output recording

    # Time evolution loop
    for step in range(1, nt + 1):
        # --- Split-step Fourier method ---
        # 1. Apply half-step potential evolution in position space.
        psi = expV_half * psi

        # 2. Transform to momentum space.
        psi_k = np.fft.fft(psi)

        # 3. Apply full-step kinetic evolution in momentum space.
        psi_k = expT * psi_k

        # 4. Transform back to position space.
        psi = np.fft.ifft(psi_k)

        # 5. Apply another half-step potential evolution in position space.
        psi = expV_half * psi

        # Record the wavefunction every "output_step" steps.
        if step % output_step == 0:
            psi_rec[out_index, :] = psi
            t_rec[out_index] = step * dt
            out_index += 1

    return x, t_rec, psi_rec

def schrodinger1Ddemo():
    """
    Runs and plots two time-dependent 1D Schrödinger equation simulations
    in a harmonic oscillator potential V(x) = 20x^2:
      (i) for the ground state of the oscillator,
      (ii) for a coherent state given by ψ(x,0) = (2/π)^(1/4) e^{-(x-2)^2}.
    
    The probability density |ψ(x,t)|^2 is plotted using pcolormesh with x on the horizontal axis
    and t on the vertical axis.
    
    Observations:
      - The ground state is stationary (aside from a global phase), so |ψ|^2 remains essentially
        constant in time.
      - The coherent state exhibits oscillatory behavior, reflecting the classical motion of a 
        displaced Gaussian in a harmonic potential.
    """
    # Simulation parameters
    L = 20        # total spatial length (domain: -L/2 to L/2)
    N = 512       # number of spatial grid points
    nt = 10000    # number of time steps (adjust as needed for longer simulations)
    dt = 0.001    # time step
    output_step = 10  # record every 10 time steps

    # Define the harmonic oscillator potential
    def Vfun(x):
        return 20 * x**2

    # Ground state of the harmonic oscillator.
    # For a potential V(x)=20x^2 = (1/2)*ω^2 x^2, we have ω^2=40 so ω = 2√10.
    # The ground state wavefunction is
    #   ψ₀(x) = (mω/πℏ)^(1/4) exp(-mω x²/(2ℏ))
    # With m = ℏ = 1, this becomes:
    #   ψ₀(x) = ((2√10)/π)^(1/4) exp(-√10 * x²)
    def psifun_ground(x):
        norm = (2 * np.sqrt(10) / np.pi)**0.25
        return norm * np.exp(-np.sqrt(10) * x**2)

    # Coherent state:
    # Given by ψ(x,0) = (2/π)^(1/4) exp( -(x-2)² )
    def psifun_coherent(x):
        norm = (2 / np.pi)**0.25
        return norm * np.exp(-(x - 2)**2)

    # Run simulation for the ground state
    x, t_ground, psi_ground = schrodinger1d_integrate(psifun_ground, Vfun, nt, dt, N, L, output_step)

    # Run simulation for the coherent state
    x, t_coherent, psi_coherent = schrodinger1d_integrate(psifun_coherent, Vfun, nt, dt, N, L, output_step)

    # Create subplots: one for each initial condition
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot ground state evolution
    pcm0 = axs[0].pcolormesh(x, t_ground, np.abs(psi_ground)**2, shading='auto')
    axs[0].set_ylabel('Time t')
    axs[0].set_title('Ground State Evolution')
    fig.colorbar(pcm0, ax=axs[0], label='|ψ(x,t)|²')

    # Plot coherent state evolution
    pcm1 = axs[1].pcolormesh(x, t_coherent, np.abs(psi_coherent)**2, shading='auto')
    axs[1].set_xlabel('Position x')
    axs[1].set_ylabel('Time t')
    axs[1].set_title('Coherent State Evolution')
    fig.colorbar(pcm1, ax=axs[1], label='|ψ(x,t)|²')

    plt.tight_layout()
    plt.show()
def schrodinger2d_integrate(psifun, Vfun, nt, dt, N, L, output_step):
    """
    Integrates the 2D time-dependent Schrödinger equation
      i∂ψ/∂t = -½ (∂²ψ/∂x² + ∂²ψ/∂y²) + V(x,y) ψ(x,y,t)
    using a split-step Fourier method.
    
    Inputs:
      psifun      : Function specifying the initial wavefunction. When called with 2D arrays x and y,
                    returns an array containing ψ(x,y,0).
      Vfun        : Function specifying the potential. When called with 2D arrays x and y,
                    returns an array containing V(x,y).
      nt          : Number of time steps to take (integer).
      dt          : Time step (float).
      N           : Number of spatial discretization points in each direction (integer). Total points = N².
      L           : Total length of the computational domain in each direction (float).
                    The domain is x,y ∈ [–L/2, L/2).
      output_step : Number of time steps between output records (integer).
                    At every output_step time steps, the wavefunction is saved.
                    
    Returns:
      X, Y  : 2D arrays (shape N×N) containing the x- and y–coordinates of the spatial discretization points.
      t     : 1D array containing the times at which the wavefunction was recorded (first element is 0).
      psi   : 3D array recording the wavefunction, with psi[i,j,k] = ψ(x_k, y_j, t_i).
              Note that psi[0,:,:] is the initial wavefunction.
    """
    # For simplicity we take hbar = m = 1.
    hbar = 1.0
    m = 1.0

    # Set up the spatial grid.
    x1d = np.linspace(-L/2, L/2, N, endpoint=False)
    y1d = np.linspace(-L/2, L/2, N, endpoint=False)
    # Create 2D coordinate arrays (X: x-values, Y: y-values)
    X, Y = np.meshgrid(x1d, y1d)
    
    # Initial wavefunction (force complex type)
    psi = np.array(psifun(X, Y), dtype=complex)
    
    # Evaluate the potential on the grid.
    V = Vfun(X, Y)
    
    # Precompute the potential evolution operator for a half time step.
    expV_half = np.exp(-1j * V * dt/2)
    
    # Set up the momentum grid (for each direction).
    dx = L / N
    k1d = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # wave numbers in 1D
    # Kinetic evolution phase factors (for a full dt split into two halves along x and y):
    # Here the kinetic operator is exp[-i dt/2 * k^2] for each direction.
    expKx = np.exp(-1j * dt/2 * (k1d**2))  # for x–direction (shape: (N,))
    expKy = np.exp(-1j * dt/2 * (k1d**2))  # for y–direction (shape: (N,))

    # Determine the number of outputs.
    n_output = nt // output_step + 1
    psi_rec = np.empty((n_output, N, N), dtype=complex)
    t_rec = np.empty(n_output)
    
    # Record the initial state.
    psi_rec[0, :, :] = psi
    t_rec[0] = 0.0
    out_idx = 1
    
    # Main time evolution loop.
    for step in range(1, nt + 1):
        # 1. Half step potential evolution in physical space.
        psi = expV_half * psi
        
        # 2. Kinetic evolution along x.
        #    FFT along x (axis=1), multiply by phase factor, then inverse FFT.
        psi = np.fft.fft(psi, axis=1)
        psi = psi * expKx[None, :]  # broadcasting over rows (y direction)
        psi = np.fft.ifft(psi, axis=1)
        
        # 3. Kinetic evolution along y.
        #    FFT along y (axis=0), multiply by phase factor, then inverse FFT.
        psi = np.fft.fft(psi, axis=0)
        psi = psi * expKy[:, None]  # broadcasting over columns (x direction)
        psi = np.fft.ifft(psi, axis=0)
        
        # 4. Second half step potential evolution.
        psi = expV_half * psi
        
        # 5. Record output if required.
        if step % output_step == 0:
            psi_rec[out_idx, :, :] = psi
            t_rec[out_idx] = step * dt
            out_idx += 1

    return X, Y, t_rec, psi_rec

def schrodinger2d_demo():
    """
    Demonstrates an animated solution to the 2D time-dependent Schrödinger equation.
    
    In this demo a circular well potential is used:
         V(x,y) = 0     for x² + y² < 16,
         V(x,y) = 100   otherwise.
         
    The initial wavefunction is a Gaussian wavepacket centered at (-2, 0) with a momentum kick in the x–direction.
    The animation shows pcolormesh-generated pseudocolor plots of the probability density |ψ(x,y,t)|².
    
    Observations (see code comments):
      - The wavepacket is initially localized inside the well.
      - When it reaches the high-potential boundary (roughly a circle of radius 4),
        part of the wave is reflected. Interference patterns develop.
      - With a coarse grid (here 50×50) the simulation runs quickly while capturing the qualitative behavior.
    """

    """
    Parameter and potential discussion:

    1. Domain Size (L) and Grid Resolution (N):
       - In this demo, L is 10 and N is 50. With a small N, the simulation runs quickly but the resolution
         is coarse. Increasing N improves spatial resolution but will require more computation.
       - Changing L affects the size of the computational domain. If the wavepacket approaches the boundary,
         increasing L may help avoid artificial reflections due to periodic boundary conditions.

    2. Potential V(x, y):
       - The circular well potential is defined as V(x,y)=0 for x²+y²<16 and V(x,y)=100 otherwise.
       - for a Double well potential, V(x,y) can be defined as 50 * (x**2 - 4)**2
          the following behaviors were observed:
	        •	Ground State: 
            The probability density developed two distinct peaks—each localized in one of the wells—reflecting the symmetric double-well 
            structure.

	        •	Coherent State: 
            When using a displaced Gaussian as the initial condition, 
            the coherent state exhibited tunneling between the wells. 
            The probability density oscillated between the two wells over time, 
            and interference patterns emerged during the tunneling events.

        - Square barrier potential can be defined as np.where(np.abs(x) < 2, 0, 100)
            the following behaviors were observed:
             	•	Ground State: 
                The wavefunction remained localized within the central low-potential region, 
                though some distortions were noticeable near the barrier edges.
	            •	Coherent State: 
                The coherent state’s evolution showed clear reflections and some transmissions upon hitting the barrier. 
                This resulted in noticeable interference fringes and a split probability density as some of the 
                wavefunction were reflected while others tunneled through the barrier.

    3. Time Step (dt) and Output Frequency (output_step):
       - dt=0.05 is chosen here; ensure that dt is small enough for. Choosing a smaller dt slows down the simulation.
       - Recording every time step (output_step = 1) gave a smooth animation but increases memory usage.
    """
    # Define the circular well potential.
    def Vfun(x, y):
        return np.where(x**2 + y**2 < 16, 0, 100)
    
    # Define the initial Gaussian wavepacket.
    def psifun(x, y):
        # Parameters of the Gaussian
        x0, y0 = -2.0, 0.0  # initial center
        sigma = 0.8
        kx0 = 5.0  # momentum in x–direction
        ky0 = 0.0  # momentum in y–direction
        norm = 1.0 / (np.pi * sigma**2)**0.5
        return norm * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) * np.exp(1j * (kx0 * x + ky0 * y))
    
    # Simulation parameters.
    N = 50      # grid: 50x50 points (coarse grids often work well with split-step)
    L = 10.0     # spatial domain: x, y in [-L/2, L/2)
    nt = 300       # number of time steps
    dt = 0.05    # time step
    output_step = 1  # record every time step for the animation
    
    # Compute the time evolution.
    X, Y, t_vals, psi_rec = schrodinger2d_integrate(psifun, Vfun, nt, dt, N, L, output_step)
    
    # Set up the figure for animation.
    fig, ax = plt.subplots(figsize=(6, 5))
    # Compute initial probability density.
    density = np.abs(psi_rec[0])**2
    # pcolormesh requires X and Y arrays of shape (N,N). With shading='auto' the density array shape is accepted.
    quad = ax.pcolormesh(X, Y, density, shading='auto', cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f"t = {t_vals[0]:.2f}")
    fig.colorbar(quad, ax=ax, label='|ψ(x,y,t)|²')
    
    # Define the update function for the animation.
    def update(frame):
        # Update the probability density.
        new_density = np.abs(psi_rec[frame])**2
        # pcolormesh stores the facecolors in a flattened array.
        # (For shading='auto', the array length is (N-1)*(N-1).)
        quad.set_array(new_density.ravel())
        ax.set_title(f"t = {t_vals[frame]:.2f}")
        return quad,
    
    anim = FuncAnimation(fig, update, frames=200, interval=50, blit=False)
    plt.show()

def main():
    schrodinger1Ddemo()
    schrodinger2d_demo()

if __name__ == "__main__":
    main()