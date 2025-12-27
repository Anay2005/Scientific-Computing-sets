
"""
In the so-called three-body problem, three objects whose masses are m0, m1, and m2, and whose
positions are r0, r1, and r2, interact gravitationally.

"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from pprint import pprint

import scipy.integrate

# function to do numerical integration of the three-body problem using scipy.integrate.ode
def three_body_integrate(m, r_init, v_init, t):

    """
    
    Observations on integration schemes:
      - Using 'dopri5' (an explicit Runge-Kutta method) works well.
      - Switching to a scheme like 'vode' or 'lsoda' may change how errors accumulate:
          • You may observe slight differences in trajectories over long time intervals,
            especially for chaotic regimes, as even small numerical differences can grow.
            However my eye couldn't notice any significant differences in the examples provided.
          • Energy conservation properties can also vary; symplectic integrators (not provided by ode)
            are often preferred for long-term orbital integrations because they better preserve invariants.
        
    """

    # Pack initial state: positions then velocities.
    # For three bodies in 2D, we have 3*2 = 6 position components and 6 velocity components.
    y0 = np.concatenate((r_init.flatten(), v_init.flatten()))
    
    def f(t, y, m):
        # Unpack the state vector into positions and velocities.
        positions = y[:6].reshape(3, 2)
        velocities = y[6:].reshape(3, 2)
        
        # Compute all pairwise differences between positions using broadcasting.
        # diffs[i, j] = positions[j] - positions[i]
        diffs = positions[None, :, :] - positions[:, None, :]
        
        # Extract the pairwise differences (skipping the diagonal, where i == j).
        # Order: for i=0: (r1-r0, r2-r0), i=1: (r0-r1, r2-r1), i=2: (r0-r2, r1-r2)
        pairwise_diffs = [diffs[i, j] for i in range(3) for j in range(3) if i != j]
        
        # Compute the gravitational accelerations (assuming G=1).
        # For body 0: contributions from bodies 1 and 2.
        a0 = (m[1] * pairwise_diffs[0] / np.linalg.norm(pairwise_diffs[0])**3 +
              m[2] * pairwise_diffs[1] / np.linalg.norm(pairwise_diffs[1])**3)
        # For body 1: contributions from bodies 0 and 2.
        a1 = (m[0] * pairwise_diffs[2] / np.linalg.norm(pairwise_diffs[2])**3 +
              m[2] * pairwise_diffs[3] / np.linalg.norm(pairwise_diffs[3])**3)
        # For body 2: contributions from bodies 0 and 1.
        a2 = (m[0] * pairwise_diffs[4] / np.linalg.norm(pairwise_diffs[4])**3 +
              m[1] * pairwise_diffs[5] / np.linalg.norm(pairwise_diffs[5])**3)
        
        # Combine the accelerations for each body into one array.
        accelerations = np.array([a0, a1, a2])
        
        # The time derivatives:
        # d(positions)/dt = velocities,
        # d(velocities)/dt = accelerations.
        dydt = np.concatenate((velocities.flatten(), accelerations.flatten()))
        return dydt

    # Set up the ODE solver.
    solver = scipy.integrate.ode(f)
    solver.set_integrator('lsoda')
    solver.set_initial_value(y0, t[0])
    solver.set_f_params(m)
    
    # Arrays to store the solution at each time step.
    sol = []
    times = []
    sol.append(y0.copy())
    times.append(t[0])
    
    # Integrate over the given time array.
    for time in t[1:]:
        solver.integrate(time)
        if not solver.successful():
            raise RuntimeError("Integration step failed at t = {}".format(time))
        sol.append(solver.y.copy())
        times.append(time)
    
    sol = np.array(sol)  # shape: (n_steps, 12)
    pprint(sol)
    
    # Separate the solution into positions and velocities.
    # sol[:, :6] means all rows, first 6 columns
    # sol[:, 6:] means all rows, columns last 6 columns
    # reshape(-1, 3, 2) means reshape the array to have 3 blocks of 2 columns each
    # -1 means the number of rows is inferred from the number of columns
    positions = sol[:, :6].reshape(-1, 3, 2)  # shape: (n_steps, 3, 2)
    velocities = sol[:, 6:].reshape(-1, 3, 2)   # shape: (n_steps, 3, 2)
    return positions, velocities
   
def three_body_demo():
    # A “hierarchical” system with two orbiting bodies and a smaller faraway third body
    m = [1,1,0.01]

    r_init = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 10.0]])

    v_init = np.array([[0.0, 0.5], [0.0, -0.5], [0.0, 0.3]])

    t = np.linspace(0.0, 100.0, 1000)

    positions,velocities = three_body_integrate(m, r_init, v_init, t)

    # Plot the trajectories
    plt.figure(figsize=(8,6))
    for i in range(3):
        plt.plot(positions[:, i, 0], positions[:, i, 1], label=f'Body {i}')
        plt.plot(positions[0, i, 0], positions[0, i, 1], 'o')  # start point
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Three-Body Problem with smaller faraway body')
    plt.legend()
    plt.grid(True)
    plt.show()

    # A figure 8 periodic orbit of three equal-mass bodies
    m = [1, 1, 1]

    # Positions:
    r_init = np.array([
        [ 0.97000436,  -0.24308753],
        [-0.97000436,  0.24308753],
        [ 0.0,         0.0]
    ])

    # Velocities:
    v_init = np.array([
        [ 0.46620368,  0.43236573],
        [ 0.46620368,  0.43236573],
        [-0.93240736, -0.86473146]
    ])

    # Time array for integration
    t = np.linspace(0.0, 10.0, 1000)  # You can adjust the end time

    # Integrate
    positions, velocities = three_body_integrate(m, r_init, v_init, t)

    # Plot
    plt.figure()
    for i in range(3):
        plt.plot(positions[:, i, 0], positions[:, i, 1], label=f'Body {i}')
        plt.plot(positions[0, i, 0], positions[0, i, 1], 'o')  # start point

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Three-Body Figure-8 Orbit (equal masses)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Pythagorean three-body problem:
    #  -- 3 masses at corners of a 3-4-5 triangle with zero initial velocity.
    
    m = [3.0, 4.0, 5.0]  # The ratio 3:4:5
    
    r_init = np.array([
        [0.0, 0.0],  # Body 0 (mass 3) at (0,0)
        [3.0, 0.0],  # Body 1 (mass 4) at (3,0)
        [0.0, 4.0]   # Body 2 (mass 5) at (0,4)
    ])
    
    # All velocities start at zero
    v_init = np.zeros((3, 2))
    
    # Integrate from t=0 to t=10 (or longer) in 1000 steps
    t = np.linspace(0, 10.0, 1000)
    
    # Use your existing integrator (for example, the 'three_body_integrate' you defined)
    positions, velocities = three_body_integrate(m, r_init, v_init, t)
    
    # Plot trajectories
    plt.figure(figsize=(8,6))
    for i in range(3):
        # positions[:, i, 0] means all time steps, body i, x-coordinate and 0th column has x-coordinates
        # positions[:, i, 1] means all time steps, body i, y-coordinate and 1st column has y-coordinates
        plt.plot(positions[:, i, 0], positions[:, i, 1], label=f'Body {i}')
        # This picks out the first time step and marks the initial position of each body
        plt.plot(positions[0, i, 0], positions[0, i, 1], 'o')  # Mark initial position
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Pythagorean Three-Body Problem')
    plt.legend()
    plt.grid(True)
    plt.show()


    

def three_body_spectrum_demo():
    """
        
    Observations:
    
      (i) Figure-8 Orbit (Periodic):
          - The FFT of the x-coordinate of one body reveals distinct, sharp peaks corresponding to the dominant periodicities but smooths out
            for higher angular velocity.
    
      (ii) Pythagorean Three-Body Problem (Chaotic):
          - The FFT in this case shows a broad range of frequencies, indicating a complex, aperiodic evolution.
          - The power spectrum’s wide spread is characteristic of chaotic systems where multiple frequencies interact.
    
    """
    # -----------------------------
    # (A) Setup 1: Figure-8 (Periodic)
    # -----------------------------
    # All masses = 1
    m_periodic = [1.0, 1.0, 1.0]
    
    # Known figure-8 initial conditions (positions & velocities)
    r_init_periodic = np.array([
        [ 0.97000436,  0.0       ],  # Body 0
        [-0.97000436,  0.0       ],  # Body 1
        [ 0.0       ,  0.0       ]   # Body 2
    ])
    v_init_periodic = np.array([
        [ 0.46620368,  0.43236573],  # Body 0
        [ 0.46620368,  0.43236573],  # Body 1
        [-0.93240736, -0.86473146]   # Body 2
    ])
    
    # We'll integrate from t=0 to t=20 in 2000 steps (you can extend if you like)
    t = np.linspace(0, 20, 2000)
    
    # Integrate
    pos_periodic, vel_periodic = three_body_integrate(m_periodic, r_init_periodic, v_init_periodic, t)
    
    # We'll analyze the x-coordinate of body 0
    x_data_periodic = pos_periodic[:, 0, 0]
    
    # Compute the FFT
    X_per = np.fft.fft(x_data_periodic)
    dt = t[1] - t[0]
    freqs = np.fft.fftfreq(len(x_data_periodic), d=dt)
    omega = 2.0 * np.pi * freqs
    power_periodic = np.abs(X_per)**2
    
    # Keep only the positive frequencies
    mask = freqs >= 0
    omega_pos_periodic = omega[mask]
    power_pos_periodic = power_periodic[mask]
    
    # -----------------------------
    # (B) Setup 2: Pythagorean (Chaotic or more complicated)
    # -----------------------------
    # Masses in ratio 3:4:5
    m_chaotic = [3.0, 4.0, 5.0]
    
    # Place them at corners of a 3-4-5 triangle with zero velocity
    r_init_chaotic = np.array([
        [ 0.0,  0.0],  # Body 0 (mass 3)
        [ 3.0,  0.0],  # Body 1 (mass 4)
        [ 0.0,  4.0],  # Body 2 (mass 5)
    ])
    v_init_chaotic = np.zeros((3, 2))
    
    # We can use the same time array or a separate one
    # Here let's just reuse t = [0..20] for demonstration
    pos_chaotic, vel_chaotic = three_body_integrate(m_chaotic, r_init_chaotic, v_init_chaotic, t)
    
    # x-coordinate of body 0 again
    x_data_chaotic = pos_chaotic[:, 0, 0]
    
    # FFT
    X_chaotic = np.fft.fft(x_data_chaotic)
    power_chaotic = np.abs(X_chaotic)**2
    
    # Positive frequencies
    omega_pos_chaotic = omega[mask]
    power_pos_chaotic = power_chaotic[mask]
    
    
    # -----------------------------
    # Plot Frequency-Domain (Power Spectrum)
    # -----------------------------
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax3.set_title("Power Spectrum (Figure-8) periodic")
    ax3.semilogy(omega_pos_periodic, power_pos_periodic, 'b-')

    ax3.set_xlabel(r"$\omega$")
    ax3.set_ylabel(r"$|X(\omega)|^2$")
    ax3.grid(True)
    
    ax4.set_title("Power Spectrum (Pythagorean) chaotic")
    ax4.semilogy(omega_pos_chaotic, power_pos_chaotic, 'r-')
    ax4.set_xlabel(r"$\omega$")
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()


    

def main():
    three_body_demo()
    three_body_spectrum_demo()

if __name__ == "__main__":
    main()