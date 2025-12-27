
This repository contains Python implementations of advanced numerical algorithms applied to problems in classical mechanics, quantum mechanics, statistical physics, and data analysis. The solutions demonstrate high-performance computing techniques, including Monte Carlo simulations, spectral methods, and sparse matrix operations.

## Dependencies
* **Python 3.x**
* **NumPy** (Vectorized numerical operations)
* **SciPy** (ODE solvers, sparse linear algebra, FFT)
* **Matplotlib** (Data visualization and animation)

---

## Project Structure

### Problem Set 0: Wave Physics & Stochastic Processes
**Focus:** Visualization and Random Walks
* **Interference Patterns:** Simulation of scalar monochromatic waves from multiple point sources. [cite_start]Implements 2D intensity heatmaps and time-dependent wave animations to visualize constructive and destructive interference [cite: 233-240, 271].
* **The Gambler's Ruin:** Monte Carlo simulation of a stochastic random walk (1D). [cite_start]Analyzes statistical properties such as the probability of ruin and the mean time to ruin, verifying theoretical predictions for "fair" and "unfair" games [cite: 274-279, 290-293].

### Problem Set 1: Linear Algebra & 1D Quantum Systems
**Focus:** Algorithm Complexity and Eigenvalue Problems
* **Gaussian Elimination:** Implementation of a linear solver from scratch, including row reduction and partial pivoting. [cite_start]Includes performance profiling (log-log analysis) to demonstrate $O(N^3)$ scaling compared to SciPy's optimized routines [cite: 327-330, 343, 359].
* **1D Tight-Binding Models:** Construction of Hamiltonians for discrete 1D quantum chains.
    * [cite_start]**Random Chain:** analysis of Anderson localization-like effects[cite: 412].
    * [cite_start]**Su-Schrieffer-Heeger (SSH) Model:** Study of topological phases and edge states in a dimerized chain[cite: 423, 442].

### Problem Set 2: Dynamical Systems & Time-Evolution
**Focus:** ODE Integration and Spectral Methods
* **Three-Body Problem:** Numerical solution of coupled differential equations for gravitational N-body systems using `scipy.integrate.ode`. [cite_start]Includes visualization of chaotic orbits and Fourier analysis (FFT) of trajectories to identify periodicity [cite: 475-479, 489, 516].
* **Split-Step Fourier Method:** Solver for the time-dependent Schrödinger equation (TDSE) in 1D and 2D. Utilizes Fast Fourier Transforms (FFT) to alternate between position and momentum space for efficient time-stepping. [cite_start]Features animations of wavepackets in harmonic and circular well potentials [cite: 521-524, 543, 548].

### Problem Set 3: 2D Quantum Statics
**Focus:** Sparse Matrices and Finite Difference Methods
* **2D Time-Independent Schrödinger Equation:** Finite-difference eigensolver for arbitrary 2D potentials (e.g., "Particle in a Box", "Stadium Billiard"). [cite_start]Utilizes `scipy.sparse` and Arnoldi iteration (`eigsh`) to efficiently solve for low-lying energy states [cite: 137-141, 151, 163].
* **Honeycomb Lattice Tight-Binding:** Simulation of complex 2D quantum lattices (Graphene-like structures). [cite_start]Implements magnetic vector potentials to study the effects of external magnetic fields on energy spectra [cite: 170, 182-189].

### Problem Set 4: Statistical Mechanics & Bayesian Inference
**Focus:** Markov Chain Monte Carlo (MCMC)
* **2D Ising Model:** Metropolis Monte Carlo simulation of spin systems on a square lattice. [cite_start]Calculates thermodynamic quantities (Magnetization, Heat Capacity, Susceptibility) and implements re-weighting techniques to estimate properties across temperatures from a single run [cite: 26-30, 45, 49].
* **Bayesian Parameter Estimation ($\Lambda$-CDM):** Application of MCMC to fit cosmological models to observational Hubble data. [cite_start]Estimates posterior distributions for the Hubble constant ($H_0$), matter density ($\Omega_m$), and dark energy density ($\Omega_{\Lambda}$) [cite: 59-64, 86].

---

## Usage
Each problem is contained in a separate Python script. To run a specific simulation, execute the file from the terminal:

```bash
python ps0_interference.py