"""
    Solve the linear system A x = b using Gaussian elimination with pivoting.
    Parameters:
        A (ndarray): Coefficient matrix of size (n x n).
        b (ndarray): Right-hand side vector of size (n,).
    Returns:
        x (ndarray): Solution vector of size (n,).
"""

import numpy as np
import scipy as sp
import timeit
import matplotlib.pyplot as plt
from scipy.stats import linregress


# function to perform row reduction in place
def row_reduce(A, b, n):
    # Forward Elimination: transform A into an upper triangular matrix
    for pivot_row in range(n):
        # Eliminate entries below the pivot (A[pivot_row, pivot_row])
        for row_to_reduce in range(pivot_row + 1, n):
            # If pivot is zero, the system is singular
            if A[pivot_row, pivot_row] == 0:
                raise ValueError("Matrix is singular.")
            
            factor = A[row_to_reduce, pivot_row] / A[pivot_row, pivot_row]

             # Subtract the factor * pivot_row from the current row to create a 0 in A[pivot_row, row_to_reduce]
            A[row_to_reduce, pivot_row:] -= factor * A[pivot_row, pivot_row:]
            b[row_to_reduce] -= factor * b[pivot_row]

# function to perform pivoting
# pivoting ensures that when a zero pivot is encountered, the row with the largest absolute value in the pivot column is swapped with the current row
def pivot(A, b, n):
    # Loop over each row
    for row in range(n):
        # Find the row with the largest absolute value in the pivot column
        pivot_row = np.argmax(np.abs(A[row:, row])) + row

        # Swap the rows in the augmented matrix
        A[[row, pivot_row]] = A[[pivot_row, row]]
        b[[row, pivot_row]] = b[[pivot_row, row]]
# function to perform back substitution and finc the solution
def gauss_elimination(A, b):
    x = np.zeros_like(b)
    n = len(b)
    pivot(A, b, n)
    row_reduce(A, b, n)
    # remember in range function the second argument is exclusive and the third argument is the step size
    for row in range(n-1, -1, -1):
        if A[row, row] == 0:
            raise ValueError("Matrix is singular.")
        # we perform the dot product of the row and the solution vector in the reverse order
        x[row] = (b[row] - np.dot(A[row, row+1:], x[row+1:])) / A[row, row]

    return x
# function to generate a random invertible matrix of given size
def generate_random_invertible_matrix(size):
    """
    Generate a random invertible matrix of given size.
    
    :param size: Size of the matrix 
    :param complex_type: If True, generate a complex matrix; otherwise real.
    :return: A random invertible matrix of shape (size, size).
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    while True:
        # decide randomly whether to generate a real or complex matrix
        complex_type = np.random.choice([True, False])

        if not complex_type:
            # Generate a random real matrix
            mat = np.random.randn(size, size)
        else:
            # Generate a random complex matrix
            mat = np.random.randn(size, size) + 1j * np.random.randn(size, size)
        
        if np.linalg.det(mat) != 0:
            return mat
        
def generate_random_vector(size):
    """
    Generate a random vector of given size.
    
    :param size: Size of the vector.
    :return: A random vector of shape (size,).
    """
    return np.random.randn(size)
# Funtion to compare the runtime of the Gaussian elimination with the numpy.linalg.solve function and plot using log plot
def gaussian_eliminate_profile():
    """
    Compare the runtime of:
      1) A custom Gaussian Elimination implementation 
      2) `scipy.linalg.solve`

    Over matrix sizes N from 10 to 60, each timed over 1000 calls.

    On a log–log scale, we fit lines:
      - Gauss data => slope ~ 1.83 (example)
      - scipy data => slope ~ 0.91 (example)

    Observations:
      1. Both methods theoretically scale as O(N^3). The fitted slopes 
         are lower (~1.83 vs. ~0.91) due to overheads and the limited
         range of N. For truly large N, we'd expect both slopes to 
         approach ~3. 
      2. `scipy.linalg.solve` is consistently faster (lower curve). This 
         reflects highly optimized underlying BLAS/LAPACK routines compared 
         to pure Python loops in the custom approach.
      3. The best-fit lines look fairly “clean” because we only have 
         two methods performing essentially the same type of factorization 
         (LU or equivalent). Each data series has relatively little scatter.

    Why does this produce a cleaner fit than, for example, profiling 
    multiple distinct operations (det, eig, expm)?
      - Fewer algorithms: We only compare two routines that each solve
        a linear system. In the broader matrix profiling (det/eigvals/expm), 
        each operation uses a different algorithm, with different overhead 
        and scaling patterns, producing more varied scatter.
      - More consistent routines: Both are essentially do-lu-factor-and-solve.
      - Less variation: We measure many calls at each size with the same 
        operation, reducing random fluctuations.
      - Wider range + repeated calls: We collect more robust data, so the 
        regression in log–log space yields a neat, tight fit.
    """
    # Define the range of problem sizes
    N = np.arange(50,100)

    # Lists to store timing results (no repeated trials, no median)
    times_gauss = []
    times_scipy = []

    # For each matrix size n, time both methods once
    for n in N:
        A = generate_random_invertible_matrix(n)
        b = generate_random_vector(n)

        # Time your custom Gaussian elimination for 100 calls
        t_gauss = timeit.timeit(lambda: gauss_elimination(A.copy(), b.copy()), number=50)
        times_gauss.append(t_gauss)

        # Time scipy.linalg.solve for 100 calls
        t_scipy = timeit.timeit(lambda: sp.linalg.solve(A.copy(), b.copy()), number=50)
        times_scipy.append(t_scipy)

    # Convert lists to NumPy arrays
    times_gauss = np.array(times_gauss)
    times_scipy = np.array(times_scipy)

    # Prepare for log–log linear regression
    logN = np.log(N)

    # Gaussian elimination regression
    logT_gauss = np.log(times_gauss)
    slope_g, intercept_g, r_g, p_g, std_g = linregress(logN, logT_gauss)

    # scipy.linalg.solve regression
    logT_scipy = np.log(times_scipy)
    slope_s, intercept_s, r_s, p_s, std_s = linregress(logN, logT_scipy)

    # Generate a smooth range for plotting best-fit lines
    N_fit = np.linspace(N.min(), N.max(), 200)
    T_fit_gauss = np.exp(intercept_g) * (N_fit**slope_g)
    T_fit_scipy = np.exp(intercept_s) * (N_fit**slope_s)

    # ---- Plotting ----
    plt.figure(figsize=(8,6))

    # Plot raw data (Gaussian elimination)
    plt.loglog(N, times_gauss, 'o', markersize=1, label='Gauss data')
    plt.loglog(N_fit, T_fit_gauss, '-', 
               label=f"Gauss fit (slope={slope_g:.2f})")

    # Plot raw data (scipy.linalg.solve)
    plt.loglog(N, times_scipy, 's', markersize=1, label='scipy data')
    plt.loglog(N_fit, T_fit_scipy, '-', 
               label=f"scipy fit (slope={slope_s:.2f})")

    plt.title("Comparison: Gaussian Elimination vs scipy.linalg.solve (log–log scale)")
    plt.xlabel("Matrix size N")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.tight_layout()
    plt.show()
# function to profile the performation of numpy/scipy functions for matrix operations
def matrices_profile():
    """
    Profile the performance of NumPy/Scipy functions for various matrix operations
    (eigvals, eig, det, expm) over matrix sizes from 1 to 50.
    
    Observations (from the final plot):
    1. det appears with the lowest measured slope (≈ 0.71) for small range of problem sizes, though it converges to
      O(N^3) for large range of N. Overheads and highly optimized BLAS can make it look
       'faster' in log–log slope terms for small N.
    2. expm has a slope around 1.03 for small range of problem sizes, but it converges to O(N^3) matrix multiplications for larger range.
       Internally (e.g., scaling & squaring). For these smaller sizes, overheads
       + vectorization can reduce the effective slope.
    3. eigvals/eig show slopes around 1.5–1.54 for small range of problem sizes, it converges to O(N^3) routine
       (QR algorithm). Again, over small N, you won't see slope=3 exactly, due to
       similar overhead effects.
    4. In true “big N” asymptotic sense, these operations often converge to O(N^3).
       For the relatively small range here, best-fit slopes in log–log plots are
       below 3, reflecting overhead and optimization details.
    """
    # Define the problem sizes
    N = np.arange(50, 100)

    # Dictionary to hold { "operation_name": function(A) }:
    operations = {
        'eigvals':  lambda A: np.linalg.eigvals(A),
        'eig':      lambda A: np.linalg.eig(A),
        'det':      lambda A: np.linalg.det(A),
        'expm':     lambda A: sp.linalg.expm(A)
    }

    # Dictionary to store times for each operation: { op_name: [times...] }
    times = {op_name: [] for op_name in operations.keys()}

    # Time each operation for each size n
    for n in N:
        A = generate_random_invertible_matrix(n)
        for op_name, op_func in operations.items():

            elapsed = timeit.timeit(lambda: op_func(A), number=100)
            times[op_name].append(elapsed)

    # Convert times lists to NumPy arrays 
    for op_name in times:
        times[op_name] = np.array(times[op_name])

    # Create subplots (2x2 for 4 operations)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()  

    # For consistent best-fit lines, define a fine range for N in linear space
    N_fit = np.linspace(N.min(), N.max(), 200)  
    logN_fit = np.log(N_fit)


    for ax, (op_name, t_array) in zip(axes, times.items()):
        # Perform linear regression in ln–ln space:
        # log(t) = slope * log(N) + intercept
        logN = np.log(N)
        logT = np.log(t_array)
        slope, intercept, r_value, p_value, std_err = linregress(logN, logT)

        # Compute best-fit line for these N_fit values:
        # t_fit = exp(intercept) * N_fit^slope
        t_fit = np.exp(intercept) * (N_fit ** slope)

        # Plot raw data on log–log scale
        ax.loglog(N, t_array, 'o', markersize=1, linestyle='none', label=f"{op_name} data")

        # Plot best-fit line
        ax.loglog(N_fit, t_fit, '-', label=f"best-fit\nslope={slope:.2f}")

        ax.set_title(op_name)
        ax.legend()
        ax.set_xlabel("N")
        ax.set_ylabel("Time [s]")

    fig.suptitle("Matrix Operation Profiling: Data + Best-Fit Lines")
    fig.tight_layout()
    plt.show()
   


def main():
    # Example run
    A = np.array([[2, 1, -1], [1, 1, -1], [-1, -1, 2]], dtype=float)
    b = np.array([1, 2, 1], dtype=float)
    x = gauss_elimination(A, b)
    print(x)

    # Profile the Gaussian elimination function
    gaussian_eliminate_profile()
    # Profile the matrix operations
    matrices_profile()






if __name__ == "__main__":
    main()