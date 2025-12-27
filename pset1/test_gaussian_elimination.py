import unittest
import numpy as np

from gaussian import pivot, row_reduce, gauss_elimination

class TestGaussianElimination(unittest.TestCase):

    # ------------------------------------------
    # Test pivot()
    # ------------------------------------------
    def test_pivot_no_swap_needed(self):
        """
        If the matrix is already set up with largest pivot in each row,
        pivot should not change the order of rows.
        """
        A = np.array([[3, 2], 
                      [1, 4]], dtype=float)
        b = np.array([1, 2], dtype=float)

        # Make copies for reference
        A_original = A.copy()
        b_original = b.copy()

        pivot(A, b, 2)
        # No swapping should occur
        self.assertTrue(np.array_equal(A, A_original), "Matrix A should remain unchanged when no swap is needed.")
        self.assertTrue(np.array_equal(b, b_original), "Vector b should remain unchanged when no swap is needed.")

    def test_pivot_swap_needed(self):
        """
        Check if pivot function correctly swaps rows when a larger pivot
        is in a lower row.
        """
        A = np.array([[1, 3],
                      [5, 2]], dtype=float)
        b = np.array([6, 7], dtype=float)

        pivot(A, b, 2)
        # Now row with '5' should be on top
        A_expected = np.array([[5, 2],
                               [1, 3]], dtype=float)
        b_expected = np.array([7, 6], dtype=float)

        self.assertTrue(np.array_equal(A, A_expected), "Matrix A pivot rows not swapped correctly.")
        self.assertTrue(np.array_equal(b, b_expected), "Vector b pivot rows not swapped correctly.")

    def test_pivot_multiple_swaps(self):
        """
        For a larger matrix, ensure pivot function performs row swaps for each column.
        """
        A = np.array([
            [2, 1, 3],
            [0, 10, 1],
            [15, 2, 1]], dtype=float
        )
        b = np.array([1, 2, 3], dtype=float)

        pivot(A, b, 3)
        # The largest pivot in the first column is in row 2 (value 15).
        # After swapping the first row with row 2, A should have 15 in the top-left.
        # Then for the second pivot, the largest pivot in column 1 among rows 1..2 is row 1 (itself).
        # So final pivot checks.

        A_expected = np.array([
            [15, 2, 1],
            [0, 10, 1],
            [2, 1, 3]], dtype=float
        )
        b_expected = np.array([3, 2, 1], dtype=float)

        self.assertTrue(np.array_equal(A, A_expected), "Matrix A not pivoted correctly with multiple swaps.")
        self.assertTrue(np.array_equal(b, b_expected), "Vector b not pivoted correctly with multiple swaps.")

    # ------------------------------------------
    # Test row_reduce()
    # ------------------------------------------
    def test_row_reduce_simple(self):
        """
        Basic test to ensure row reduction produces an upper triangular matrix.
        """
        A = np.array([[2, 1], [1, 3]], dtype=float)
        b = np.array([8, 13], dtype=float)
        n = 2
        
        row_reduce(A, b, n)
        
        # After forward elimination, A should be upper triangular:
        # pivot row is [2, 1], 
        # factor = (1/2) for second row,
        # second row becomes [1 - (1/2)*2, 3 - (1/2)*1] = [0, 2.5]
        # b becomes [8, 13 - (1/2)*8] = [8, 9]
        
        A_expected = np.array([[2, 1],
                               [0, 2.5]], dtype=float)
        b_expected = np.array([8, 9], dtype=float)
        
        # Use allclose with a tolerance for floating point
        self.assertTrue(np.allclose(A, A_expected), "Row reduction (A) did not produce expected upper triangular form.")
        self.assertTrue(np.allclose(b, b_expected), "Row reduction (b) did not produce expected vector results.")
        

    def test_row_reduce_singular_raises(self):
        """
        row_reduce should raise ValueError if a zero pivot is encountered (singular).
        """
        A = np.array([[0, 1], 
                      [2, 3]], dtype=float)
        b = np.array([1, 4], dtype=float)
        
        with self.assertRaises(ValueError):
            row_reduce(A, b, 2)

    def test_row_reduce_larger(self):
        """
        Test row reduction on a 3x3 to ensure upper triangular form is correct.
        """
        A = np.array([
            [2, 1, -1],
            [1, 1, -1],
            [-1, -1, 2]], dtype=float
        )
        b = np.array([1, 2, 1], dtype=float)
        n = 3

        row_reduce(A, b, n)
        # The resulting A should be upper triangular and b consistent with elimination steps.

        # We'll verify only that the sub-diagonal elements are zero (within floating tolerance),
        # because exact numeric checks might be more complex. You can do exact checks if you like.
        for i in range(n):
            for j in range(i):
                self.assertAlmostEqual(A[i, j], 0.0, places=7, 
                                       msg="Row reduction did not zero out sub-diagonal elements as expected.")

    # ------------------------------------------
    # Test gauss_elimination()
    # ------------------------------------------

     # ------------------------------------------
    def test_gauss_elimination_1x1(self):
        """
        Solve a simple 1x1 system, e.g., 4x = 8 => x = 2.
        """
        A = np.array([[4.0]], dtype=float)
        b = np.array([8.0], dtype=float)
        x = gauss_elimination(A, b)

        # Check the result
        self.assertEqual(len(x), 1, "Solution vector for 1x1 system should have length 1.")
        self.assertAlmostEqual(x[0], 2.0, msg="1x1 system solution is incorrect.")

    def test_gauss_elimination_0x0(self):
        """
        Solve an empty 0x0 system. Expect an empty solution vector.
        """
        A = np.array([], dtype=float).reshape(0,0)
        b = np.array([], dtype=float)
        x = gauss_elimination(A, b)

        # For a 0x0 system, we expect an empty solution
        self.assertEqual(x.size, 0, "0x0 system solution should be an empty array.")
    
    def test_gauss_elimination_simple(self):
        """
        Solve a simple 2x2 system and check correctness of the solution.
        """
        A = np.array([[2, 1], [1, 3]], dtype=float)
        b = np.array([8, 13], dtype=float)
        x = gauss_elimination(A, b)

        # Solve by hand: 2x + y = 8, x + 3y = 13
        # From the first equation, y = 8 - 2x
        # Substituting into the second: x + 3(8-2x) = 13 => x + 24 - 6x = 13 => -5x = -11 => x = 11/5 = 2.2
        # Then y = 8 - 2(2.2) = 3.6
        x_expected = np.array([2.2, 3.6])
        self.assertTrue(np.allclose(x, x_expected), 
                        f"Expected solution {x_expected}, got {x}.")

    def test_gauss_elimination_singular_raises(self):
        """
        Ensure gauss_elimination raises ValueError for a singular matrix
        """
        A = np.array([
            [1, 2],
            [2, 4]], dtype=float
        )
        b = np.array([3, 6], dtype=float)

        # This system is singular: second row is just 2x first row
        with self.assertRaises(ValueError):
            gauss_elimination(A, b)

    def test_gauss_elimination_3x3(self):
        """
        Test a 3x3 system whose solution is known.
        """
        A = np.array([
            [2, 1, -1],
            [1, 1, -1],
            [-1, -1, 2]], dtype=float
        )
        b = np.array([1, 2, 1], dtype=float)

        # Solve using our function
        x = gauss_elimination(A, b)

        # One can solve by a separate known method or compare with numpy.linalg.solve for correctness
        # We'll just do that to check correctness:
        A_copy = np.array([
            [2, 1, -1],
            [1, 1, -1],
            [-1, -1, 2]], dtype=float
        )
        b_copy = np.array([1, 2, 1], dtype=float)
        x_expected = np.linalg.solve(A_copy, b_copy)

        self.assertTrue(np.allclose(x, x_expected),
                        f"Expected solution {x_expected}, got {x}.")

    def test_gauss_elimination_random(self):
        """
        Generate a random non-singular system, solve it, 
        and compare to numpy.linalg.solve.
        """
        np.random.seed(0)  # for reproducibility
        A = np.random.rand(5, 5)
        # Ensure A is not singular by making it diagonally dominant
        for i in range(5):
            A[i, i] += 5

        b = np.random.rand(5)
        
        # Solve with our method
        A_copy = A.copy()
        b_copy = b.copy()
        x_custom = gauss_elimination(A_copy, b_copy)

        # Compare to numpy's solve
        x_np = np.linalg.solve(A, b)

        self.assertTrue(np.allclose(x_custom, x_np),
                        "Solution from gauss_elimination does not match numpy.linalg.solve for a random system.")

      # ------------------------------------------
    # New Tests for Complex Matrices and Vectors
    # ------------------------------------------

    # ------------------------------------------
    # Test pivot() with complex numbers
    # ------------------------------------------
    def test_pivot_complex_no_swap_needed(self):
        """
        If the complex matrix is already set up with largest pivot (by magnitude) in each row,
        pivot should not change the order of rows.
        """
        A = np.array([[3+4j, 2], 
                      [1, 4+3j]], dtype=complex)
        b = np.array([1+1j, 2+2j], dtype=complex)

        # Make copies for reference
        A_original = A.copy()
        b_original = b.copy()

        pivot(A, b, 2)
        # No swapping should occur
        self.assertTrue(np.array_equal(A, A_original), "Complex Matrix A should remain unchanged when no swap is needed.")
        self.assertTrue(np.array_equal(b, b_original), "Complex Vector b should remain unchanged when no swap is needed.")

    def test_pivot_complex_swap_needed(self):
        """
        Check if pivot function correctly swaps rows when a larger pivot
        (by magnitude) is in a lower row for complex matrices.
        """
        A = np.array([[1+1j, 3],
                      [2+2j, 2]], dtype=complex)
        b = np.array([6+6j, 7+7j], dtype=complex)

        pivot(A, b, 2)
        # Now row with '2+2j' should be on top
        A_expected = np.array([[2+2j, 2],
                               [1+1j, 3]], dtype=complex)
        b_expected = np.array([7+7j, 6+6j], dtype=complex)

        self.assertTrue(np.array_equal(A, A_expected), "Complex Matrix A pivot rows not swapped correctly.")
        self.assertTrue(np.array_equal(b, b_expected), "Complex Vector b pivot rows not swapped correctly.")

    def test_pivot_complex_multiple_swaps(self):
        """
        For a larger complex matrix, ensure pivot function performs row swaps for each column based on magnitude.
        """
        A = np.array([
            [1+1j, 2, 3],
            [4+4j, 5, 6],
            [7+7j, 8, 9]
        ], dtype=complex)
        b = np.array([1+1j, 2+2j, 3+3j], dtype=complex)

        pivot(A, b, 3)
        # The largest pivot in the first column is in row 2 (7+7j with magnitude sqrt(49 + 49) = sqrt(98))
        # Then the next largest in the second column among remaining rows, etc.

        A_expected = np.array([
            [7+7j, 8, 9],
            [4+4j, 5, 6],
            [1+1j, 2, 3]
        ], dtype=complex)
        b_expected = np.array([3+3j, 2+2j, 1+1j], dtype=complex)

        self.assertTrue(np.array_equal(A, A_expected), "Complex Matrix A not pivoted correctly with multiple swaps.")
        self.assertTrue(np.array_equal(b, b_expected), "Complex Vector b not pivoted correctly with multiple swaps.")

    # ------------------------------------------
    # Test row_reduce() with complex numbers
    # ------------------------------------------
    def test_row_reduce_complex_simple(self):
        """
        Basic test to ensure row reduction produces an upper triangular matrix for complex systems.
        """
        A = np.array([[2+0j, 1], [1, 3+0j]], dtype=complex)
        b = np.array([8+0j, 13+0j], dtype=complex)
        n = 2
        
        row_reduce(A, b, n)
        
        # After forward elimination, A should be upper triangular:
        # pivot row is [2, 1], 
        # factor = (1/2) for second row,
        # second row becomes [1 - (1/2)*2, 3 - (1/2)*1] = [0, 2.5]
        # b becomes [8, 13 - (1/2)*8] = [8, 9]
        
        A_expected = np.array([[2+0j, 1],
                               [0, 2.5+0j]], dtype=complex)
        b_expected = np.array([8+0j, 9+0j], dtype=complex)
        
        # Use allclose with a tolerance for floating point
        self.assertTrue(np.allclose(A, A_expected), "Row reduction (A) did not produce expected upper triangular form for complex numbers.")
        self.assertTrue(np.allclose(b, b_expected), "Row reduction (b) did not produce expected vector results for complex numbers.")

    def test_row_reduce_complex_singular_raises(self):
        """
        row_reduce should raise ValueError if a zero pivot is encountered (singular) in a complex matrix.
        """
        A = np.array([[0+0j, 1], 
                      [2, 3]], dtype=complex)
        b = np.array([1+0j, 4], dtype=complex)
        
        with self.assertRaises(ValueError):
            row_reduce(A, b, 2)

    def test_row_reduce_complex_larger(self):
        """
        Test row reduction on a 3x3 complex matrix to ensure upper triangular form is correct.
        """
        A = np.array([
            [2+0j, 1, -1],
            [1, 1+0j, -1],
            [-1, -1, 2+0j]
        ], dtype=complex)
        b = np.array([1+0j, 2, 1], dtype=complex)
        n = 3

        row_reduce(A, b, n)
        # The resulting A should be upper triangular and b consistent with elimination steps.

        # We'll verify only that the sub-diagonal elements are zero (within floating tolerance),
        # because exact numeric checks might be more complex. You can do exact checks if you like.
        for i in range(n):
            for j in range(i):
                self.assertAlmostEqual(A[i, j], 0.0+0j, places=7, 
                                       msg="Row reduction did not zero out sub-diagonal elements as expected for complex numbers.")

    # ------------------------------------------
    # Test gauss_elimination() with complex numbers
    # ------------------------------------------
    def test_gauss_elimination_complex_simple(self):
        """
        Solve a simple 2x2 complex system and check correctness of the solution.
        """
        A = np.array([[2+0j, 1], [1, 3+0j]], dtype=complex)
        b = np.array([8+0j, 13+0j], dtype=complex)
        x = gauss_elimination(A, b)

        # Solve by hand: 2x + y = 8, x + 3y = 13
        # From the first equation, y = 8 - 2x
        # Substituting into the second: x + 3(8-2x) = 13 => x + 24 - 6x = 13 => -5x = -11 => x = 11/5 = 2.2
        # Then y = 8 - 2(2.2) = 3.6
        x_expected = np.array([2.2+0j, 3.6+0j])
        self.assertTrue(np.allclose(x, x_expected), 
                        f"Expected complex solution {x_expected}, got {x}.")

    def test_gauss_elimination_complex_singular_raises(self):
        """
        Ensure gauss_elimination raises ValueError for a singular complex matrix
        """
        A = np.array([
            [1+1j, 2+2j],
            [2+2j, 4+4j]
        ], dtype=complex)
        b = np.array([3+3j, 6+6j], dtype=complex)

        # This system is singular: second row is just 2x first row
        with self.assertRaises(ValueError):
            gauss_elimination(A, b)

    def test_gauss_elimination_complex_3x3(self):
        """
        Test a 3x3 complex system whose solution is known.
        """
        A = np.array([
            [2+0j, 1, -1],
            [1, 1+0j, -1],
            [-1, -1, 2+0j]
        ], dtype=complex)
        b = np.array([1+0j, 2, 1], dtype=complex)

        # Solve using our function
        x = gauss_elimination(A, b)

        # Compare with numpy's solve
        A_copy = np.array([
            [2+0j, 1, -1],
            [1, 1+0j, -1],
            [-1, -1, 2+0j]
        ], dtype=complex)
        b_copy = np.array([1+0j, 2, 1], dtype=complex)
        x_expected = np.linalg.solve(A_copy, b_copy)

        self.assertTrue(np.allclose(x, x_expected),
                        f"Expected complex solution {x_expected}, got {x}.")

    def test_gauss_elimination_complex_random(self):
        """
        Generate a random non-singular complex system, solve it, 
        and compare to numpy.linalg.solve.
        """
        np.random.seed(1)  # for reproducibility
        A_real = np.random.rand(4, 4)
        A_imag = np.random.rand(4, 4)
        A = A_real + 1j * A_imag
        # Ensure A is not singular by making it diagonally dominant in magnitude
        for i in range(4):
            A[i, i] += 10 + 10j  # Adding a large complex number to the diagonal

        b_real = np.random.rand(4)
        b_imag = np.random.rand(4)
        b = b_real + 1j * b_imag
        
        # Solve with our method
        A_copy = A.copy()
        b_copy = b.copy()
        x_custom = gauss_elimination(A_copy, b_copy)

        # Compare to numpy's solve
        x_np = np.linalg.solve(A, b)

        self.assertTrue(np.allclose(x_custom, x_np),
                        "Complex solution from gauss_elimination does not match numpy.linalg.solve for a random complex system.")
if __name__ == '__main__':
    unittest.main()