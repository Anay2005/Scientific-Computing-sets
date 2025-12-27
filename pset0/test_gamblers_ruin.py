import unittest
import numpy as np
from io import StringIO
import sys

from gamblers_ruin import simulate, gamblers_ruin_stats, gamblers_ruin_trajectory_plot

class TestGamblersRuinSimulation(unittest.TestCase):

    def setUp(self):
        # Set a fixed seed for reproducibility in tests
        np.random.seed(42)

    def test_simulate_shape_and_values(self):
        """
        Test that the simulate function returns arrays of correct shape and that 
        the trajectories behave as expected for edge cases.
        """
        p = 0.5
        x_init = 10
        nsim = 5
        max_rounds = 100

        ruin_rounds, trajectories = simulate(p, x_init, nsim, max_rounds)

        # Check the shape of trajectories
        self.assertEqual(trajectories.shape, (nsim, max_rounds+1), 
                         "Trajectories array shape incorrect.")

        # Each simulation should start with initial capital
        for sim in range(nsim):
            self.assertEqual(trajectories[sim, 0], x_init, 
                             "Initial capital not set correctly in trajectory.")

        # Check that ruin rounds list has length <= nsim
        self.assertTrue(len(ruin_rounds) <= nsim, 
                        "Length of ruin rounds list exceeds number of simulations.")

    def test_simulate_ruin_at_zero_probability(self):
        """
        With p=0, gambler always loses and should reach ruin quickly.
        """
        p = 0.0
        x_init = 5
        nsim = 3
        max_rounds = 10

        ruin_rounds, trajectories = simulate(p, x_init, nsim, max_rounds)

        # Since p=0, the gambler should always lose until ruin in at most x_init rounds.
        for sim in range(nsim):
            # Find first occurrence of ruin in trajectory
            ruin_indices = np.where(trajectories[sim] <= 0)[0]
            self.assertTrue(len(ruin_indices) > 0, "Gambler did not ruin when p=0")
            # Check that ruin occurs no later than x_init rounds
            self.assertLessEqual(ruin_indices[0], x_init, 
                                 "Ruin did not occur within expected rounds for p=0")

    def test_simulate_no_ruin_when_p_one(self):
        """
        With p=1.0, gambler never loses a round. Therefore, no ruin should occur.
        """
        p = 1.0
        x_init = 10
        nsim = 2
        max_rounds = 20

        ruin_rounds, trajectories = simulate(p, x_init, nsim, max_rounds)

        # Check that ruin_rounds list contains all NaN values indicating no ruin
        for r in ruin_rounds:
            self.assertTrue(np.isnan(r), "Ruin occurred when p=1 should not happen.")
        
        # Check that trajectories never hit zero
        for sim in range(nsim):
            self.assertTrue(np.all(trajectories[sim] > 0), 
                            "Trajectory reached non-positive capital when p=1.")

    def test_consistency_with_fixed_seed(self):
        """
        Running simulate multiple times with the same seed should yield the same results.
        """
        p = 0.4
        x_init = 10
        nsim = 1
        max_rounds = 50

        # Capture output of two simulations with the same seed
        np.random.seed(123)
        ruin_rounds_1, trajectories_1 = simulate(p, x_init, nsim, max_rounds)

        np.random.seed(123)
        ruin_rounds_2, trajectories_2 = simulate(p, x_init, nsim, max_rounds)

        # Compare results
        np.testing.assert_array_equal(trajectories_1, trajectories_2,
                                      "Trajectories differ despite same seed.")
        self.assertEqual(ruin_rounds_1, ruin_rounds_2,
                         "Ruin rounds lists differ despite same seed.")

    def test_mean_calculation_in_stats(self):
        """
        Test a simple scenario in gamblers_ruin_stats by capturing input and output.
        This will simulate a simplified run with predefined inputs to the prompts.
        """
        # Redirect stdin and stdout to simulate user input and capture prints.
        user_input = StringIO("1\n10\n1\n0.5\n")
        sys.stdin = user_input
        
        # Since gamblers_ruin_stats produces plots and prompts for input,
        # we simply run it to ensure no runtime errors occur.
        try:
            gamblers_ruin_stats()
        except Exception as e:
            self.fail(f"gamblers_ruin_stats() raised an exception unexpectedly: {e}")
        finally:
            # Restore original stdin
            sys.stdin = sys.__stdin__

# Run the tests
if __name__ == '__main__':
    unittest.main()