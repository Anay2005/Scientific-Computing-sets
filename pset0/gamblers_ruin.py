import numpy as np
import matplotlib.pyplot as plt

def gamblers_ruin_trajectory_plot(p, x_init, nsim, max_rounds):

    # Assert checks
    assert 0 <= p <= 1, "Probability p must be between 0 and 1 (inclusive)."
    assert isinstance(x_init, float) or isinstance(x_init, int), "Initial capital x_init must be a real number (float or int)."
    assert isinstance(nsim, int) and nsim > 0, "Number of simulations nsim must be a positive integer."
    assert isinstance(max_rounds, int) and max_rounds > 0, "max_rounds must be a positive integer."

    # set the graph
    plt.figure(figsize=(10, 6))
    plt.axhline(y=0, color='red', linestyle='--', label="Ruin Line")
    plt.title("Gambler's Ruin Trajectories")
    plt.xlabel("Rounds (t)")
    plt.ylabel("Capital (x)")
    plt.grid()

    _, trajectories = simulate(p, x_init, nsim, max_rounds)
    # Loop over each simulation and plot its trajectory
    for sim in range(nsim):
        plt.plot(trajectories[sim, :], label=f"Sim {sim+1}", alpha=0.6)


    plt.legend()
    plt.show()
    

def simulate(p, x_init, nsim, max_rounds):
    """
    Simulates the gambler's ruin trajectory.
    
    Parameters:
        p (float): Probability of winning each round.
        x_init (int): Initial capital of the gambler.
        nsim (int): Number of independent simulations.
        max_rounds (int): Maximum number of rounds to simulate.
    """

    # Assert checks
    assert 0 <= p <= 1, "Probability p must be between 0 and 1 (inclusive)."
    assert isinstance(x_init, float) or isinstance(x_init, int), "Initial capital x_init must be a real number (float or int)."
    assert isinstance(nsim, int) and nsim > 0, "Number of simulations nsim must be a positive integer."
    assert isinstance(max_rounds, int) and max_rounds > 0, "max_rounds must be a positive integer."

    # The indices of the rows are simulation number and the indices of columns are round numbers
    trajectories = np.full((nsim, max_rounds+1), x_init)
    # List or array to collect ruin rounds for each simulation
    all_ruin_rounds = []
    for sim in range(nsim):
        

        # simulate one trajectory, get a random sequence of -1 and 1 with probabilities p and 1-p, length of array is max_rounds
        outcomes = np.random.choice([-1, 1], size=max_rounds, p=[1-p, p])
        # Calculate the cumulative sum of outcomes, the total winnings at each round
        cumulative_outcomes = np.cumsum(outcomes)
        # Calculate the gambler's capital at each round
        trajectory = np.concatenate([[x_init], x_init + cumulative_outcomes])
        ruin_idx = np.where(trajectory <= 0)[0]

        # get the first ruin round number
        first_ruin = ruin_idx[0] if ruin_idx.size >= 1 else np.nan
        if ruin_idx.size > 0:
            # fill entire computed trajectory:
            trajectories[sim, :trajectory.size] = trajectory
            all_ruin_rounds.append(first_ruin) 
            # Everything after the first ruin is 0
            # but only if first_ruin < trajectory.size
            if first_ruin < trajectory.size:
                trajectories[sim, first_ruin:] = 0
        else:
            # no ruin
            trajectories[sim, :trajectory.size] = trajectory

    # Return collected ruin rounds data
    return all_ruin_rounds, trajectories


def gamblers_ruin_stats():
    # we will plot a histogram of the number of rounds(T) until ruin for p<=0.5
    p = 0.4
    x_init = 50
    nsim = 1000
    max_rounds = 5000

    # Assert checks for this specific setup
    assert 0 <= p <= 0.5, "Probability p must be between 0 and 0.5 (inclusive)."
    assert isinstance(x_init, float) or isinstance(x_init, int), "Initial capital x_init must be a real number (float or int)."
    assert isinstance(nsim, int) and nsim > 0, "nsim must be a positive integer."
    assert isinstance(max_rounds, int) and max_rounds > 0, "max_rounds must be a positive integer."
    # Explanation of parameters:
    # p <= 0.5 ensures that the gambler has a fair chance of winning or losing each round otherwise insuffcient ruin_rounds data would be collected to make plots.
    # x_init = 50 provides an initial capital to observe the ruin process.
    # nsim = 1000 offers a large sample size for reliable statistical analysis.
    # max_rounds = 5000 is set sufficiently high to ensure that most simulations reach ruin
    
    # Now ruin_rounds contains the first ruin round for each simulation
    ruin_rounds, _ = simulate(p, x_init, nsim, max_rounds)
    
    # Create a histogram 
    plt.figure(figsize=(10, 6))
    # plt.hist expects an array like input
    plt.hist(ruin_rounds, bins=30, edgecolor='black')
    plt.title("Histogram of Rounds Until Ruin")
    plt.xlabel("Rounds until Ruin (T)")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()
   

    # Now we will plot mean T vs p for multiple values of x_init

    # Ask for the no.of x_init values 
    x_init_values_count = int(input("How many initial capital values do you need the simulation for?: "))
    assert x_init_values_count > 0, "Number of initial capital values must be a positive integer."
    print("\n")

    # place to store the x_init values
    x_init_values = []
    # Ask the user to enter the x_init values
    for _ in range(x_init_values_count):
        x_init_value = float(input("Please enter the initial capital values: "))
        assert isinstance(x_init, float) or isinstance(x_init, int), "Initial capital x_init must be a real number (float or int)."
        x_init_values.append(x_init_value)

    print("\n")
    # generate p values
    p_values_generated = np.linspace(0.0, 0.5, 50)
    # set up the plot
    plt.figure(figsize=(10, 6))
    plt.title("Mean Rounds to Ruin vs. p for Different x_init")
    plt.xlabel("p")
    plt.ylabel("⟨T⟩")
    plt.grid()
    # simulate for multiple x_init, the outer for loop is intended for run for small number of x_init values
    for count in range(x_init_values_count):
        # make a place to store mean of all_ruin_rounds
        mean_T = []
        # make a place to store uncertainity of each T
        error_bars = [] 
        for p in p_values_generated:
            
            ruin_rounds, _ = simulate(p, x_init_values[count], nsim, max_rounds)
            # filter out the nan values
            # np.isnan(r) checks if a given value is nan(return true) or not(returns false)
            valid_rounds = [r for r in ruin_rounds if not np.isnan(r)]
            # checks if valid_rounds is non_empty
            if valid_rounds:
                mean_val = np.mean(valid_rounds)
                std_val = np.std(valid_rounds)
                se = std_val / np.sqrt(len(valid_rounds))
            else:
                mean_val = np.nan
                se = np.nan
            # store the mean value of T from each sim
            mean_T.append(mean_val)
            # Now calculate and append the error(std deviation of T)
            error_bars.append(se)

        # Plot error bars for this x_init
        # The uncertainty in the ⟨T⟩ plots is represented by the standard error of the mean.
        plt.errorbar(p_values_generated, mean_T, yerr=error_bars, fmt='-o', capsize=3,
                     label=f"x_init = {x_init_values[count]}", alpha=0.7, markersize=4)
        

   
    plt.legend()
    plt.show()


    # Now we will plot mean T versus initial capital x_0 for one or more choices of p with error bars

    # generate x_0 values
    x_0_values_generated = np.linspace(1, 50, 50)
    plt.figure(figsize=(10, 6))
    plt.title("Mean Rounds to Ruin vs. x_0 for Different p")
    plt.xlabel("x_0")
    plt.ylabel("⟨T⟩")
    plt.grid()

    
    # Ask for the no.of probability values 
    p_values_count = int(input("How many probability values do you need the simulation for?: "))
    assert p_values_count > 0, "Number of probability values must be a positive integer."
    print("\n")
    # place to store the p values
    p_values_inputted = []
    # Ask the user to enter the probability values
    for _ in range(p_values_count):
        p_value = float(input("Please enter the probability values: "))
        assert 0 <= p_value <= 0.5, "Probability p must be between 0 and 1 (inclusive)."
        p_values_inputted.append(p_value)

    print("\n")

    # For multiple 'p' values run simulations for all the x_values generated, the outer for loop is intended for run for small number of p values
    for p_count in range(p_values_count):
        mean_T = []      # Reset mean_T for the current probability
        error_bars = []  # Reset error_bars for the current probability

        # For each x_0 value, run simulations at the current probability to compute statistics
        for x_0 in x_0_values_generated:
            
            ruin_rounds, _ = simulate(p_values_inputted[p_count], x_0, nsim, max_rounds)
            # Remove out the nan values
            # np.isnan(r) checks if a given value is nan(return true) or not(returns false)
            valid_rounds = [r for r in ruin_rounds if not np.isnan(r)]
            # checks if valid_rounds is non_empty
            if valid_rounds:
                mean_val = np.mean(valid_rounds)
                std_val = np.std(valid_rounds)
                se = std_val / np.sqrt(len(valid_rounds))
            else:
                mean_val = np.nan
                se = np.nan
                
            mean_T.append(mean_val)
            error_bars.append(se)
        
        # Plot the results for this probability value
        plt.errorbar(x_0_values_generated, mean_T, yerr=error_bars, fmt='-o', capsize=1,markersize=2, alpha=0.7, label=f"p = {p_values_inputted[p_count]}")

    plt.legend()
    plt.show()
   

    


    


def main():
    # Parameters for the simulation and trajectory plot
    p = 0.6
    x_init = 100
    nsim = 500
    max_rounds = 1000

    # Assert checks
    assert 0 <= p <= 1, "Probability p must be between 0 and 1 (inclusive)."
    assert isinstance(x_init, float) or isinstance(x_init, int), "Initial capital x_init must be a real number (float or int)."
    assert isinstance(nsim, int) and nsim > 0, "Number of simulations nsim must be a positive integer."
    assert isinstance(max_rounds, int) and max_rounds > 0, "max_rounds must be a positive integer."
    # Simulate and plot the gambler's ruin trajectory
    gamblers_ruin_trajectory_plot(p, x_init, nsim, max_rounds)
    # Simulate and plot the gambler's ruin trajectory and make 3 different plots
    gamblers_ruin_stats()


if __name__ == "__main__":
    main()
