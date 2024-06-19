# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def uavs_initialization(N, random_seed, x_range, y_range):
    """
    Initialize UAVs with random (x, y) positions.

    Parameters:
    N (int): Number of UAVs.
    random_seed (int): Seed for the random number generator.
    x_range (tuple): Range for the x coordinates (min, max).
    y_range (tuple): Range for the y coordinates (min, max).

    Returns:
    list: A list of tuples representing the (x, y) positions of the UAVs.
    """
    np.random.seed(random_seed)
    uavs_list = []

    for _ in range(N):
        x = np.random.randint(x_range[0], x_range[1] + 1)
        y = np.random.randint(y_range[0], y_range[1] + 1)
        uavs_list.append((x, y))

    return uavs_list

def uav_plotting(initial_positions, x_range, y_range, title, final_positions=None):
    """
    Plot the initial and final positions of UAVs.

    Parameters:
    initial_positions (list): List of tuples representing the initial (x, y) positions of the UAVs.
    x_range (tuple): Range for the x coordinates (min, max).
    y_range (tuple): Range for the y coordinates (min, max).
    title (str): Title of the plot.
    final_positions (list, optional): List of tuples representing the final (x, y) positions of the UAVs.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    
    for (x, y) in initial_positions:
        plt.plot(x, y, "bo")
    plt.plot([], [], "bo", label="Initial Positions")
    
    if final_positions is not None:
        for (x, y) in final_positions:
            plt.plot(x, y, "rx")
        plt.plot([], [], "rx", label="Final Positions")
    
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.title(title)
    plt.xlabel("Horizontal positions")
    plt.ylabel("Height")
    plt.grid(True)
    plt.legend()
    plt.show()

def get_n_nearest_neighbors(n_neighbors, uav_index, uavs_positions):
    """
    Get the indices of the n nearest neighbors to a UAV.

    Parameters:
    n_neighbors (int): Number of nearest neighbors to find.
    uav_index (int): Index of the UAV for which to find the neighbors.
    uavs_positions (list): List of tuples representing the (x, y) positions of all UAVs.

    Returns:
    np.ndarray: Array of indices of the n nearest neighbors.
    """
    uavs_positions = np.array(uavs_positions)
    uav_position = uavs_positions[uav_index]
    distances = np.linalg.norm(uavs_positions - uav_position, axis=1)
    nearest_neighbors_indices = np.argsort(distances)[:n_neighbors + 1]
    
    return np.sort(nearest_neighbors_indices).astype(int)

def get_random_neighbors(n_uavs, uav_index, random_seed, n_neighbors=None):
    """
    Get random neighbors for a UAV.

    Parameters:
    n_uavs (int): Total number of UAVs.
    uav_index (int): Index of the UAV for which to find the neighbors.
    random_seed (int): Seed for the random number generator.
    n_neighbors (int, optional): Number of neighbors to select randomly.

    Returns:
    np.ndarray: Array of indices of the random neighbors.
    """
    np.random.seed(random_seed)
    if not n_neighbors:
        neighbors = np.random.choice(np.delete(np.arange(n_uavs), uav_index), size=np.random.randint(1, n_uavs), replace=False)
    else:
        neighbors = np.random.choice(np.delete(np.arange(n_uavs), uav_index), size=n_neighbors, replace=False)
        
    if uav_index not in neighbors:
        neighbors = np.append(neighbors, uav_index)

    return np.sort(neighbors).astype(int)

def update_height(heights, neighbors, protocol):
    """
    Update the height of a UAV based on the specified protocol.

    Parameters:
    heights (list): List of heights of all UAVs.
    neighbors (np.ndarray): Indices of the neighboring UAVs.
    protocol (str): Protocol to use for updating the height ('arithmetic_mean', 'geometric_mean', 'harmonic_mean', 'mean_of_order_2').

    Returns:
    float: The updated height.
    """
    neighbors_heights = np.array(list(map(lambda x: heights[x], neighbors)))
    if protocol == "arithmetic_mean":
        return np.mean(neighbors_heights)
    elif protocol == "geometric_mean":
        return np.exp(np.mean(np.log(neighbors_heights)))
    elif protocol == "harmonic_mean":
        return len(neighbors) / np.sum(1.0 / neighbors_heights)
    elif protocol == "mean_of_order_2":
        return (np.sum(neighbors_heights ** 2) / len(neighbors)) ** 0.5
    else:
        raise ValueError("Unknown protocol")

def convergence_check(uavs_positions, iteration):
    """
    Check if all UAVs have converged to the same height.

    Parameters:
    uavs_positions (list): List of tuples representing the (x, y) positions of all UAVs.
    iteration (int): The current iteration number.

    Returns:
    int or None: The iteration number if convergence has occurred, otherwise None.
    """
    heights = [position[1] for position in uavs_positions]
    int_heights = np.array(heights, dtype=int)
    if np.all(int_heights == int_heights[0]):
        return iteration

    return None

def plot_results(result, protocol, n_uavs, n_iterations):
    """
    Plot the heights of UAVs over iterations.

    Parameters:
    result (list): List of UAV positions over iterations.
    protocol (str): Protocol used for the simulation.
    n_uavs (int): Number of UAVs.
    n_iterations (int): Number of iterations.

    Returns:
    None
    """
    heights = np.zeros((n_iterations, n_uavs))
    for t in range(n_iterations):
        for i in range(n_uavs):
            heights[t, i] = result[t][i][1]

    plt.figure(figsize=(10, 6))
    for i in range(n_uavs):
        plt.plot(heights[:, i], label=f'UAV {i+1}')
    plt.title(f'Vertical Alignment of UAVs using {protocol.replace("_", " ").title()} Protocol')
    plt.xlabel('Time Steps')
    plt.ylabel('Height')
    plt.legend()
    plt.grid(True)
    plt.show()

def simulation(n_uavs, uavs_list, n_iterations, protocol, topology, n_neighbors, random_seed, random_neighbors_flag=True):
    """
    Run the UAV simulation.

    Parameters:
    n_uavs (int): Number of UAVs.
    uavs_list (list): List of tuples representing the initial (x, y) positions of the UAVs.
    n_iterations (int): Number of iterations.
    protocol (str): Protocol to use for updating the height ('arithmetic_mean', 'geometric_mean', 'harmonic_mean', 'mean_of_order_2').
    topology (str): Topology to use ('all_to_all', 'n_nearest_neighbors', 'random').
    n_neighbors (int): Number of neighbors to consider for the 'n_nearest_neighbors' and 'random' topologies.
    random_seed (int): Seed for the random number generator.
    random_neighbors_flag (bool): Flag indicating whether to use random neighbors.

    Returns:
    tuple: A tuple containing the result (list of UAV positions over iterations) and the convergence iteration number (or None if no convergence).
    """
    uavs_positions = uavs_list.copy()
    result = [uavs_positions.copy()]
    convergence_iteration = None

    for iteration in range(n_iterations):
        next_uavs_positions = uavs_positions.copy()

        for uav_index, uav_position in enumerate(uavs_positions):
            if topology == "all_to_all":
                neighbors = np.arange(n_uavs).astype(int)
            elif topology == "n_nearest_neighbors":
                neighbors = get_n_nearest_neighbors(n_neighbors, uav_index, uavs_positions)
            elif topology == "random":
                if random_neighbors_flag:
                    neighbors = get_random_neighbors(n_uavs, uav_index, random_seed)
                else:
                    neighbors = get_random_neighbors(n_uavs, uav_index, random_seed, n_neighbors)

            heights = [next_uavs_positions[i][1] for i in range(n_uavs)]
            new_height = update_height(heights, neighbors, protocol)
            next_uavs_positions[uav_index] = (next_uavs_positions[uav_index][0], new_height)

        uavs_positions = next_uavs_positions
        if not convergence_iteration:
            convergence_iteration = convergence_check(uavs_positions, iteration)

        result.append(uavs_positions.copy())
        
    return result, convergence_iteration

def results_dataframe_analysis(dataframe, target_column_name, numerical_column_name, analysis_criterion, max_value=None):
    """
    Analyze the results in the DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame.
    target_column_name (str or list): Column name(s) to group by.
    numerical_column_name (str): Column name of the numerical values.
    analysis_criterion (str): Criterion to use for analysis ('mean' or 'median').
    max_value (float or None): The value to replace None values in numerical_column_name. If None, uses float('inf').

    Returns:
    tuple: A tuple containing the best and worst elements based on the analysis criterion.
    """
    if analysis_criterion not in ["mean", "median"]:
        raise ValueError("analysis_criterion must be 'mean' or 'median'")

    original_none_mask = dataframe[numerical_column_name].isna().copy()
    
    if max_value is None:
        max_value = float("inf")

    dataframe[numerical_column_name] = dataframe[numerical_column_name].apply(lambda x: max_value if pd.isna(x) else x)
    
    grouped = dataframe.groupby(target_column_name)[numerical_column_name].agg([analysis_criterion]).reset_index()
    min_value = grouped[analysis_criterion].min()
    max_value_calculated = grouped[analysis_criterion].max()
    
    best_elements = grouped[grouped[analysis_criterion] == min_value]
    worst_elements = grouped[grouped[analysis_criterion] == max_value_calculated]
    
    best_element_list = [{"target_column_name": row[target_column_name], "value": row[analysis_criterion]} for index, row in best_elements.iterrows()]
    
    worst_element_list = []
    for index, row in worst_elements.iterrows():
        value = row[analysis_criterion]
        if max_value != float("inf") and original_none_mask[dataframe[target_column_name] == row[target_column_name]].any():
            value = f"{max_value}+"
        elif value == max_value:
            value = f"{max_value}+"
        worst_element_list.append({"target_column_name": row[target_column_name], "value": value})
    
    return best_element_list, worst_element_list

def display_results(title, target, best_means, worst_means, best_medians, worst_medians, double_line, line):
    """
    Display the results of the analysis in a structured format.

    Parameters:
    title (str): The title of the results section.
    target (str): The target of the analysis (e.g., protocol, topology).
    best_means (list of dict): The best elements based on mean convergence iteration.
    worst_means (list of dict): The worst elements based on mean convergence iteration.
    best_medians (list of dict): The best elements based on median convergence iteration.
    worst_medians (list of dict): The worst elements based on median convergence iteration.
    double_line (str): The line used for major separation (e.g., a string of equal signs).
    line (str): The line used for minor separation (e.g., a string of dashes).

    Returns:
    None

    This function prints the best and worst elements based on mean and median convergence iteration
    in a structured and readable format.
    """
    print(double_line)
    print(title)
    print(double_line)

    print(f"The best {target} by mean convergence iteration:")
    best_mean_value = round(best_means[0]["value"], 2)
    for best_mean in best_means:
        print(best_mean["target_column_name"])
    print(f"The best {target} mean value: {best_mean_value}")
    print(line)

    print(f"The worst {target} by mean convergence iteration:")
    try:
        worst_mean_value = round(worst_means[0]["value"], 2)
    except:
        worst_mean_value = worst_means[0]["value"]
    for worst_mean in worst_means:
        print(worst_mean["target_column_name"])
    print(f"The worst {target} mean value: {worst_mean_value}")
    print(line)

    print(f"The best {target} by median convergence iteration:")
    best_median_value = round(best_medians[0]["value"], 2)
    for best_median in best_medians:
        print(best_median["target_column_name"])
    print(f"The best {target} median value: {best_median_value}")
    print(line)

    print(f"The worst {target} by median convergence iteration:")
    try:
        worst_median_value = round(worst_medians[0]["value"], 2)
    except:
        worst_median_value = worst_medians[0]["value"]
    for worst_median in worst_medians:
        print(worst_median["target_column_name"])
    print(f"The worst {target} median value: {worst_median_value}")
    print(line)



