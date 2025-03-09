"""
Diffusion Limited Aggregation (DLA) Simulation

This module simulates the growth of a DLA cluster with varying sticking probabilities (p_s).
It models the aggregation process where particles perform random walks until they
stick to an existing aggregate structure. The simulation results are analyzed to
understand how different sticking probabilities affect cluster morphology (width, height, density).

Key parameters:
- Grid size (height x width)
- Target cluster size
- Sticking probability (p_s)
- Seed position (starting aggregate point)
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

np.random.seed(0)

EMPTY_STATE = 0
WALKER_STATE = 1
AGGREGATE_STATE = 2


@njit
def init_grid(height, width, seed):
    """
    Initialize the simulation grid with a seed point for the aggregate.
    
    Parameters:
    -----------
    height, width : int
        Dimensions of the grid
    seed : tuple
        (x, y) coordinates of the initial aggregate point
        
    Returns:
    --------
    np.ndarray
        Initialized grid with the seed marked as AGGREGATE_STATE
    """
    x, y = seed
    grid = np.zeros((height, width))
    grid[x, y] = AGGREGATE_STATE
    return grid


@njit
def identify_neighbors(i, j, height, width):
    """
    Identify the four adjacent neighbors of a cell with periodic boundaries in the x-direction.
    
    Parameters:
    -----------
    i, j : int
        Row and column indices of the cell
    height, width : int
        Dimensions of the grid
        
    Returns:
    --------
    np.ndarray
        Array of (row, column) coordinates for the four adjacent neighbors
    """
    return np.array([
        (max(0, i-1), j),             # Top
        (min(height-1, i+1), j),        # Bottom
        (i, (j-1) % width),           # Left (Periodic)
        (i, (j+1) % width)            # Right (Periodic)
    ], dtype=np.int64)


@njit
def place_walker(grid):
    """
    Place a walker in a random empty cell in the second row of the grid.
    
    Parameters:
    -----------
    grid : np.ndarray
        Current state of the simulation grid
        
    Returns:
    --------
    np.ndarray
        Updated grid with a new walker placed
    """
    empty_cells = np.argwhere(grid[1, :] == EMPTY_STATE).flatten()
    if empty_cells.size > 0:
        y = np.random.choice(empty_cells)  # Choose a column index
        grid[1, y] = WALKER_STATE  # Place walker at row 1, chosen column
    return grid


@njit
def move_walker(grid, i, j, height, width):
    """
    Move a walker to a random empty neighboring cell.
    
    Parameters:
    -----------
    grid : np.ndarray
        Current state of the simulation grid
    i, j : int
        Row and column indices of the walker
    height, width : int
        Dimensions of the grid
        
    Returns:
    --------
    np.ndarray
        Updated grid with walker moved to a new position
    """
    if grid[i, j] == WALKER_STATE:
        neighbors = identify_neighbors(i, j, height, width)
        empty_neighbors = []
        for ni, nj in neighbors:
            if grid[ni, nj] == EMPTY_STATE:
                empty_neighbors.append((ni, nj))
        if len(empty_neighbors) > 0:
            idx = np.random.randint(len(empty_neighbors))
            target_x, target_y = empty_neighbors[idx]
            grid[target_x, target_y] = WALKER_STATE
            grid[i, j] = EMPTY_STATE
    return grid


@njit
def apply_vertical_boundaries(grid, i, j):
    """
    Handle walkers that reach the top or bottom boundary by removing them
    and placing a new walker.
    
    Parameters:
    -----------
    grid : np.ndarray
        Current state of the simulation grid
    i, j : int
        Row and column indices of the walker
        
    Returns:
    --------
    np.ndarray
        Updated grid with boundary conditions applied
    """
    height, _ = grid.shape
    if i == 0 or i == height - 1:
        grid[i, j] = EMPTY_STATE  # remove walker from top and bottom rows
        grid = place_walker(grid)  # add a new walker randomly
    return grid


@njit
def apply_aggregation(grid, i, j, height, width, p_s):
    """
    Check if a walker should stick to an adjacent aggregate based on sticking probability.
    
    Parameters:
    -----------
    grid : np.ndarray
        Current state of the simulation grid
    i, j : int
        Row and column indices of the walker
    height, width : int
        Dimensions of the grid
    p_s : float
        Sticking probability (0-1)
        
    Returns:
    --------
    np.ndarray
        Updated grid with aggregation applied if conditions are met
    """
    p = np.random.random()
    neighbors = identify_neighbors(i, j, height, width)
    # Shuffle without using random.shuffle
    for idx in np.random.permutation(len(neighbors)):
        ni, nj = neighbors[idx]
        if grid[ni, nj] == AGGREGATE_STATE and p <= p_s:
            grid[i, j] = AGGREGATE_STATE
            return grid
    return grid


@njit
def simulate(height, width, size, p_s, seed, max_steps, store_all=True):
    """
    Run the DLA simulation until the cluster reaches the target size or max steps.
    
    Parameters:
    -----------
    height, width : int
        Dimensions of the grid
    size : int
        Target cluster size (number of aggregate cells)
    p_s : float
        Sticking probability (0-1)
    seed : tuple
        (x, y) coordinates of the initial aggregate point
    max_steps : int
        Maximum number of simulation steps
    store_all : bool
        Whether to store all intermediate states
        
    Returns:
    --------
    np.ndarray
        3D array of simulation states if store_all=True,
        or final state wrapped in a 3D array if store_all=False
    """
    # Initialize grid and variables
    grid = init_grid(height, width, seed)
    cluster_size = np.sum(grid == AGGREGATE_STATE)
    step = 0

    if store_all:
        results = np.zeros((max_steps, height, width), dtype=np.int64)

    while cluster_size < size and step < max_steps:
        # Print progress every 1000 steps
        if step % 1000 == 0:
            print("Simulation progress: step", step,
                  "cluster size", cluster_size)
        new_grid = grid.copy()

        # Place walker in a random empty cell at row 1
        empty_cells = np.where(new_grid[1, :] == EMPTY_STATE)[0]
        if empty_cells.size > 0:
            y = empty_cells[np.random.randint(empty_cells.size)]
            new_grid[1, y] = WALKER_STATE

        walker_positions = np.argwhere(new_grid == WALKER_STATE)
        for pos in walker_positions:
            i, j = pos
            new_grid = apply_aggregation(new_grid, i, j, height, width, p_s)

        walker_positions = np.argwhere(new_grid == WALKER_STATE)
        for pos in walker_positions:
            i, j = pos
            new_grid = move_walker(new_grid, i, j, height, width)

        walker_positions = np.argwhere(new_grid == WALKER_STATE)
        for pos in walker_positions:
            i, j = pos
            new_grid = apply_vertical_boundaries(new_grid, i, j)

        if store_all:
            results[step] = new_grid

        grid = new_grid  # Update grid for next step
        cluster_size = np.sum(grid == AGGREGATE_STATE)
        step += 1

    print("Simulation finished at step", step, ". Cluster size:", cluster_size)

    if store_all:
        # Return only the filled portion of the results array
        return results[:step]
    else:
        # Wrap the final grid into a 3D array (one frame)
        final = np.empty((1, height, width), dtype=np.int64)
        final[0] = grid
        return final


@njit
def calculate_cluster_dimensions(results):
    """
    Calculate the height and width of the aggregate cluster.
    
    Parameters:
    -----------
    results : np.ndarray
        3D array containing simulation states
        
    Returns:
    --------
    tuple
        (cluster_height, cluster_width)
    """
    # results is assumed to be a 3D array with the final frame as the last element (or only element)
    last_frame = results[-1]
    aggregate_positions = np.argwhere(last_frame == AGGREGATE_STATE)
    cluster_height = (aggregate_positions[:, 0].max(
    ) - aggregate_positions[:, 0].min() + 1)
    cluster_width = (aggregate_positions[:, 1].max(
    ) - aggregate_positions[:, 1].min() + 1)
    return cluster_height, cluster_width


@njit
def calculate_density(results, size):
    """
    Calculate the density of the aggregate cluster.
    
    Parameters:
    -----------
    results : np.ndarray
        3D array containing simulation states
    size : int
        Number of aggregate cells
        
    Returns:
    --------
    float
        Density (ratio of aggregate cells to bounding box area)
    """
    # results is assumed to be a 3D array with the final frame as the last element (or only element)
    cluster_height, cluster_width = calculate_cluster_dimensions(results)
    density = size / (cluster_height * cluster_width)
    return density


def p_sweep(height, width, steps, p_s_vals, max_steps, seed, num_runs):
    """
    Run multiple simulations with different sticking probabilities and analyze results.
    
    Parameters:
    -----------
    height, width : int
        Dimensions of the grid
    steps : int
        Target cluster size (number of aggregate cells)
    p_s_vals : array-like
        Array of sticking probability values to test
    max_steps : int
        Maximum number of simulation steps
    seed : tuple
        (x, y) coordinates of the initial aggregate point
    num_runs : int
        Number of simulation runs for each p_s value
        
    Returns:
    --------
    tuple
        (visualization_results, avg_densities, avg_heights, avg_widths,
         std_densities, std_heights, std_widths, all_results)
    """
    # Store the first run of each p_s value for visualization
    visualization_results = []
    # Store all results for each p_s value for statistical analysis
    all_results = []
    avg_densities = []
    avg_heights = []
    avg_widths = []
    std_densities = []
    std_heights = []
    std_widths = []

    for p_s in p_s_vals:
        print("Starting simulations for p_s =", p_s)
        densities = []
        heights = []
        widths = []
        p_s_results = []  # All results for this p_s value
        for i in range(num_runs):
            print(f"  p_s = {p_s}: Run {i+1}/{num_runs}")
            # Run simulation without storing all frames to save memory
            result = simulate(height, width, steps, p_s,
                              seed, max_steps, store_all=False)
            p_s_results.append(result)

            if i == 0:
                visualization_results.append(result)

            cluster_height, cluster_width = calculate_cluster_dimensions(
                result)
            heights.append(cluster_height)
            widths.append(cluster_width)

            density = calculate_density(result, steps)
            densities.append(density)

        avg_density = np.mean(densities)
        avg_height = np.mean(heights)
        avg_width = np.mean(widths)

        std_density = np.std(densities, ddof=1) if len(densities) > 1 else 0
        std_height = np.std(heights, ddof=1) if len(heights) > 1 else 0
        std_width = np.std(widths, ddof=1) if len(widths) > 1 else 0

        avg_densities.append(avg_density)
        avg_heights.append(avg_height)
        avg_widths.append(avg_width)

        std_densities.append(std_density)
        std_heights.append(std_height)
        std_widths.append(std_width)

        all_results.append(p_s_results)
        print("Finished simulations for p_s =", p_s,
              "with average density =", avg_density, "±", std_density,
              "average height =", avg_height, "±", std_height,
              "average width =", avg_width, "±", std_width)

    return visualization_results, avg_densities, avg_heights, avg_widths, std_densities, std_heights, std_widths, all_results


def plot_metrics(p_s_vals, avg_densities, avg_heights, avg_widths,
                 std_densities, std_heights, std_widths, save=False):
    """
    Plot the metrics from the p_sweep analysis.
    
    Parameters:
    -----------
    p_s_vals : array-like
        Array of sticking probability values
    avg_densities, avg_heights, avg_widths : array-like
        Arrays of average metric values for each p_s
    std_densities, std_heights, std_widths : array-like
        Arrays of standard deviations for each p_s
    save : bool
        Whether to save the plot to a file
    """
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].errorbar(p_s_vals, avg_widths, yerr=std_widths, fmt='go-',
                     linewidth=2, markersize=8, capsize=5, elinewidth=1.5,
                     label='Mean ± Std Dev')
    axes[0].set_xlabel(r"$p_s$", fontsize=16)
    axes[0].set_ylabel("Cluster Width", fontsize=16)
    axes[0].set_title("Cluster Width vs. $p_s$", fontsize=18)
    axes[0].grid(True)
    axes[0].legend()

    axes[1].errorbar(p_s_vals, avg_heights, yerr=std_heights, fmt='ro-',
                     linewidth=2, markersize=8, capsize=5, elinewidth=1.5,
                     label='Mean ± Std Dev')
    axes[1].set_xlabel(r"$p_s$", fontsize=16)
    axes[1].set_ylabel("Cluster Height", fontsize=16)
    axes[1].set_title("Cluster Height vs. $p_s$", fontsize=18)
    axes[1].grid(True)
    axes[1].legend()

    axes[2].errorbar(p_s_vals, avg_densities, yerr=std_densities, fmt='bo-',
                     linewidth=2, markersize=8, capsize=5, elinewidth=1.5,
                     label='Mean ± Std Dev')
    axes[2].set_xlabel(r"$p_s$", fontsize=16)
    axes[2].set_ylabel("Density", fontsize=16)
    axes[2].set_title("Density vs. $p_s$", fontsize=18)
    axes[2].grid(True)
    axes[2].legend()

    # Add more space between subplots
    plt.tight_layout()

    if save:
        plt.savefig("fig/cluster_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_p_sweep(visualization_results, p_s_vals, save=False):
    """
    Plot the final state of simulations with different sticking probabilities.
    
    Parameters:
    -----------
    visualization_results : list
        List of simulation results for each p_s value
    p_s_vals : array-like
        Array of sticking probability values
    save : bool
        Whether to save the plot to a file
    """
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

    num_plots = len(p_s_vals)
    rows = (num_plots // 3) + (num_plots % 3 > 0)  # Auto-calculate rows
    fig, ax = plt.subplots(rows, 3, figsize=(15, 5 * rows), squeeze=False)
    for i in range(len(visualization_results)):
        r, c = i // 3, i % 3
        ax[r, c].imshow(visualization_results[i][-1], cmap="cubehelix")
        ax[r, c].set_title(
            fr"MC-Simulation with $p_s$ = {p_s_vals[i]:.1f}", fontsize=18)
        ax[r, c].axis("off")
    # Hide unused subplots
    for i in range(num_plots, rows * 3):
        r, c = i // 3, i % 3
        if r < len(ax) and c < len(ax[0]):
            ax[r, c].axis('off')
    plt.tight_layout()
    if save:
        plt.savefig("fig/p_sweep.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    height = 100
    width = 100
    size = 1000

    p_s_vals = np.linspace(0.1, 1, 6)
    max_steps = 50_000
    num_runs = 10  
    seed = (height - 1, (width // 2))

    visualization_results, avg_densities, avg_heights, avg_widths, std_densities, std_heights, std_widths, all_results = p_sweep(
        height, width, size, p_s_vals, max_steps, seed, num_runs
    )

    plot_metrics(p_s_vals, avg_densities, avg_heights, avg_widths,
                 std_densities, std_heights, std_widths, save=True)

    plot_p_sweep(visualization_results, p_s_vals, save=True)