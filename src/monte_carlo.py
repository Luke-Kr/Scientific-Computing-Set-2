"""
Diffusion Limited Aggregation (DLA) Simulation

This module implements a DLA simulation with varying sticking probability (p_s).
It includes functions to run the simulation, calculate cluster properties, and
visualize results with different sticking probabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numba import njit

np.random.seed(0)

EMPTY_STATE = 0
WALKER_STATE = 1
AGGREGATE_STATE = 2


@njit
def init_grid(height, width, seed):
    """Initialize the simulation grid with a seed point for aggregation."""
    x, y = seed
    grid = np.zeros((height, width))
    grid[x, y] = AGGREGATE_STATE
    return grid


@njit
def place_walker(grid):
    """Place a walker in an empty cell in the second row of the grid."""
    empty_cells = np.argwhere(grid[1, :] == EMPTY_STATE).flatten()
    if empty_cells.size > 0:
        y = np.random.choice(empty_cells)
        grid[1, y] = WALKER_STATE
    return grid


@njit
def identify_neighbors(i, j, height, width):
    """Return the coordinates of the four adjacent cells with periodic boundary in x-direction."""
    return np.array([
        (max(0, i-1), j),           # Top
        (min(height-1, i+1), j),    # Bottom
        (i, (j-1) % width),         # Left (Periodic)
        (i, (j+1) % width)          # Right (Periodic)
    ], dtype=np.int64)


@njit
def move_walker(grid, i, j, height, width):
    """Move a walker to a random empty neighboring cell."""
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
    """Remove walkers at vertical boundaries and place a new one."""
    height, _ = grid.shape
    if i == 0 or i == height - 1:
        grid[i, j] = EMPTY_STATE
        grid = place_walker(grid)
    return grid


@njit
def apply_aggregation(grid, i, j, height, width, p_s):
    """Check if a walker should stick to an adjacent aggregate based on sticking probability."""
    p = np.random.random()
    neighbors = identify_neighbors(i, j, height, width)
    for idx in np.random.permutation(len(neighbors)):
        ni, nj = neighbors[idx]
        if grid[ni, nj] == AGGREGATE_STATE and p <= p_s:
            grid[i, j] = AGGREGATE_STATE
            return grid
    return grid


@njit
def simulate(height, width, size, p_s, seed, max_steps, store_all=True):
    """
    Run the DLA simulation until cluster reaches target size or max steps.
    
    Parameters:
    -----------
    height, width : int
        Grid dimensions
    size : int
        Target cluster size
    p_s : float
        Sticking probability (0-1)
    seed : tuple
        Initial position (x,y) for the aggregate
    max_steps : int
        Maximum simulation steps
    store_all : bool
        Whether to store all intermediate states
        
    Returns:
    --------
    np.ndarray
        3D array of simulation states
    """
    grid = init_grid(height, width, seed)
    cluster_size = np.sum(grid == AGGREGATE_STATE)
    step = 0

    if store_all:
        results = np.zeros((max_steps, height, width), dtype=np.int64)
    
    while cluster_size < size and step < max_steps:
        if step % 1000 == 0:
            print("Simulation progress: step", step, "cluster size", cluster_size)
        new_grid = grid.copy()

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

        grid = new_grid
        cluster_size = np.sum(grid == AGGREGATE_STATE)
        step += 1

    print("Simulation finished at step", step, ". Cluster size:", cluster_size)
    
    if store_all:
        return results[:step]
    else:
        final = np.empty((1, height, width), dtype=np.int64)
        final[0] = grid
        return final


@njit
def calculate_cluster_dimensions(results):
    """Calculate the height and width of the aggregated cluster."""
    last_frame = results[-1]
    aggregate_positions = np.argwhere(last_frame == AGGREGATE_STATE)
    cluster_height = (aggregate_positions[:, 0].max() - aggregate_positions[:, 0].min() + 1)
    cluster_width = (aggregate_positions[:, 1].max() - aggregate_positions[:, 1].min() + 1)
    return cluster_height, cluster_width


@njit
def calculate_density(results, size):
    """Calculate the density of the aggregated cluster."""
    cluster_height, cluster_width = calculate_cluster_dimensions(results)
    density = size / (cluster_height * cluster_width)
    return density


def p_sweep(height, width, steps, p_s_vals, max_steps, seed, num_runs):
    """
    Run multiple simulations with different sticking probabilities.
    
    Parameters:
    -----------
    height, width : int
        Grid dimensions
    steps : int
        Target cluster size
    p_s_vals : array-like
        Array of sticking probability values to test
    max_steps : int
        Maximum simulation steps
    seed : tuple
        Initial position (x,y) for the aggregate
    num_runs : int
        Number of runs for each p_s value
        
    Returns:
    --------
    tuple
        (visualization_results, avg_densities, avg_heights, avg_widths,
         std_densities, std_heights, std_widths, all_results)
    """
    visualization_results = []
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
        p_s_results = []
        for i in range(num_runs):
            print(f"  p_s = {p_s}: Run {i+1}/{num_runs}")
            result = simulate(height, width, steps, p_s, seed, max_steps, store_all=False)
            p_s_results.append(result)

            if i == 0:
                visualization_results.append(result)

            cluster_height, cluster_width = calculate_cluster_dimensions(result)
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
    """Plot the metrics from the p_sweep analysis."""
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
    
    plt.tight_layout()
    
    if save:
        plt.savefig("fig/cluster_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_p_sweep(visualization_results, p_s_vals, save=False):
    """Plot the final state of simulations with different sticking probabilities."""
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })
    
    num_plots = len(p_s_vals)
    rows = (num_plots // 3) + (num_plots % 3 > 0)
    fig, ax = plt.subplots(rows, 3, figsize=(15, 5 * rows), squeeze=False)
    for i in range(len(visualization_results)):
        r, c = i // 3, i % 3
        ax[r, c].imshow(visualization_results[i][-1], cmap="cubehelix")
        ax[r, c].set_title(fr"MC-Simulation with $p_s$ = {p_s_vals[i]:.1f}", fontsize=18)
        ax[r, c].axis("off")
        
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