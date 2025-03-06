import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numba import njit

np.random.seed(0)

EMPTY_STATE = 0
WALKER_STATE = 1
AGGREGATE_STATE = 2


@njit
def init_grid(height, width):
    grid = np.zeros((height, width))
    grid[height - 1, (width//2)] = AGGREGATE_STATE
    return grid


@njit
def place_walker(grid):
    empty_cells = np.argwhere(grid[1, :] == EMPTY_STATE).flatten()

    if empty_cells.size > 0:
        y = np.random.choice(empty_cells)  # Choose a column index
        grid[1, y] = WALKER_STATE  # Place walker at row 1, chosen column
    return grid


@njit
def identify_neighbors(i, j, height, width):
    return np.array([
        (max(0, i-1), j),  # Top
        (min(height-1, i+1), j),  # Bottom
        (i, (j-1) % width),  # Left (Periodic)
        (i, (j+1) % width)  # Right (Periodic)
    ], dtype=np.int64)


@njit
def move_walker(grid, i, j, height, width):
    if grid[i, j] == WALKER_STATE:
        neighbors = identify_neighbors(i, j, height, width)
        empty_neighbors = []

        for ni, nj in neighbors:
            if grid[ni, nj] == EMPTY_STATE:
                empty_neighbors.append((ni, nj))

        if len(empty_neighbors) > 0:
            # Choose index instead of `random.choice`
            idx = np.random.randint(len(empty_neighbors))
            target_x, target_y = empty_neighbors[idx]
            grid[target_x, target_y] = WALKER_STATE
            grid[i, j] = EMPTY_STATE

    return grid


@njit
def apply_vertical_boundaries(grid, i, j):
    height, _ = grid.shape
    if i == 0 or i == height - 1:
        grid[i, j] = EMPTY_STATE  # remove walker from top and bottom rows
        grid = place_walker(grid)  # add a new walker randomly
    return grid


@njit
def apply_aggregation(grid, i, j, height, width, p_s):
    p = np.random.random()
    neighbors = identify_neighbors(i, j, height, width)

    # Shuffle without using `random.shuffle`
    for idx in np.random.permutation(len(neighbors)):
        ni, nj = neighbors[idx]
        if grid[ni, nj] == AGGREGATE_STATE and p <= p_s:
            grid[i, j] = AGGREGATE_STATE
            return grid

    return grid


@njit
def simulate(height, width, size, p_s, max_steps):
    # Pre-allocate with a reasonable max_steps value
    results = np.zeros((max_steps, height, width), dtype=np.int64)
    grid = np.zeros((height, width), dtype=np.int64)
    grid = init_grid(height, width)

    cluster_size = 0
    step = 0

    while cluster_size < size and step < max_steps:
        print(f"Step {step}")
        new_grid = grid.copy()

        # Place walker
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

        results[step] = new_grid
        grid = new_grid  # Update grid

        cluster_size = np.sum(grid == AGGREGATE_STATE)
        step += 1

    print(f"Simulation finished at step {step}. Cluster size: {cluster_size}")
    # Return only the filled portion of the results array
    return results[:step]


def save_simulation(results, filename):
    plt.imsave(filename, results, cmap="viridis")


def animate_simulation(results):
    fig, ax = plt.subplots()
    img = ax.imshow(results[0], cmap="viridis", interpolation="nearest")

    def animate(i):
        img.set_data(results[i])
        return [img]

    ani = animation.FuncAnimation(
        fig, animate, frames=len(results), interval=50, blit=True)
    plt.show()


def p_sweep(height, width, steps, p_s_vals, max_steps, num_runs):
    # Store the first run of each p_s value for visualization
    visualization_results = []
    # Store all results for each p_s value for statistical analysis
    all_results = []
    avg_densities = []

    for p_s in p_s_vals:
        densities = []
        p_s_results = []  # All results for this p_s value
        
        for i in range(num_runs):
            print(f"Simulating for p_s = {p_s}. Run {i+1}/{num_runs}...")
            result = simulate(height, width, steps, p_s, max_steps)  # Run simulation
            
            # Store the result for statistical analysis
            p_s_results.append(result)
            
            # Store the first run for visualization
            if i == 0:
                visualization_results.append(result)
                
            density = calculate_density(result, steps)
            densities.append(density)
        
        # Store all results for this p_s value
        all_results.append(p_s_results)
        
        avg_density = np.mean(densities)
        avg_densities.append(avg_density)

    return visualization_results, avg_densities, all_results


def plot_avg_density(p_s_vals, avg_densities, save=False):
    plt.plot(p_s_vals, avg_densities, "bo")
    plt.xlabel(r"$p_s$")
    plt.ylabel("Average Density")
    plt.title("Average Density vs. $p_s$")
    plt.grid(True)
    if save:
        plt.savefig("fig/avg_density.png")
    plt.show()


def plot_p_sweep(visualization_results, p_s_vals, save=False):
    num_plots = len(p_s_vals)
    rows = (num_plots // 3) + (num_plots % 3 > 0)  # Auto-calculate rows
    fig, ax = plt.subplots(rows, 3, figsize=(15, 5 * rows), squeeze=False)  # Ensure it's always 2D

    for i in range(len(visualization_results)):
        r, c = i // 3, i % 3
        ax[r, c].imshow(visualization_results[i][-1], cmap="cubehelix")
        ax[r, c].set_title(fr"$p_s$ = {p_s_vals[i]:.1f}")
        ax[r, c].axis("off")

    # Hide unused subplots
    for i in range(num_plots, rows * 3):
        r, c = i // 3, i % 3
        if r < len(ax) and c < len(ax[0]):
            ax[r, c].axis('off')
            
    plt.tight_layout()
    if save:
        plt.savefig("fig/p_sweep.png")
    plt.show()

def plot_last_frame(results):
    plt.imshow(results[-1], cmap="magma")
    plt.show()


@njit
def calculate_density(results, size):
    last_frame = results[-1]

    # Find aggregate positions
    aggregate_positions = np.argwhere(last_frame == AGGREGATE_STATE)

    # Cluster height: max row - min row
    cluster_height = (aggregate_positions[:, 0].max() -
                      aggregate_positions[:, 0].min() + 1)

    # Cluster width: max column - min column
    cluster_width = (aggregate_positions[:, 1].max() -
                     aggregate_positions[:, 1].min() + 1)

    density = size / (cluster_height * cluster_width)
    return density


if __name__ == "__main__":
    height = 101
    width = 101
    size = 500
    p_s_vals = np.linspace(0.1, 0.9, 5)
    max_steps = 16_000
    num_runs = 2
    
    #results = simulate(height, width, size, p_s_vals[1], max_steps)
    #plot_last_frame(results)
    # shortened_results = results[7500:-1]
    # animate_simulation(shortened_results)

    visualization_results, avg_densities, all_results = p_sweep(height, width, size, p_s_vals, max_steps, num_runs)
    plot_p_sweep(visualization_results, p_s_vals, save=True)
    plot_avg_density(p_s_vals, avg_densities, save=True)
