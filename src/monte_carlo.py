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
    x, y = seed
    grid = np.zeros((height, width))
    grid[x, y] = AGGREGATE_STATE
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
        (max(0, i-1), j),             # Top
        (min(height-1, i+1), j),        # Bottom
        (i, (j-1) % width),           # Left (Periodic)
        (i, (j+1) % width)            # Right (Periodic)
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
    # Shuffle without using random.shuffle
    for idx in np.random.permutation(len(neighbors)):
        ni, nj = neighbors[idx]
        if grid[ni, nj] == AGGREGATE_STATE and p <= p_s:
            grid[i, j] = AGGREGATE_STATE
            return grid
    return grid


@njit
def simulate(height, width, size, p_s, seed, max_steps, store_all=True):
    # Initialize grid and variables
    grid = init_grid(height, width, seed)
    cluster_size = np.sum(grid == AGGREGATE_STATE)
    step = 0

    if store_all:
        results = np.zeros((max_steps, height, width), dtype=np.int64)
    
    while cluster_size < size and step < max_steps:
        # Print progress every 1000 steps
        if step % 1000 == 0:
            print("Simulation progress: step", step, "cluster size", cluster_size)
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
        print("Starting simulations for p_s =", p_s)
        densities = []
        p_s_results = []  # All results for this p_s value
        for i in range(num_runs):
            print(f"  p_s = {p_s}: Run {i+1}/{num_runs}")
            # Run simulation without storing all frames to save memory
            result = simulate(height, width, steps, p_s, seed, max_steps, store_all=False)
            p_s_results.append(result)

            if i == 0:
                visualization_results.append(result)

            density = calculate_density(result, steps)
            densities.append(density)
        avg_density = np.mean(densities)
        avg_densities.append(avg_density)
        all_results.append(p_s_results)
        print("Finished simulations for p_s =", p_s, "with average density =", avg_density)
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
    fig, ax = plt.subplots(rows, 3, figsize=(15, 5 * rows), squeeze=False)
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
    # results is assumed to be a 3D array with the final frame as the last element (or only element)
    last_frame = results[-1]
    aggregate_positions = np.argwhere(last_frame == AGGREGATE_STATE)
    cluster_height = (aggregate_positions[:, 0].max() - aggregate_positions[:, 0].min() + 1)
    cluster_width = (aggregate_positions[:, 1].max() - aggregate_positions[:, 1].min() + 1)
    density = size / (cluster_height * cluster_width)
    return density


if __name__ == "__main__":
    height = 101
    width = 101
    size = 500
    p_s_vals = np.linspace(0.1, 0.9, 5)
    max_steps = 20_000
    num_runs = 10
    seed = (height - 1, (width // 2))

    # For p-sweep, run without storing all frames to save memory
    visualization_results, avg_densities, all_results = p_sweep(height, width, size, p_s_vals, max_steps, num_runs)
    plot_p_sweep(visualization_results, p_s_vals, save=True)
    plot_avg_density(p_s_vals, avg_densities, save=True)
