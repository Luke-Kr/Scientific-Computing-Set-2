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
def simulate(height, width, steps, p_s):
    results = np.zeros((steps, height, width), dtype=np.int64)
    grid = np.zeros((height, width), dtype=np.int64)
    grid = init_grid(height, width)

    for k in range(steps):
        print(f"Step {k+1}/{steps}")
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

        results[k] = new_grid
        grid = new_grid  # Update grid

    return results


def save_simulation(results, filename):
    plt.imsave(filename, results, cmap="viridis")


def animate_simulation(results):
    fig, ax = plt.subplots()
    img = ax.imshow(results[0], cmap="viridis", interpolation="nearest")

    def animate(i):
        img.set_data(results[i])
        return [img]

    ani = animation.FuncAnimation(
        fig, animate, frames=len(results), interval=100, blit=True)
    plt.show()


def p_sweep(height, width, steps, p_s_vals):
    results_arr = []

    for p_s in p_s_vals:
        print(f"Simulating for p_s = {p_s}")
        result = simulate(height, width, steps, p_s)  # Run simulation
        results_arr.append(result)
        print("Done")

    return results_arr


def plot_p_sweep(results_arr, p_s_vals, save=False):
    num_plots = len(p_s_vals)
    rows = (num_plots // 3) + (num_plots % 3 > 0)  # Auto-calculate rows
    fig, ax = plt.subplots(rows, 3, figsize=(
        15, 5 * rows), squeeze=False)  # Ensure it's always 2D

    for i, result in enumerate(results_arr):
        r, c = i // 3, i % 3
        ax[r, c].imshow(result[-1], cmap="cubehelix")
        ax[r, c].set_title(fr"$p_s$ = {p_s_vals[i]:.1f}")
        ax[r, c].axis("off")

    # Hide unused subplots
    for i in range(num_plots, rows * 3):
        fig.delaxes(ax[i // 3, i % 3])

    plt.tight_layout()
    if save:
        plt.savefig("fig/p_sweep.png")
    plt.show()


def plot_last_frame(results):
    plt.imshow(results[-1], cmap="viridis_r")
    plt.show()


if __name__ == "__main__":
    height = 101
    width = 101
    steps = 10_000

    p_s_vals = np.linspace(0.1, 0.5, 3)
    results = simulate(height, width, steps, p_s_vals[1])
    shortened_results = results[9000:10000]
    animate_simulation(shortened_results)