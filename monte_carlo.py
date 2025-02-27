import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random
from tqdm import tqdm
from numba import jit


EMPTY_STATE = 0
WALKER_STATE = 1
AGGREGATE_STATE = 2


def init_grid(height, width):
    grid = np.zeros((height, width))
    grid[height - 1, (width//2)] = AGGREGATE_STATE
    return grid


def place_walker(grid):
    empty_cells = np.argwhere(grid == EMPTY_STATE)
    if empty_cells.size > 0:
        x, y = empty_cells[np.random.choice(len(empty_cells))]
        grid[x, y] = WALKER_STATE
    return grid


def identify_neighbors(grid, i, j):
    height, width = grid.shape
    neighbors = [
        (max(0, i-1), j),  # Top neighbor (clamped to grid)
        (min(height-1, i+1), j),  # Bottom neighbor (clamped to grid)
        (i, (j-1) % width),  # Left neighbor (periodic)
        (i, (j+1) % width)  # Right neighbor (periodic)
    ]

    return neighbors


def move_walker(grid, i, j):
    if grid[i, j] == WALKER_STATE:
        neighbors = identify_neighbors(grid, i, j)
        empty_neighbors = [(ni, nj)
                           for ni, nj in neighbors if grid[ni, nj] == EMPTY_STATE]

        if empty_neighbors:
            target_x, target_y = random.choice(empty_neighbors)
            grid[target_x, target_y] = WALKER_STATE
            grid[i, j] = EMPTY_STATE

    return grid


def apply_vertical_boundaries(grid, i, j):
    height, _ = grid.shape
    if i == 0 or i == height - 1:
        grid[i, j] = EMPTY_STATE  # remove walker from top and bottom rows
        grid = place_walker(grid)  # add a new walker randomly
    return grid


def apply_aggregation(grid, i, j, p_s):
    p = random.random()  # Generate a random number between 0 and 1
    neighbors = identify_neighbors(grid, i, j)
    random.shuffle(neighbors)  # Randomize neighbor check order

    for ni, nj in neighbors:
        if grid[ni, nj] == AGGREGATE_STATE and p <= p_s:
            grid[i, j] = AGGREGATE_STATE  # Walker aggregates
            return grid  # Stop further processing, as aggregation happened

    return grid  # If no aggregation, the walker keeps moving



def simulate(height, width, steps, p_s, save_last=False):
    results = np.zeros((steps, height, width))
    grid = init_grid(height, width)
    prev_grid = np.copy(grid)

    for k in tqdm(range(steps), desc="Simulating", total=steps):
        new_grid = np.copy(prev_grid)

        # Ensure only one walker is added per step
        new_grid = place_walker(new_grid)

        walker_positions = np.argwhere(new_grid == WALKER_STATE)

        for i, j in walker_positions:
            new_grid = apply_vertical_boundaries(new_grid, i, j)
            new_grid = apply_aggregation(new_grid, i, j, p_s)

        # Re-check walkers before moving
        walker_positions = np.argwhere(new_grid == WALKER_STATE)

        for i, j in walker_positions:
            new_grid = move_walker(new_grid, i, j)

        results[k] = new_grid
        prev_grid = np.copy(new_grid)

    if save_last:
        save_simulation(results[-1], "monte_carlo_result.png")
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


def plot_p_sweep(results_arr, p_s_vals):
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    for i, result in enumerate(results_arr):
        ax[i // 3, i % 3].imshow(result[-1], cmap="viridis")
        ax[i // 3, i % 3].set_title(fr"$p_s$ = {p_s_vals[i]:.1f}")
        ax[i // 3, i % 3].axis("off")  # Hide axes

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    height = 101
    width = 101
    steps = 2000
    	
    p_s = 1
    results = simulate(height, width, steps, p_s)
    animate_simulation(results)