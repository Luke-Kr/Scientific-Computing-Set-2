import numpy as np
import matplotlib.pyplot as plt
from numba import jit


def init_grid(height, width):
    grid = np.zeros(height, width)
    grid[-1, width//2] = 2
    return grid


def place_walker(grid):
    placed = False
    height, width = grid.shape

    while not placed:
        x = np.random.randint(0, height)
        y = np.random.randint(0, width)
        if grid[x, y] == 0:
            grid[x, y] = 1
            placed = True

    return grid


def identify_neighbors(grid, i, j):
    if grid[i, j] == 1:
        neighbors = [
            (i-1, j),  # up
            (i+1, j),  # down
            (i, j-1 % width),  # left
            (i, j+1 % width)  # right
        ]

        return neighbors


def move_walker(grid, i, j):

    neighbors = identify_neighbors(grid, i, j)
    empty_neighbors = [(ni, nj)
                       for ni, nj in neighbors if grid[ni, nj] == 0]

    if empty_neighbors:
        target_x, target_y = np.random.choice(empty_neighbors)
        grid[target_x, target_y] = 1
        grid[i, j] = 0

        return grid


def apply_vertical_boundaries(grid, i, j):
    pass


def apply_aggregation(grid, i, j):
    neighbors = identify_neighbors(grid, i, j)
    for n in neighbors:
        if n == 2:
            grid[i, j] = 2

    return grid


def simulate(height, width, steps):
    grid = init_grid(height, width)
    for _ in range(steps):
        new_grid = np.copy(grid)
        # A single new walker gets placed per time step
        new_grid = place_walker(grid)

        for i in range(height):
            for j in range(width):
                new_grid = apply_aggregation(new_grid, i, j)
                new_grid = move_walker(new_grid, i, j)

    return grid


if __name__ == "__main__":
    height = 100
    width = 100
    steps = 1000

    plt.imshow(grid)
    plt.show()