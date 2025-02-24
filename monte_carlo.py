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


def move_walkers(grid):
    height, width = grid.shape
    new_grid = grid.copy()
    for i in range(height):
        for j in range(width):
            if grid[i, j] == 1:
                neighbors = [
                    (i-1, j),  # up
                    (i+1, j),  # down
                    (i, j-1),  # left
                    (i, j+1)  # right
                ]

                empty_neighbors = [(ni, nj)
                                   for ni, nj in neighbors if grid[ni, nj] == 0]
                if empty_neighbors:
                    target_x, target_y = np.random.choice(empty_neighbors)
                    new_grid[target_x, target_y] = 1
                    new_grid[i, j] = 0

    return new_grid


def simulate(height, width, steps):
    grid = init_grid(height, width)
    for _ in range(steps):
        new_grid = np.copy(grid)
        new_grid = place_walker(grid)
        new_grid = move_walkers(new_grid)
        # ...
    return grid


if __name__ == "__main__":
    height = 100
    width = 100
    steps = 1000
    grid = init_grid(height, width)

    plt.imshow(grid)
    plt.show()
