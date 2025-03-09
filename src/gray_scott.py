"""
Gray-Scott Model Simulation

This script simulates the Gray-Scott reaction-diffusion model using different boundary conditions.
The model is solved on a 2D grid with periodic, Dirichlet, Neumann, or Robin boundary conditions.
The results are visualized and saved as an image.
"""

import numpy as np
import matplotlib.pyplot as plt

def initialize_grid(N):
    """
    Initialize the grid with initial conditions for u and v.

    Parameters:
    N (int): Size of the grid (NxN).

    Returns:
    tuple: Initialized grids u and v.
    """
    u = np.ones((N, N), dtype=np.float64) * 0.5
    v = np.zeros((N, N), dtype=np.float64)

    # Center square with v = 0.25
    center = N // 2
    size = N // 10
    v[center - size:center + size, center - size:center + size] = 0.25

    # Add small random noise
    noise_u = np.random.uniform(-0.01, 0.01, (N, N))
    noise_v = np.random.uniform(-0.01, 0.01, (N, N))
    u += noise_u
    v += noise_v

    return u, v

def apply_boundary_conditions(grid, bc_type, alpha=0.5, beta=1.0, gamma=0.0):
    """
    Apply boundary conditions to the grid.

    Parameters:
    grid (ndarray): The grid to apply boundary conditions to.
    bc_type (str): Type of boundary condition ('periodic', 'dirichlet', 'neumann', 'robin').
    alpha (float): Alpha parameter for Robin boundary condition.
    beta (float): Beta parameter for Robin boundary condition.
    gamma (float): Gamma parameter for Robin boundary condition.
    """
    if bc_type == "periodic":  # Periodic
        grid[0, :] = grid[-2, :]
        grid[-1, :] = grid[1, :]
        grid[:, 0] = grid[:, -2]
        grid[:, -1] = grid[:, 1]
    elif bc_type == "dirichlet":  # Dirichlet (zero at the boundaries)
        grid[0, :] = 0
        grid[-1, :] = 0
        grid[:, 0] = 0
        grid[:, -1] = 0
    elif bc_type == "neumann":  # Neumann (copy neighboring values)
        grid[0, :] = grid[1, :]
        grid[-1, :] = grid[-2, :]
        grid[:, 0] = grid[:, 1]
        grid[:, -1] = grid[:, -2]
    elif bc_type == "robin":  # Robin (mixed) boundary conditions
        grid[0, :] = (gamma - beta * grid[1, :]) / alpha
        grid[-1, :] = (gamma - beta * grid[-2, :]) / alpha
        grid[:, 0] = (gamma - beta * grid[:, 1]) / alpha
        grid[:, -1] = (gamma - beta * grid[:, -2]) / alpha

def compute_laplacian(grid):
    """
    Compute the Laplacian of the grid using finite differences.

    Parameters:
    grid (ndarray): The grid to compute the Laplacian of.

    Returns:
    ndarray: The Laplacian of the grid.
    """
    return (
        np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
        np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) - 4 * grid
    )

def gray_scott_simulation(N, Du, Dv, f, k, dt, steps, bc_type):
    """
    Run the Gray-Scott simulation.

    Parameters:
    N (int): Size of the grid (NxN).
    Du (float): Diffusion rate of u.
    Dv (float): Diffusion rate of v.
    f (float): Feed rate.
    k (float): Kill rate.
    dt (float): Time step.
    steps (int): Number of simulation steps.
    bc_type (str): Type of boundary condition ('periodic', 'dirichlet', 'neumann', 'robin').

    Returns:
    tuple: Final grids u and v after simulation.
    """
    u, v = initialize_grid(N)

    for _ in range(steps):
        lap_u = compute_laplacian(u)
        lap_v = compute_laplacian(v)

        uvv = u * v * v
        u += (Du * lap_u - uvv + f * (1 - u)) * dt
        v += (Dv * lap_v + uvv - (f + k) * v) * dt

        # Apply boundary conditions directly inside the loop
        apply_boundary_conditions(u, bc_type)
        apply_boundary_conditions(v, bc_type)

    return u, v

if __name__ == '__main__':
    # Simulation parameters
    N = 200
    Du, Dv = 0.16, 0.08
    f, k = 0.035, 0.060
    dt, steps = 1.0, 10000
    bc_type = "robin" # Change boundary condition type here

    u, v = gray_scott_simulation(N, Du, Dv, f, k, dt, steps, bc_type)

    # Set global font size
    plt.rcParams.update({'font.size': 20})

    # Create figure with tight layout to remove whitespace
    fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
    im = ax.imshow(v, cmap='inferno')

    ax.set_title(f'Gray-Scott Model ({bc_type} BC)')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046)

    # Save figure without extra whitespace
    plt.savefig('fig/gray_scott.png', bbox_inches='tight')
