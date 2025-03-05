import numpy as np
import matplotlib.pyplot as plt

def initialize_grid(N):
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

def apply_boundary_conditions(grid, bc_type):
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

def compute_laplacian(grid):
    return (
        np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
        np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) - 4 * grid
    )

def gray_scott_simulation(N, Du, Dv, f, k, dt, steps, bc_type):
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
    bc_type = "periodic" # Change boundary condition type here

    u, v = gray_scott_simulation(N, Du, Dv, f, k, dt, steps, bc_type)

    plt.imshow(v, cmap='inferno')
    plt.title(f'Gray-Scott Model ({bc_type} BC)')
    plt.colorbar()
    plt.savefig('fig/gray_scott.png')
