import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
from matplotlib.animation import FuncAnimation




def init_mask(N):
    mask = np.zeros((N, N), dtype=bool)
    # print(mask.shape)
    mask[-1, N//2] = True

    return mask

@jit(nopython=True)
def sor_simulation(omega: float, grid: np.ndarray, max_iter: int, N: int, tol: float, mask: np.ndarray):

    history = [grid.copy()]
    for t in range(1, max_iter + 1):
        for i in range(1, N-1):
            for j in range(N):
                if mask[i, j] == 1.0:
                    continue
                old = grid[i, j]
                left = grid[i, (j - 1) % (N)]
                right = grid[i, (j + 1) % (N)]
                up = grid[i + 1, j]
                down = grid[i - 1, j]

                grid[i, j] = (1 - omega) * old + (omega / 4) * \
                    (up + down + left + right)
                if grid[i, j] < 0:
                    grid[i, j] = 0

        # Check for convergence
        if np.allclose(grid, history[-1], atol=tol):
            # print(f"Converged at t = {t}")
            break
        history.append(grid.copy())

    return grid, t

@jit(nopython=True)
def get_candidates(eta, grid, N, mask):
    
    indices = []
    weights = []

    for i in range(N):
        for j in range(N):
            if mask[i, j]:
                continue  # Skip occupied cells

            left = mask[i, (j - 1) % (N)]
            right = mask[i, (j + 1) % (N)]
            up = mask[i + 1, j] if i < N - 1 else False
            down = mask[i - 1, j] if i > 0 else False

            if left or right or up or down:
                # border_mask[i, j] = True
                indices.append((i, j))
                weights.append(grid[i, j] ** eta)

    return weights, indices

def dla_simulation(omega: float, eta: float ,grid: np.ndarray, max_iter: int, N: int, tol: float, mask: np.ndarray):
    """
    Runs the Successive Over-Relaxation (SOR) iterative method for solving Laplace's equation.

    Parameters:
    omega (float): Relaxation factor for SOR.
    grid (np.ndarray): 2D NumPy array representing the simulation grid (N+1, N+1).
    max_iter (int): Maximum number of iterations before stopping.
    N (int): Grid size (N x N domain with additional boundary layer).
    tol (float): Convergence tolerance.

    Returns:
    tuple: (history, t) where history is a list of grid states and t is the iteration count.
    """
    mask_history = [mask.copy()]
    grid_history = [grid.copy()]
    convergence_times = []
    for _ in range(1, max_iter + 1):
        grid, t = sor_simulation(omega, grid, max_iter, N, tol, mask)
        convergence_times.append(t)
        weights, indices = get_candidates(eta, grid, N, mask)
        if len(indices) > 0:
            # Normalize to sum to 1 (assuming all weights are non-negative)
            sum = np.sum(weights)
            if sum == 0:
                print("No food")
                return mask_history, grid_history
                # probabilities = np.ones_like(weights) / len(weights)
            # else:
            probabilities = weights / sum
            # print(weights, probabilities)

            # Choose an index with weighted probability
            chosen_flat_idx = np.random.choice(len(weights), p=probabilities)

            # Map back to the 2D index
            chosen_idx = indices[chosen_flat_idx]

            assert not mask[chosen_idx], "Chosen index is already occupied."
            mask[chosen_idx] = True
            grid[chosen_idx] = 0
        else:
            print("No valid indices to choose from.")
        mask_history.append(mask.copy())
        grid_history.append(grid.copy())
    print(f"Avg convergence time: {np.mean(convergence_times)}, std: {np.std(convergence_times)}")
    return mask_history, grid_history, convergence_times


if __name__ == '__main__':
    # Parameters
    N = 100            # Grid size (N x N)
    max_iter = 1000
    tol = 1e-5
    omega = 1.8  # Relaxation factor
    eta = 0.2  # Growth probability
    print(f"max_iter: {max_iter}")

    mask = init_mask(N)
    print("Mask shape:", mask.shape)
    print(mask)

    grid = np.zeros((N, N))
    grid[0, :] = 1.0
    print("Grid shape:", grid.shape)
    print(grid)

    mask_history, grid_history, _ = dla_simulation(omega, eta, grid, max_iter, N, tol, mask)
    
    mask_history = np.array(mask_history, dtype=bool)
    print(f"Final mask: {mask_history[-1]}")
    np.savetxt("mask.csv", mask_history[-1], delimiter=" ", fmt="%d")

    # Find the first array that has a true value
    final_height = np.argmax(np.any(mask_history[-1], axis=1))
    print(f"Final height: {N - final_height}")
    print(mask_history[-1][final_height])
    

    # animate the simulation
    fig, ax = plt.subplots()

    image = grid_history[0]
    image[mask_history[0]] = 10
    img = ax.imshow(image, cmap='viridis', alpha=1, vmin=0, vmax=1)

    def update(frame):
        image = grid_history[frame]
        image[mask_history[frame]] = 10
        img.set_data(image)
        ax.set_title(f"Frame {frame}")
        return img
    
    ani = FuncAnimation(fig, update, frames=len(mask_history), interval=10)
    plt.colorbar(img)


    plt.show()

