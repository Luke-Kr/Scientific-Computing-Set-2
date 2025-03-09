import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from matplotlib.animation import FuncAnimation

def init_mask(N) -> np.ndarray:
    """
    Initialize the mask for the occupied cells.
    
    Args:
    N (int): Grid size

    Returns:
    np.ndarray: Mask for the occupied cells
    """
    mask = np.zeros((N, N), dtype=bool)
    mask[-1, N//2] = True

    return mask

@jit(nopython=True)
def sor_simulation(omega: float, grid: np.ndarray, max_iter: int, N: int, tol: float, mask: np.ndarray) -> tuple:
    """
    Perform a SOR simulation on the grid.

    Args:
    omega (float): Relaxation factor
    grid (np.ndarray): Grid
    max_iter (int): Maximum number of iterations
    N (int): Grid size
    tol (float): Tolerance for convergence
    mask (np.ndarray): Mask for occupied cells

    Returns:
    tuple: Tuple containing the final grid and the number of iterations to converge
    """

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
            break
        history.append(grid.copy())

    return grid, t

@jit(nopython=True)
def get_candidates(eta, grid, N, mask):
    """
    Get the indices of the candidate cells for the next particle and their weights.
    
    Args:
    eta (float): Growth probability
    grid (np.ndarray): Grid
    N (int): Grid size
    mask (np.ndarray): Mask for occupied cells

    Returns:
    tuple: Tuple containing the weights and indices of the candidate cells
    """
    
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

def dla_simulation(omega: float, eta: float ,grid: np.ndarray, max_size: int, N: int, tol: float, mask: np.ndarray) -> tuple:
    """
    Simulate a DLA cluster growth using the SOR method.

    Args:
    omega (float): Relaxation factor
    eta (float): Growth probability
    grid (np.ndarray): Initial grid
    max_size (int): Maximum number of particles
    N (int): Grid size
    tol (float): Tolerance for convergence
    mask (np.ndarray): Mask for occupied cells

    Returns:
    tuple: Tuple containing the mask history, grid history and convergence times
    """	
    mask_history = [mask.copy()]
    grid_history = [grid.copy()]
    convergence_times = []
    for _ in range(1, max_size + 1):
        grid, t = sor_simulation(omega, grid, 1000, N, tol, mask)
        convergence_times.append(t)
        weights, indices = get_candidates(eta, grid, N, mask)
        if len(indices) > 0:
            sum = np.sum(weights)
            if sum == 0:
                print("No food")
                return mask_history, grid_history
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
    return mask_history, grid_history, convergence_times


def find_optimal_omega() -> np.ndarray:
    """ 
    Calculate the average convergence time for different omega values

    Returns:
    np.ndarray: Array of shape (10, 2) where each row contains the average convergence time and standard deviation for a given omega value
    """
    results = []
    for omega in np.linspace(1.0, 1.9, 10):
        runs = []
        for r in range(10):
            grid = np.zeros((100, 100))
            grid[0, :] = 1.0
            mask = init_mask(100)
            _, _, t = dla_simulation(omega, 1, grid, 1000, 100, 1e-5, mask)
            runs.append(np.mean(t))
        print(f"Omega: {omega}, Average t: {np.mean(runs)}, std: {np.std(runs)}")
        results.append([np.mean(runs), np.std(runs)])
    return np.array(results)


if __name__ == '__main__':
    # Parameters
    N = 100            # Grid size (N x N)
    max_size = 1000
    tol = 1e-5
    omega = 1.8  # Relaxation factor
    eta = 0.2  # Growth probability


    mask = init_mask(N)
    grid = np.zeros((N, N))
    grid[0, :] = 1.0
    mask_history, grid_history, _ = dla_simulation(omega, eta, grid, max_size, N, tol, mask)
    mask_history = np.array(mask_history, dtype=bool)

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

