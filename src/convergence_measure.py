import numpy as np
import matplotlib.pyplot as plt

from DLA import init_mask, dla_simulation

def find_optimal_omega():
    results = []
    for omega in np.linspace(1.0, 1.9, 10):
        runs = []
        for r in range(10):
            grid = np.zeros((100, 100))
            grid[0, :] = 1.0
            mask = init_mask(100)
            _, _, t = dla_simulation(omega, 1, grid, 1000, 100, 1e-5, mask)
            np.save(f"data/omega_{omega}_run_{r}.npy", grid)
            runs.append(t)
        print(f"Omega: {omega}, Average t: {np.mean(runs)}, std: {np.std(runs)}")
        results.append([np.mean(runs), np.std(runs)])
    return np.array(results)

if "__main__" == __name__:
    results = find_optimal_omega()

    plt.plot(np.linspace(1.0, 1.9, 10), results[:, 0], label="Average convergence time")
    plt.fill_between(np.linspace(1.0, 1.9, 10), results[:, 0] - results[:, 1], results[:, 0] + results[:, 1], alpha=0.2, label="Standard deviation")
    plt.xlabel("Omega")
    plt.ylabel("Time")
    plt.legend()
    plt.show()

