## Scientific-Computing-Set-2

2.2 Monte Carlo Diffusion Limited Aggregation (DLA) Simulation

A Python implementation of Monte Carlo Diffusion Limited Aggregation with variable sticking probabilities. This simulation models the growth of fractal-like structures that form when particles undergoing random motion attach to a growing aggregate.

The simulation shows how different sticking probabilities affect the resulting cluster morphology, including:
- Cluster width
- Cluster height
- Cluster density

Upon running the code, some example clusters given p_r will be saved under fig/p_sweep. The resulting statistics will be saved under fig/cluster_metrics.png

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Numba

## Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/dla-simulation.git
cd dla-simulation
pip install numpy matplotlib numba

## Notebook

Run notebook.ipynb in order to have all relevant, output generating functions inone file.
