# eprop-motor-control

This directory contains code and resources for motor control experiments using e-prop (eligibility propagation) learning algorithms.

## Structure
- `eprop_network_dale-law-not-applied.py`: E-prop network implementation without Dale's law.
- `eprop-reaching-task.py`: Main script for running the reaching task experiment.
- `plot_results.py`: Script for visualizing experiment results.
- `config/`: Contains configuration files (e.g., `config.yaml`).
- `sim_results/`: Stores simulation results, including plots and data for different experiment runs.
- `trajectory*.txt`: Trajectory data files used in experiments.

## Usage

You can run the main experiment script with various options for parameter sweeps and custom configurations:

### Basic usage
Run the default experiment:
```bash
python eprop-reaching-task.py
```

### Set both excitatory and inhibitory learning rates simultaneously
```bash
python eprop-reaching-task.py --learning-rate 0.001
```

### Scan a single parameter (e.g., learning rate)
```bash
python eprop-reaching-task.py --scan-param learning_rate_exc --scan-values 0.001,0.01,0.1
```

### Scan multiple parameters (grid search)
```bash
python eprop-reaching-task.py --scan-param learning_rate_exc,neurons.n_rec --scan-values "0.001,0.01;100,200"
```

### Scan learning rates, number of RBF centers, and number of neurons together (multi-parameter grid search)
```bash
python eprop-reaching-task.py --scan-param learning_rate_exc,learning_rate_inh,rbf.num_centers,neurons.n_rec \
    --scan-values "0.001,0.01;0.001,0.01;10,20;100,200"
```
_Note: Use quotes around `--scan-values` to avoid shell parsing issues with semicolons._

### Disable plotting
```bash
python eprop-reaching-task.py --no-plot
```

- Separate parameter names with commas, and value lists with semicolons.
- All parameter names must match the config structure (e.g., `neurons.n_rec`, `rbf.num_centers`).
- Always quote `--scan-values` if using semicolons to avoid shell parsing errors.

## Environment Setup

You can create the recommended environment with all required dependencies using the provided `environment.yml` file and [mamba](https://mamba.readthedocs.io/en/latest/), a faster drop-in replacement for conda:

```bash
mamba env create -f environment.yml
mamba activate motor-controller
```

This will install all necessary conda and pip packages, including scientific libraries, NEST simulator, and additional Python tools used in this project.

## Requirements

- All dependencies are specified in `environment.yml` for easy environment setup with mamba (or conda).
- If you prefer pip, see the `requirements.txt` (if available), but mamba/conda is recommended for full compatibility (especially for NEST).

## Results
Simulation results and plots are saved in the `sim_results/` directory, organized by experiment configuration.

## Note on Dale's Law Version

The script `eprop_network_dale-law-not-applied.py` contains an older version of the network implementation without Dale's law. It is included here for reference only and may be removed or relocated in the future.

## License
<Specify your license here>.
