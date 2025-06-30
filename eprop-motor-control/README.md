# eprop-motor-control

This directory contains code, configuration, and data for simulating and analyzing motor control experiments using e-prop (eligibility propagation) learning algorithms in spiking neural networks. The project enables training and evaluation of recurrent spiking networks to perform reaching tasks, with flexible configuration, parameter sweeps, and automated result visualization. It supports features such as grid search, custom learning rates, and output analysis.

## Structure
- `eprop_network_dale-law-not-applied.py`: E-prop network implementation without Dale's law (legacy, for reference).
- `eprop-reaching-task.py`: Main script for running the reaching task experiment with e-prop learning.
- `plot_results.py`: Script for visualizing experiment results (loss curves, weights, spikes, etc.).
- `config/`: Contains configuration files (e.g., `config.yaml`) for experiment parameters.
- `sim_results/`: Stores simulation results, including plots and data for different experiment runs. Subfolders are created for each run/configuration.
- `trajectory*.txt`: Trajectory data files used in experiments (multiple files supported).

## Usage

You can run the main experiment script with various options for parameter sweeps and custom configurations:

### Basic Usage

To run a default experiment with the standard configuration, simply execute:

```bash
python eprop-reaching-task.py
```

This will:
- Use the parameters defined in `config/config.yaml`.
- Save all results, plots, and data in a new subfolder under `sim_results/` (e.g., `sim_results/default_plastic_input_to_rec_False`).
- Generate output files such as `spikes_and_dynamics.png`, `training_error.png`, and `results.npz` for later analysis.

You can then use `plot_results.py` to aggregate and compare results across different runs.

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

## Configuration

Experiment parameters are set in `config/config.yaml`. You can adjust simulation, task, RBF encoding, and neuron parameters there. For example, change the number of recurrent neurons or RBF centers to match your experiment needs.

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
Simulation results and plots are saved in the `sim_results/` directory, organized by experiment configuration. Each run creates a subfolder with files such as:
- `spikes_and_dynamics.png`: Visualization of network activity.
- `training_error.png`: Training loss curve.
- `weight_matrices.png`, `weight_time_courses.png`: Weight visualizations.
- `results.npz`: Raw results data.

You can use `plot_results.py` to aggregate and compare results across runs:
```bash
python plot_results.py
```

## Note on Dale's Law Version

The script `eprop_network_dale-law-not-applied.py` contains an older version of the network implementation without Dale's law. It is included here for reference only and may be removed or relocated in the future.

## License
<Specify your license here>.
