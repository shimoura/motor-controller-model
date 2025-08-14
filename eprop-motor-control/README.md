# eprop-motor-control

This module contains code for simulating and analyzing motor control experiments using the e-prop (eligibility propagation) learning algorithm in spiking neural networks. It trains and evaluates recurrent spiking networks performing a reaching task, with flexible configuration, parameter sweeps, and automated result visualization.

For a high-level overview of the repository, see the main `README.md` in the parent folder.

## Structure

- `eprop-reaching-task.py`: Main script for running the reaching task experiment with e-prop learning.
- `plot_results.py`: Script for visualizing experiment results like loss curves, weights, and spikes.
- `analyse_scan_results.py`: Script for analyzing parameter scan results and generating heatmaps.
- `config/`: Contains configuration files (e.g., `config.yaml`) for experiment parameters.
- `sim_results/`: Stores simulation results, including plots and data. Subfolders are created for each run.
- `dataset_motor_training/`, `nestml-neurons/`, `testing-nestml-neurons/`: See the main repository README for details about these sibling directories and their contents.
- `eprop_network_dale-law-not-applied.py`: An older version of the network without Dale's law, included for reference.
- `trained_weights_net.py`: Script for loading, analyzing, or visualizing trained network weights from experiments.

## Usage
### NESTML Neuron Compilation & Testing

To compile and install custom neuron models (NESTML):

```bash
python testing-nestml-neurons/compile_nestml_neurons.py
```

Or use the provided Jupyter notebooks in `testing-nestml-neurons/` for interactive compilation, installation, and testing of neuron models.

### Analysis of Parameter Scans

To analyze results from parameter sweeps and generate heatmaps:

```bash
python analyse_scan_results.py
```

You can run the main experiment script with various options for parameter sweeps and custom configurations.

### Basic Usage

To run a default experiment with the standard configuration, execute:

```bash
python eprop-reaching-task.py
```

This will use parameters from `config/config.yaml` and save all results into a new subfolder under `sim_results/`.

### Trained Weights Analysis

After running the main experiment, you can load and analyze the trained network weights:

```bash
python trained_weights_net.py
```

This script inspects, visualizes, or further processes the weights saved during training runs. See its docstring or comments for details on usage and options.

### Command-Line Options

- **Set both excitatory and inhibitory learning rates simultaneously:**

  ```bash
  python eprop-reaching-task.py --learning-rate 0.001
  ```

- **Use the manual RBF implementation instead of the default `rb_neuron`:**

  ```bash
  python eprop-reaching-task.py --use-manual-rbf
  ```

- **Make input-to-recurrent connections plastic:**

  ```bash
  python eprop-reaching-task.py --plastic-input-to-rec
  ```

- **Scan a single parameter (e.g., number of recurrent neurons):**

  ```bash
  python eprop-reaching-task.py --scan-param neurons.n_rec --scan-values 100,200,300
  ```

- **Scan multiple parameters (grid search):**

  ```bash
  python eprop-reaching-task.py --scan-param learning_rate_exc,rbf.num_centers --scan-values "0.01,0.001;10,20"
  ```

  *Note: Use quotes around `--scan-values` to avoid shell parsing issues with semicolons.*

- **Disable plotting:**

  ```bash
  python eprop-reaching-task.py --no-plot
  ```

## Configuration

Experiment parameters are set in `config/config.yaml`. You can adjust simulation, task, RBF encoding, and neuron parameters there. These can be overridden at runtime with command-line arguments.

## Environment Setup

The recommended environment includes NEST, NESTML, and build tools. See `environment.yml` for details. Key dependencies:

- nest-simulator
- nestml
- numpy, pandas, matplotlib, h5py, statsmodels
- cmake, make, boost, gsl

Create and activate the environment:

```bash
mamba env create -f environment.yml
mamba activate motor-controller
```

This will install all necessary packages, including the NEST simulator and NESTML.

## Results

Simulation results and plots are saved in the `sim_results/` directory, organized by experiment configuration. Each run creates a subfolder with files such as:

- `training_error.png`: The training loss curve.
- `spikes_and_dynamics.png`: Visualization of network activity.
- `weight_matrices.png` & `weight_time_courses.png`: Weight visualizations.
- `results.npz`: Raw results data.

To aggregate and compare results across runs, use `plot_results.py`:

```bash
python plot_results.py
```

## Note on Dale's Law Version

The script `eprop_network_dale-law-not-applied.py` contains an older version of the network implementation without Dale's law and is included here for reference only.

## License
<Specify your license here>