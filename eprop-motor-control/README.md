# eprop-motor-control

This directory contains code for simulating and analyzing motor control experiments using the e-prop (eligibility propagation) learning algorithm in spiking neural networks. The project trains and evaluates recurrent spiking networks performing a reaching task, with flexible configuration, parameter sweeps, and automated result visualization.

## Structure

- `eprop-reaching-task.py`: Main script for running the reaching task experiment with e-prop learning.
- `plot_results.py`: Script for visualizing experiment results like loss curves, weights, and spikes.
- `config/`: Contains configuration files (e.g., `config.yaml`) for experiment parameters.
- `sim_results/`: Stores simulation results, including plots and data. Subfolders are created for each run.
- `dataset_motor_training/`: Contains trajectory data files and the spike dataset used in experiments.
- `eprop_network_dale-law-not-applied.py`: An older version of the network without Dale's law, included for reference.

## Usage

You can run the main experiment script with various options for parameter sweeps and custom configurations.

### Basic Usage

To run a default experiment with the standard configuration, execute:

```bash
python eprop-reaching-task.py
```

This will use parameters from `config/config.yaml` and save all results into a new subfolder under `sim_results/`.

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

You can create the recommended environment using the provided `environment.yml` file and `mamba`, a faster drop-in replacement for `conda`:

```bash
mamba env create -f environment.yml
mamba activate motor-controller
```

This will install all necessary packages, including the NEST simulator.

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