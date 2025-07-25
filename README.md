# motor-controller-model

This repository contains code, data, and neuron models for simulating and analyzing motor control experiments using spiking neural networks and the e-prop (eligibility propagation) learning algorithm.

## Repository Structure

- `eprop-motor-control/` — Main module for running motor control experiments, training spiking networks, analyzing results, and visualizing outputs. See its README for detailed usage and options.
- `dataset_motor_training/` — Contains trajectory data, spike datasets, and utilities for dataset handling. Includes a README describing the dataset format.
- `nestml-neurons/` — NESTML neuron model files and generated NEST target code for custom neuron modules.
- `testing-nestml-neurons/` — Scripts and Jupyter notebooks for compiling, installing, and testing custom NESTML neuron models.
- `report/` — Documentation, figures, or analysis reports (if present).

## Getting Started

1. **Environment Setup:**
   - Use the provided `eprop-motor-control/environment.yml` to create a conda/mamba environment with all required dependencies (NEST, NESTML, Python packages, build tools).
   - Example:
     ```bash
     mamba env create -f eprop-motor-control/environment.yml
     mamba activate motor-controller
     ```

2. **Run Experiments:**
   - See `eprop-motor-control/README.md` for instructions on running experiments, parameter sweeps, and analyzing results.

3. **Dataset:**
   - See `dataset_motor_training/README.md` for details on the dataset format and usage.

## Submodule READMEs

Each main folder contains its own README with specific instructions and details. Refer to those for module-specific workflows.

## License
<Specify your license here>
