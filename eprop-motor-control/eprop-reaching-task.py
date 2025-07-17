# -*- coding: utf-8 -*-
#
# eprop_reaching_task.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

r"""
Tutorial on learning to perform a reaching task with e-prop
-----------------------------------------------------------

Training a regression model using supervised e-prop plasticity to perform a reaching task.

Description
~~~~~~~~~~~

This script demonstrates supervised learning of a regression task with a recurrent spiking
neural network (SNN) that is equipped with the eligibility propagation (e-prop) plasticity
mechanism by Bellec et al. [1]_.

In this task, the network learns to generate a reaching trajectory. The network learns to
reproduce with its overall spiking activity a one-dimensional, one-second-long target signal
which represents the reaching trajectory.

Learning in the SNN is achieved by optimizing the connection weights with e-prop plasticity.
This plasticity rule requires a specific network architecture. The model consists of a
recurrent network that receives input encoding the trajectory and projects onto readout
neurons. The readout neurons compare the network's filtered output signal, $y$, with a
teacher target signal, $y^*$. The network's training error is assessed by employing a
mean-squared error loss.

Details on the event-based NEST implementation of e-prop can be found in [2]_.

This script includes two methods for encoding the input trajectory:
1.  **rb_neuron**: A custom NEST neuron model that performs Radial Basis Function (RBF)
    encoding internally. This is the default and recommended method.
2.  **Manual RBF**: A manual implementation where RBF activation is calculated in NumPy
    and fed to Poisson generators. This can be enabled with the `--use-manual-rbf` flag.

References
~~~~~~~~~~

.. [1] Bellec G, Scherr F, Subramoney F, Hajek E, Salaj D, Legenstein R, Maass W (2020).
       A solution to the learning dilemma for recurrent networks of spiking neurons.
       Nature Communications, 11:3625. https://doi.org/10.1038/s41467-020-17236-y

.. [2] Korcsak-Gorzo A, Stapmanns J, Espinoza Valverde JA, Dahmen D, van Albada SJ,
       Bolten M, Diesmann M. Event-based implementation of eligibility propagation
       (in preparation)
"""

# %% ###########################################################################################################
# Import libraries
# ~~~~~~~~~~~~~~~~
# We begin by importing all libraries required for the simulation, analysis, and visualization.

import matplotlib

matplotlib.use("Agg")

import nest
import copy
import numpy as np
import yaml
import sys
from pathlib import Path
import os
import itertools

# Add the parent directory to the system path for dataset import
sys.path.append(str(Path(__file__).resolve().parent.parent / "dataset_motor_training"))
from load_dataset import load_data_file
from plot_results import (
    plot_training_error,
    plot_spikes_and_dynamics,
    plot_weight_time_courses,
    plot_weight_matrices,
    plot_all_loss_curves,
)


def run_simulation(
    n_rec=None,
    n_out=None,
    learning_rate_exc=None,
    learning_rate_inh=None,
    exc_ratio=None,
    config_path_override=None,
    result_dir=None,
    plot_results=True,
    plastic_input_to_rec=False,
    use_manual_rbf=False,
    **override_kwargs,
):
    """
    Run the e-prop reaching task simulation with optional parameter overrides.

    Args:
        n_rec (int, optional): Number of recurrent neurons.
        n_out (int, optional): Number of output neurons.
        learning_rate_exc (float, optional): Learning rate for excitatory synapses.
        learning_rate_inh (float, optional): Learning rate for inhibitory synapses.
        exc_ratio (float, optional): Ratio of excitatory neurons.
        config_path_override (str/Path, optional): Path to a custom config file.
        result_dir (str, optional): Directory for result files.
        plot_results (bool): Whether to generate and save plots.
        plastic_input_to_rec (bool): If True, make input-to-recurrent connections plastic.
        use_manual_rbf (bool): If True, use manual RBF implementation instead of rb_neuron.
        override_kwargs (dict): Other config overrides using dot notation (e.g., 'rbf.num_centers').
    """
    # %% ###########################################################################################################
    # Load configuration
    # ~~~~~~~~~~~~~~~~~~~~
    # Load the base configuration from the YAML file and override any parameters
    # specified through function arguments or keyword arguments.
    config_path = (
        Path(config_path_override)
        if config_path_override
        else Path(__file__).resolve().parent / "config" / "config.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override config with function arguments
    if n_rec is not None:
        config["neurons"]["n_rec"] = n_rec
    if n_out is not None:
        config["neurons"]["n_out"] = n_out
    if exc_ratio is not None:
        config["neurons"]["exc_ratio"] = exc_ratio
    # Map user-friendly 'learning_rate' to NEST's 'eta'
    if learning_rate_exc is not None:
        config["synapses"]["exc"]["optimizer"]["eta"] = learning_rate_exc
    if learning_rate_inh is not None:
        config["synapses"]["inh"]["optimizer"]["eta"] = learning_rate_inh
    # Remove any accidental learning_rate keys to avoid confusion
    config["synapses"]["exc"]["optimizer"].pop("learning_rate", None)
    config["synapses"]["inh"]["optimizer"].pop("learning_rate", None)

    # Apply any other keyword argument overrides using dot notation
    for k, v in override_kwargs.items():
        keys = k.split(".")
        d = config
        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = v

    # %% ###########################################################################################################
    # Setup Simulation
    # ~~~~~~~~~~~~~~~~
    # Extract parameters from the config and set up the simulation kernel and timing.

    # Extract simulation and task parameters
    sim_cfg = config["simulation"]
    task_cfg = config["task"]
    step_ms = sim_cfg["step"]
    n_timesteps_per_sequence = int(round(task_cfg["sequence"] / step_ms))

    # Define the timing structure of the experiment
    duration = {
        "step": step_ms,
        "sequence": task_cfg["sequence"],
        "learning_window": task_cfg["sequence"],
        "extension_sim": step_ms, # Extension time after task ends
    }
    trajectory_ids_to_use = task_cfg["trajectory_ids_to_use"]
    n_samples_per_trajectory_to_use = int(task_cfg["n_samples_per_trajectory_to_use"])
    n_samples = len(trajectory_ids_to_use) * n_samples_per_trajectory_to_use
    n_iter = int(task_cfg["n_iter"])
    duration["task"] = n_timesteps_per_sequence * n_samples * n_iter * step_ms
    duration["sim"] = duration["task"] + duration["extension_sim"]

    # Set up NEST kernel
    params_setup = {
        "eprop_learning_window": duration["learning_window"],
        "eprop_reset_neurons_on_update": False,
        "eprop_update_interval": duration["sequence"],
        "print_time": sim_cfg["print_time"],
        "resolution": duration["step"],
        "total_num_virtual_procs": sim_cfg["total_num_virtual_procs"],
        "rng_seed": sim_cfg["rng_seed"],
    }
    nest.ResetKernel()
    nest.set(**params_setup)

    # %% ###########################################################################################################
    # Create Neurons
    # ~~~~~~~~~~~~~~
    # We create input, recurrent, and readout neurons based on the configuration.
    # The input layer's implementation depends on the `use_manual_rbf` flag.
    nrn_cfg = config["neurons"]
    rbf_cfg = config["rbf"]
    num_centers = int(rbf_cfg["num_centers"])

    if use_manual_rbf:
        # Manual RBF: Input layer is a set of Poisson generators
        print("Using manual RBF implementation.")
        n_in = num_centers
        gen_poisson_in = nest.Create("inhomogeneous_poisson_generator", n_in)
    else:
        # Default: Use the custom 'rb_neuron' model for RBF encoding
        print("Using 'rb_neuron' for RBF implementation.")
        nest.Install("motor_neuron_module")
        params_rb_neuron = nrn_cfg["rb"]
        n_rb = num_centers
        gen_poisson_in = nest.Create("inhomogeneous_poisson_generator")
        nrns_rb = nest.Create("rb_neuron", n_rb)
        # Configure the rb_neuron population
        params_rb_neuron["simulation_steps"] = int(
            duration["sim"] / duration["step"] + 1
        )
        params_rb_neuron["sdev"] = rbf_cfg["scale_rate"] * rbf_cfg["width"]
        nest.SetStatus(nrns_rb, params_rb_neuron)

    # Create recurrent and readout populations
    n_rec = int(nrn_cfg["n_rec"])
    n_out = int(nrn_cfg["n_out"])
    exc_ratio = nrn_cfg["exc_ratio"]
    n_rec_exc = int(n_rec * exc_ratio)
    n_rec_inh = n_rec - n_rec_exc
    params_nrn_rec = nrn_cfg["rec"]
    params_nrn_out = nrn_cfg["out"]

    nrns_rec_exc = nest.Create("eprop_iaf_bsshslm_2020", n_rec_exc, params_nrn_rec)
    nrns_rec_inh = nest.Create("eprop_iaf_bsshslm_2020", n_rec_inh, params_nrn_rec)
    nrns_rec = nrns_rec_exc + nrns_rec_inh

    nrns_out = nest.Create("eprop_readout_bsshslm_2020", n_out, params_nrn_out)
    gen_rate_target = nest.Create("step_rate_generator", n_out)

    # %% ###########################################################################################################
    # Create Recorders
    # ~~~~~~~~~~~~~~~~
    # We set up recorders to monitor membrane potentials, signals, spikes, and weights.
    rec_cfg = config["recording"]
    n_record, n_record_w = rec_cfg["n_record"], rec_cfg["n_record_w"]
    if n_record < 1 or n_record_w < 1:
        raise ValueError("'n_record' and 'n_record_w' must be at least 1.")

    start_time = duration["step"]
    stop_time = duration["task"]
    params_mm_rec = {
        **rec_cfg["mm_rec"],
        "interval": duration["step"],
        "start": start_time,
        "stop": stop_time,
    }
    params_mm_out = {
        **rec_cfg["mm_out"],
        "interval": duration["step"],
        "start": start_time,
        "stop": stop_time,
    }
    params_wr = {
        "senders": nrns_rec[:n_record_w],
        "targets": nrns_rec[:n_record_w] + nrns_out,
        "start": start_time,
        "stop": stop_time,
    }
    params_sr = {"start": start_time, "stop": stop_time}

    mm_rec = nest.Create("multimeter", params_mm_rec)
    mm_out = nest.Create("multimeter", params_mm_out)
    spike_recorder = nest.Create("spike_recorder", params_sr)
    weight_recorder = nest.Create("weight_recorder", params_wr)
    nrns_rec_record = nrns_rec[:n_record]

    # %% ###########################################################################################################
    # Create Connections
    # ~~~~~~~~~~~~~~~~~~
    # Define connectivity rules and synapse parameters, then build the network.
    syn_cfg = config["synapses"]
    gradient_batch_size = int(task_cfg["gradient_batch_size"])
    w_input, w_rec, g = syn_cfg["w_input"], syn_cfg["w_rec"], syn_cfg["g"]

    # Define synapse models and parameters
    params_conn_all_to_all = {"rule": "all_to_all", "allow_autapses": False}
    params_conn_one_to_one = {"rule": "one_to_one"}
    params_conn_bernoulli = {
        "rule": "pairwise_bernoulli",
        "p": syn_cfg["conn_bernoulli_p"],
        "allow_autapses": False,
    }
    params_syn_eprop_exc = {
        "optimizer": {**syn_cfg["exc"]["optimizer"], "batch_size": gradient_batch_size},
        "average_gradient": syn_cfg["average_gradient"],
        "weight_recorder": weight_recorder,
    }
    params_syn_eprop_inh = {
        "optimizer": {**syn_cfg["inh"]["optimizer"], "batch_size": gradient_batch_size},
        "weight": syn_cfg["inh"]["weight"],
        "average_gradient": syn_cfg["average_gradient"],
        "weight_recorder": weight_recorder,
    }
    params_syn_base = {
        "synapse_model": "eprop_synapse_bsshslm_2020",
        "delay": duration["step"],
        "tau_m_readout": params_nrn_out["tau_m"],
    }
    params_syn_rec_exc = {
        **copy.deepcopy(params_syn_base),
        "weight": nest.math.redraw(
            nest.random.normal(mean=w_rec, std=w_rec * 0.1), min=0.0, max=1000.0
        ),
        "synapse_model": "eprop_synapse_bsshslm_2020_exc",
    }
    params_syn_rec_inh = {
        **copy.deepcopy(params_syn_base),
        "weight": nest.math.redraw(
            nest.random.normal(mean=-w_rec * g, std=g * w_rec * 0.1),
            min=-1000.0,
            max=0.0,
        ),
        "synapse_model": "eprop_synapse_bsshslm_2020_inh",
    }
    params_syn_feedback = {
        "synapse_model": "eprop_learning_signal_connection_bsshslm_2020",
        "delay": duration["step"],
        "weight": nest.math.redraw(
            nest.random.normal(mean=w_rec, std=w_rec * 0.1), min=0.0, max=1000.0
        ),
    }
    params_syn_rate_target = {
        "synapse_model": "rate_connection_delayed",
        "delay": duration["step"],
        "receptor_type": syn_cfg["receptor_type"],
    }
    params_syn_static = {"synapse_model": "static_synapse", "delay": duration["step"]}

    # Copy base models to create specific plastic synapse types
    nest.CopyModel(
        "eprop_synapse_bsshslm_2020",
        "eprop_synapse_bsshslm_2020_exc",
        params_syn_eprop_exc,
    )
    nest.CopyModel(
        "eprop_synapse_bsshslm_2020",
        "eprop_synapse_bsshslm_2020_inh",
        params_syn_eprop_inh,
    )

    # Connect input layer to recurrent layer
    if use_manual_rbf:
        params_syn_input = {
            "synapse_model": "static_synapse",
            "delay": duration["step"],
            "weight": nest.math.redraw(
                nest.random.normal(mean=w_input, std=w_input * 0.1), min=0.0, max=1000.0
            ),
        }
        if plastic_input_to_rec:
            nest.Connect(
                gen_poisson_in, nrns_rec, params_conn_all_to_all, params_syn_rec_exc
            )
        else:
            for i, poisson_node in enumerate(gen_poisson_in):
                nest.Connect(
                    poisson_node,
                    nrns_rec_exc[
                        int(i * n_rec_exc / n_in) : int((i + 1) * n_rec_exc / n_in)
                    ],
                    params_conn_all_to_all,
                    params_syn_input,
                )
                nest.Connect(
                    poisson_node,
                    nrns_rec_inh[
                        int(i * n_rec_inh / n_in) : int((i + 1) * n_rec_inh / n_in)
                    ],
                    params_conn_all_to_all,
                    params_syn_input,
                )
    else:  # rb_neuron connectivity
        params_syn_input_to_rb = {
            "synapse_model": "static_synapse",
            "delay": duration["step"],
            "weight": 1.0,
        }
        params_syn_rb_to_rec = {
            "synapse_model": "static_synapse",
            "delay": duration["step"],
            "weight": nest.math.redraw(
                nest.random.normal(mean=w_input, std=w_input * 0.1), min=0.0, max=1000.0
            ),
        }
        nest.Connect(
            gen_poisson_in, nrns_rb, params_conn_all_to_all, params_syn_input_to_rb
        )
        if plastic_input_to_rec:
            nest.Connect(nrns_rb, nrns_rec, params_conn_all_to_all, params_syn_rec_exc)
        else:
            for i, input_node in enumerate(nrns_rb):
                nest.Connect(
                    input_node,
                    nrns_rec_exc[
                        int(i * n_rec_exc / n_rb) : int((i + 1) * n_rec_exc / n_rb)
                    ],
                    params_conn_all_to_all,
                    params_syn_rb_to_rec,
                )
                nest.Connect(
                    input_node,
                    nrns_rec_inh[
                        int(i * n_rec_inh / n_rb) : int((i + 1) * n_rec_inh / n_rb)
                    ],
                    params_conn_all_to_all,
                    params_syn_rb_to_rec,
                )

    # Connect recurrent neurons to each other
    nest.Connect(nrns_rec_exc, nrns_rec, params_conn_bernoulli, params_syn_rec_exc)
    nest.Connect(nrns_rec_inh, nrns_rec, params_conn_bernoulli, params_syn_rec_inh)

    # Connect recurrent neurons to readout neurons
    nest.Connect(nrns_rec_exc, nrns_out, params_conn_all_to_all, params_syn_rec_exc)
    nest.Connect(nrns_rec_inh, nrns_out, params_conn_all_to_all, params_syn_rec_inh)

    # Connect readout neurons to the target signal generator
    nest.Connect(nrns_out, nrns_rec_exc, params_conn_all_to_all, params_syn_feedback)
    nest.Connect(nrns_out, nrns_rec_inh, params_conn_all_to_all, params_syn_feedback)

    # Connect the target signal generators to the readout neurons
    nest.Connect(
        gen_rate_target[0], nrns_out[0], params_conn_one_to_one, params_syn_rate_target
    )
    nest.Connect(
        gen_rate_target[1], nrns_out[1], params_conn_one_to_one, params_syn_rate_target
    )

    # Connect the recorder devices
    nest.Connect(nrns_rec, spike_recorder, params_conn_all_to_all, params_syn_static)
    nest.Connect(mm_rec, nrns_rec_record, params_conn_all_to_all, params_syn_static)
    nest.Connect(mm_out, nrns_out, params_conn_all_to_all, params_syn_static)

    # %% ###########################################################################################################
    # Load and Prepare Data
    # ~~~~~~~~~~~~~~~~~~~~~~~
    # Load the trajectory and target spike data, then resample and process it.
    dataset_path = (
        Path(__file__).resolve().parent.parent
        / "dataset_motor_training"
        / "dataset_spikes.gdf"
    )
    training_dataset = load_data_file(str(dataset_path))
    sample_ids = [
        tid * task_cfg["samples_per_trajectory_in_dataset"] + j
        for tid in trajectory_ids_to_use
        for j in range(n_samples_per_trajectory_to_use)
    ]
    assert len(sample_ids) == n_samples

    trajectories, desired_targets_list = [], {"pos": [], "neg": []}
    for sample_id in sample_ids:
        traj_num = int(training_dataset[sample_id][0][0])
        traj_file = dataset_path.parent / f"trajectory{traj_num}.txt"
        traj_data = np.loadtxt(traj_file)
        orig_num_pts, orig_dur = (
            len(traj_data),
            len(traj_data) * 0.1,
        )  # 0.1ms is original resolution
        resampled_time = np.linspace(
            0, orig_dur, n_timesteps_per_sequence, endpoint=False
        )
        orig_time = np.arange(orig_num_pts) * 0.1
        trajectories.append(np.interp(resampled_time, orig_time, traj_data))

        # Process target spike times into smoothed rates
        for i, key in enumerate(["pos", "neg"]):
            spike_times = training_dataset[sample_id][2 * i + 2]
            target_hist = np.histogram(
                spike_times,
                bins=n_timesteps_per_sequence,
                range=(0, duration["sequence"]),
            )[0]
            desired_targets_list[key].append(
                np.convolve(target_hist, np.ones(20) / 10, mode="same")
            )

    # %% ###########################################################################################################
    # Create Input and Output Signals
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Convert the processed data into timed rate signals for the NEST generators.
    if use_manual_rbf:

        def gaussian_rbf(x, center, width):
            return np.exp(-((x - center) ** 2) / (2 * width**2))

        # Standard deviation for Gaussian RBF
        width = rbf_cfg["width"]

        # Define the number of centers and create RBF inputs
        centers = np.linspace(np.min(trajectories), np.max(trajectories), num_centers)
        rbf_inputs_list = []
        for trajectory_sample in trajectories:
            rbf_inputs_for_sample = np.zeros((n_timesteps_per_sequence, num_centers))
            for i, center in enumerate(centers):
                rbf_inputs_for_sample[:, i] = gaussian_rbf(
                    trajectory_sample, center, width
                )
            rbf_inputs_list.append(rbf_inputs_for_sample)

        # Stack the RBF inputs from all samples and scale them to get firing rates.
        rate_based_rbf_inputs = np.vstack(rbf_inputs_list) * rbf_cfg["scale_rate"]

        # Tile the signals for all training iterations.
        tiled_rbf_inputs = np.tile(rate_based_rbf_inputs, (n_iter, 1))

        # Set the rate for each Poisson generator.
        in_rate_times = np.arange(tiled_rbf_inputs.shape[0]) * duration["step"] + duration["step"]
        params_gen_poisson_in = [
            {"rate_times": in_rate_times, "rate_values": tiled_rbf_inputs[:, i]}
            for i in range(num_centers)
        ]
        nest.SetStatus(gen_poisson_in, params_gen_poisson_in)
    else:  # rb_neuron input creation
        input_spk_rate = rbf_cfg["scale_rate"] * np.concatenate(trajectories)
        input_spk_rate = np.tile(input_spk_rate, n_iter)
        in_rate_times = np.arange(len(input_spk_rate)) * duration["step"] + duration["step"]
        nest.SetStatus(
            gen_poisson_in, {"rate_times": in_rate_times, "rate_values": input_spk_rate}
        )
        # Set the desired center for each rb_neuron's receptive field
        angle_centers = np.linspace(
            np.min(trajectories), np.max(trajectories), num_centers
        )
        desired_rates = rbf_cfg["scale_rate"] * angle_centers
        for i, nrn in enumerate(nrns_rb):
            nest.SetStatus(nrn, {"desired": desired_rates[i]})

    # Create target signals for the readout neurons
    concatenated_targets = {
        k: np.concatenate(v) for k, v in desired_targets_list.items()
    }
    target_amp_times = (
        np.arange(len(concatenated_targets["pos"]) * n_iter) * duration["step"] + duration["step"]
    )
    params_gen_rate_target = [
        {
            "amplitude_times": target_amp_times,
            "amplitude_values": np.tile(concatenated_targets[k], n_iter),
        }
        for k in concatenated_targets
    ]
    nest.SetStatus(gen_rate_target, params_gen_rate_target)

    # %% ###########################################################################################################
    # Pre-Simulation Finalization
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Force a final spike after the task to ensure all weights are updated for readout.
    # Also, read out the initial weight matrices for later comparison.
    final_update_time = duration["task"] + duration["step"]
    gen_spk_final_update = nest.Create(
        "spike_generator", 1, {"spike_times": [final_update_time]}
    )
    nest.Connect(gen_spk_final_update, nrns_rec, "all_to_all", {"weight": 1000.0})

    def get_weights(pop_pre, pop_post):
        conns = nest.GetConnections(pop_pre, pop_post).get(
            ["source", "target", "weight"]
        )
        if not conns["source"]:
            return {"weight_matrix": np.zeros((len(pop_post), len(pop_pre)))}
        conns["senders"] = np.array(conns["source"]) - np.min(conns["source"])
        conns["targets"] = np.array(conns["target"]) - np.min(conns["target"])
        conns["weight_matrix"] = np.zeros((len(pop_post), len(pop_pre)))
        conns["weight_matrix"][conns["targets"], conns["senders"]] = conns["weight"]
        return conns

    weights_pre_train = {
        "rec_rec": get_weights(nrns_rec, nrns_rec),
        "rec_out": get_weights(nrns_rec, nrns_out),
    }

    # %% ###########################################################################################################
    # Simulate and Process Results
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We run the simulation, read out the recorded data, and calculate the training error.
    print(f"Starting simulation for {duration['sim']:.2f} ms...")
    nest.Simulate(duration["sim"])
    print("Simulation finished.")

    weights_post_train = {
        "rec_rec": get_weights(nrns_rec, nrns_rec),
        "rec_out": get_weights(nrns_rec, nrns_out),
    }
    events_mm_rec, events_mm_out, events_sr, events_wr = (
        mm_rec.get("events"),
        mm_out.get("events"),
        spike_recorder.get("events"),
        weight_recorder.get("events"),
    )

    # Evaluate training error (loss)
    readout_signal, target_signal = (
        events_mm_out["readout_signal"],
        events_mm_out["target_signal"],
    )
    error = (readout_signal - target_signal) ** 2
    loss_indices = np.arange(0, int(duration["task"]), int(duration["sequence"]))
    loss = 0.5 * np.add.reduceat(error, loss_indices)

    # %% ###########################################################################################################
    # Plotting and Saving
    # ~~~~~~~~~~~~~~~~~~~~~
    # Generate and save plots and simulation results if requested.
    if plot_results and config.get("plotting", {}).get("do_plotting", True):
        print("Generating and saving plots...")
        out_dir = result_dir if result_dir else "."
        colors = {"blue": "#1f77b4", "red": "#d62728", "white": "#ffffff"}
        plot_training_error(loss, os.path.join(out_dir, "training_error.png"))
        plot_spikes_and_dynamics(
            events_sr,
            events_mm_rec,
            events_mm_out,
            nrns_rec,
            n_record,
            duration,
            colors,
            os.path.join(out_dir, "spikes_and_dynamics.png"),
        )
        plot_weight_time_courses(
            events_wr,
            weights_pre_train,
            nrns_rec,
            nrns_out,
            n_record_w,
            colors,
            duration,
            os.path.join(out_dir, "weight_time_courses.png"),
        )
        plot_weight_matrices(
            weights_pre_train,
            weights_post_train,
            colors,
            os.path.join(out_dir, "weight_matrices.png"),
        )

        if result_dir:
            import json

            def make_json_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [make_json_serializable(i) for i in obj]
                if isinstance(obj, Path):
                    return str(obj)
                if isinstance(obj, (np.ndarray, np.generic)):
                    return obj.tolist()
                return obj

            config_serializable = make_json_serializable(config)
            config_serializable["model_run_settings"] = {
                "plastic_input_to_rec": plastic_input_to_rec,
                "use_manual_rbf": use_manual_rbf,
            }
            np.savez(
                os.path.join(out_dir, "results.npz"),
                loss=loss,
                readout_signal=readout_signal,
                target_signal=target_signal,
            )
            with open(os.path.join(out_dir, "config.json"), "w") as f:
                json.dump(config_serializable, f, indent=2)
        print(f"Results saved to {out_dir}")


# ---
# Main execution block
# This block parses command-line arguments to run single or multiple simulations (parameter scans).
# ---
if __name__ == "__main__":
    import argparse

    # ---
    # Usage Examples:
    #
    # 1. Run a single simulation with default parameters:
    #    python eprop-reaching-task.py
    #
    # 2. Run with manual RBF encoding:
    #    python eprop-reaching-task.py --use-manual-rbf
    #
    # 3. Make input-to-recurrent connections plastic:
    #    python eprop-reaching-task.py --plastic-input-to-rec
    #
    # 4. Scan a single parameter (e.g., number of recurrent neurons):
    #    python eprop-reaching-task.py --scan-param neurons.n_rec --scan-values 100,200,300
    #
    # 5. Set both excitatory and inhibitory learning rates simultaneously:
    #    python eprop-reaching-task.py --learning-rate 0.005
    #
    # 6. Perform a grid search over multiple parameters:
    #    python eprop-reaching-task.py --scan-param learning_rate_exc,rbf.num_centers --scan-values "0.01,0.001;10,20"
    #    (Note: Use quotes around --scan-values to avoid shell parsing issues with semicolons)
    #
    # 7. Combine a fixed parameter change with a scan:
    #    python eprop-reaching-task.py --learning-rate 0.001 --scan-param neurons.n_rec --scan-values 150,250
    # ---

    parser = argparse.ArgumentParser(
        description="Run e-prop reaching task parameter scan.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--scan-param",
        type=str,
        default=None,
        help="Comma-separated list of config parameters to scan (e.g., neurons.n_rec).",
    )
    parser.add_argument(
        "--scan-values",
        type=str,
        default=None,
        help='Semicolon-separated lists of values for each scan parameter (e.g., "100,200;10,20").',
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Set both excitatory and inhibitory learning rates simultaneously.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable all plotting.")
    parser.add_argument(
        "--plastic-input-to-rec",
        action="store_true",
        help="Make input-to-recurrent connections plastic and fully connected.",
    )
    parser.add_argument(
        "--use-manual-rbf",
        action="store_true",
        help="Use manual RBF implementation instead of the default 'rb_neuron' model.",
    )
    args = parser.parse_args()

    # Prepare for single run or parameter scan
    results_dir = os.path.join(os.path.dirname(__file__), "sim_results")
    os.makedirs(results_dir, exist_ok=True)

    base_kwargs = {
        "plot_results": not args.no_plot,
        "plastic_input_to_rec": args.plastic_input_to_rec,
        "use_manual_rbf": args.use_manual_rbf,
    }
    if args.learning_rate is not None:
        base_kwargs["learning_rate_exc"] = args.learning_rate
        base_kwargs["learning_rate_inh"] = args.learning_rate

    scan_params = (
        [p.strip() for p in args.scan_param.split(",")] if args.scan_param else []
    )
    scan_values = []
    if args.scan_values:
        for val_list in args.scan_values.split(";"):
            vals = [
                float(v) if v.replace(".", "", 1).replace("-", "", 1).isdigit() else v
                for v in val_list.split(",")
            ]
            scan_values.append(vals)

    # Execute simulation runs
    if scan_params and scan_values and len(scan_params) == len(scan_values):
        print(f"--- Starting Parameter Scan ---")
        for combo in itertools.product(*scan_values):
            param_dict = dict(zip(scan_params, combo))
            scenario_kwargs = {**base_kwargs, **param_dict}
            folder_name = "_".join(
                f"{k.replace('.', '_')}_{v}" for k, v in param_dict.items()
            )
            folder_name += (
                f"_plastic_{args.plastic_input_to_rec}_manualRBF_{args.use_manual_rbf}"
            )
            sim_dir = os.path.join(results_dir, folder_name)
            os.makedirs(sim_dir, exist_ok=True)
            print(f"\nRunning scenario: {param_dict}")
            run_simulation(result_dir=sim_dir, **scenario_kwargs)
        print("\n--- Parameter Scan Finished ---")
    else:
        folder_name = f"default_plastic_{args.plastic_input_to_rec}_manualRBF_{args.use_manual_rbf}"
        sim_dir = os.path.join(results_dir, folder_name)
        os.makedirs(sim_dir, exist_ok=True)
        print("--- Running Single Default Scenario ---")
        run_simulation(result_dir=sim_dir, **base_kwargs)
        print("\n--- Default Scenario Finished ---")

    # Plot summary graph comparing all runs
    if not args.no_plot and os.path.exists(results_dir):
        print("\nGenerating summary loss curve plot...")
        plot_all_loss_curves(results_dir, showfig=False)
        print("Summary plot saved.")
