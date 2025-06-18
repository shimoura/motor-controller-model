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

Training a regression model using supervised e-prop plasticity to perform a reaching task

Description
~~~~~~~~~~~

This script demonstrates supervised learning of a regression task with a recurrent spiking neural network that
is equipped with the eligibility propagation (e-prop) plasticity mechanism by Bellec et al. [1]_.

In this task, the network learns to generate a reaching trajectory. The network learns to reproduce with its
overall spiking activity a one-dimensional, one-second-long target signal which represents the reaching trajectory.

Learning in the neural network model is achieved by optimizing the connection weights with e-prop plasticity.
This plasticity rule requires a specific network architecture depicted in Figure 1. The neural network model
consists of a recurrent network that receives frozen noise input from Poisson generators and projects onto one
readout neuron. The readout neuron compares the network signal :math:`y` with the teacher target signal
:math:`y*`, which it receives from a rate generator. In scenarios with multiple readout neurons, each individual
readout signal denoted as :math:`y_k` is compared with a corresponding target signal represented as
:math:`y_k^*`. The network's training error is assessed by employing a mean-squared error loss.

Details on the event-based NEST implementation of e-prop can be found in [2]_.

References
~~~~~~~~~~

.. [1] Bellec G, Scherr F, Subramoney F, Hajek E, Salaj D, Legenstein R, Maass W (2020). A solution to the
       learning dilemma for recurrent networks of spiking neurons. Nature Communications, 11:3625.
       https://doi.org/10.1038/s41467-020-17236-y

.. [2] Korcsak-Gorzo A, Stapmanns J, Espinoza Valverde JA, Dahmen D, van Albada SJ, Bolten M, Diesmann M.
       Event-based implementation of eligibility propagation (in preparation)
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
    plastic_input_to_rec=False,  # <-- new option
    **override_kwargs,
):
    """
    Run the e-prop reaching task simulation with optional parameter overrides.
    Args:
        n_rec: Number of recurrent neurons (int)
        n_out: Number of output neurons (int)
        learning_rate_exc: Learning rate for excitatory synapses (float)
        learning_rate_inh: Learning rate for inhibitory synapses (float)
        exc_ratio: Ratio of excitatory neurons (float)
        config_path_override: Path to config file (str or Path)
        result_dir: Directory for result files (str)
        plot_results: Whether to plot results (bool)
        override_kwargs: Other config overrides (dict)
    """

    # Load config
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
    # Ensure correct optimizer keys for NEST: eta, not learning_rate
    # If user tries to override with learning_rate, map it to eta
    if learning_rate_exc is not None:
        config["synapses"]["exc"]["optimizer"]["eta"] = learning_rate_exc
    if learning_rate_inh is not None:
        config["synapses"]["inh"]["optimizer"]["eta"] = learning_rate_inh
    # Remove any accidental learning_rate keys
    config["synapses"]["exc"]["optimizer"].pop("learning_rate", None)
    config["synapses"]["inh"]["optimizer"].pop("learning_rate", None)
    for k, v in override_kwargs.items():
        # Support dot notation for nested keys
        keys = k.split(".")
        d = config
        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = v

    # %% ###########################################################################################################
    # Setup
    # ~~~~~

    # %% ###########################################################################################################
    # Define timing of task
    # .....................
    # The task's temporal structure is now defined only in milliseconds.

    # Use parameters from config
    sim_cfg = config["simulation"]
    task_cfg = config["task"]

    n_batch = task_cfg["n_batch"]
    n_iter = task_cfg["n_iter"]
    n_samples = task_cfg["n_samples"]

    # Compute all timing values directly in milliseconds
    step_ms = sim_cfg["step"]
    n_timesteps_per_sequence = int(round(task_cfg["sequence"] / step_ms))
    duration = {
        "step": step_ms,
        "sequence": task_cfg["sequence"],
        "learning_window": task_cfg["sequence"],
        # Ensure task duration matches integer number of time steps
        "task": n_timesteps_per_sequence * n_samples * n_iter * n_batch * step_ms,
        "offset_gen": task_cfg["offset_gen"] * step_ms,
        "delay_in_rec": task_cfg["delay_in_rec"] * step_ms,
        "delay_rec_out": task_cfg["delay_rec_out"] * step_ms,
        "delay_out_norm": task_cfg["delay_out_norm"] * step_ms,
        "extension_sim": task_cfg["extension_sim"] * step_ms,
    }
    duration["delays"] = (
        duration["delay_in_rec"]
        + duration["delay_rec_out"]
        + duration["delay_out_norm"]
    )
    duration["total_offset"] = duration["offset_gen"] + duration["delays"]
    duration["sim"] = (
        duration["task"] + duration["total_offset"] + duration["extension_sim"]
    )

    # Calculate number of time steps per sequence
    n_timesteps_per_sequence = int(duration["sequence"] / duration["step"])

    # Set up simulation
    params_setup = {
        "eprop_learning_window": duration["learning_window"],
        "eprop_reset_neurons_on_update": True,
        "eprop_update_interval": duration["sequence"],
        "print_time": sim_cfg["print_time"],
        "resolution": duration["step"],
        "total_num_virtual_procs": sim_cfg["total_num_virtual_procs"],
        "rng_seed": sim_cfg["rng_seed"],
    }

    ####################

    nest.ResetKernel()
    nest.set(**params_setup)

    # %% ###########################################################################################################
    # Set input parameters
    # ~~~~~~~~~~~~~~~~~~~~~

    # RBF parameters from config
    num_centers = int(
        config["rbf"]["num_centers"]
    )  # Ensure integer for np.linspace and indexing
    centers = np.linspace(0.0, np.pi / 2.0, num_centers)
    width = config["rbf"]["width"]
    scale_rate = config["rbf"]["scale_rate"]

    # %% ###########################################################################################################
    # Create neurons
    # ~~~~~~~~~~~~~~
    # We proceed by creating a certain number of recurrent and readout neurons and setting their parameters.
    # Additionally, we already create an output target rate generator, which we will configure later.

    # Neuron parameters from config
    n_in = num_centers
    n_rec = int(config["neurons"]["n_rec"])
    n_out = int(config["neurons"]["n_out"])
    exc_ratio = config["neurons"]["exc_ratio"]
    n_rec_exc = int(n_rec * exc_ratio)
    n_rec_inh = n_rec - n_rec_exc
    params_nrn_rec = config["neurons"]["rec"]
    params_nrn_out = config["neurons"]["out"]

    ####################

    # Create inhomogeneous Poisson generator for input
    gen_poisson_in = nest.Create("inhomogeneous_poisson_generator", n_in)

    # The suffix _bsshslm_2020 follows the NEST convention to indicate in the model name the paper
    # that introduced it by the first letter of the authors' last names and the publication year.

    nrns_rec_exc = nest.Create("eprop_iaf_bsshslm_2020", n_rec_exc, params_nrn_rec)
    nrns_rec_inh = nest.Create("eprop_iaf_bsshslm_2020", n_rec_inh, params_nrn_rec)
    nrns_rec = nrns_rec_exc + nrns_rec_inh

    nrns_out = nest.Create("eprop_readout_bsshslm_2020", n_out, params_nrn_out)
    gen_rate_target = nest.Create("step_rate_generator", n_out)

    # %% ###########################################################################################################
    # Create recorders
    # ~~~~~~~~~~~~~~~~
    # We also create recorders, which, while not required for the training, will allow us to track various dynamic
    # variables of the neurons, spikes, and changes in synaptic weights. To save computing time and memory, the
    # recorders, the recorded variables, neurons, and synapses can be limited to the ones relevant to the
    # experiment, and the recording interval can be increased (see the documentation on the specific recorders). By
    # default, recordings are stored in memory but can also be written to file.

    # Helper to round down to nearest multiple of step
    def floor_to_step(val, step):
        # Ensure both val and step are floats for division, then cast to int after rounding
        return int(np.round(val / step) * step)

    # Recording parameters from config
    n_record = config["recording"]["n_record"]
    n_record_w = config["recording"]["n_record_w"]

    if n_record == 0 or n_record_w == 0:
        raise ValueError("n_record and n_record_w >= 1 required")

    params_mm_rec = config["recording"]["mm_rec"]
    params_mm_rec["interval"] = duration["step"]
    params_mm_rec["start"] = duration["offset_gen"] + duration["delay_in_rec"]
    params_mm_rec["stop"] = (
        duration["offset_gen"] + duration["delay_in_rec"] + duration["task"]
    )

    params_mm_out = config["recording"]["mm_out"]
    params_mm_out["interval"] = duration["step"]
    params_mm_out["start"] = duration["total_offset"]
    params_mm_out["stop"] = duration["total_offset"] + duration["task"]

    params_wr = {
        "senders": nrns_rec[
            :n_record_w
        ],  # limit senders to subsample weights to record
        "targets": nrns_rec[:n_record_w]
        + nrns_out,  # limit targets to subsample weights to record from
        "start": duration["total_offset"],
        "stop": duration["total_offset"] + duration["task"],
    }

    params_sr = {
        "start": duration["total_offset"],
        "stop": duration["total_offset"] + duration["task"],
    }

    ####################

    mm_rec = nest.Create("multimeter", params_mm_rec)
    mm_out = nest.Create("multimeter", params_mm_out)
    sr = nest.Create("spike_recorder", params_sr)
    wr = nest.Create("weight_recorder", params_wr)

    nrns_rec_record = nrns_rec[:n_record]

    # %% ###########################################################################################################
    # Create connections
    # ~~~~~~~~~~~~~~~~~~
    # Now, we define the connectivity and set up the synaptic parameters, with the synaptic weights drawn from
    # normal distributions. After these preparations, we establish the enumerated connections of the core network,
    # as well as additional connections to the recorders.

    params_conn_all_to_all = {"rule": "all_to_all", "allow_autapses": False}
    params_conn_one_to_one = {"rule": "one_to_one"}
    params_conn_bernoulli = {
        "rule": "pairwise_bernoulli",
        "p": 0.1,
        "allow_autapses": False,
    }

    # Synapse parameters from config
    w_default = config["synapses"]["w_default"]
    w_rec = config["synapses"]["w_rec"]
    g = config["synapses"]["g"]

    params_syn_eprop_exc = {
        "optimizer": {
            **config["synapses"]["exc"]["optimizer"],
            "batch_size": n_batch,
        },
        "average_gradient": config["synapses"]["average_gradient"],
        "weight_recorder": wr,
    }

    params_syn_eprop_inh = {
        "optimizer": {
            **config["synapses"]["inh"]["optimizer"],
            "batch_size": n_batch,
        },
        "weight": config["synapses"]["inh"]["weight"],
        "average_gradient": config["synapses"]["average_gradient"],
        "weight_recorder": wr,
    }

    params_syn_base = {
        "synapse_model": "eprop_synapse_bsshslm_2020",
        "delay": duration["step"],  # ms, dendritic delay
        "tau_m_readout": params_nrn_out[
            "tau_m"
        ],  # ms, for technical reasons pass readout neuron membrane time constant
    }

    params_syn_input = {
        "synapse_model": "static_synapse",
        "delay": duration["step"],
        "weight": nest.math.redraw(
            nest.random.normal(mean=w_default, std=w_default * 0.1), min=0.0, max=1000.0
        ),
    }

    # Define the parameters for the recurrent connections
    params_syn_rec_exc = copy.deepcopy(params_syn_base)
    params_syn_rec_exc["weight"] = nest.math.redraw(
        nest.random.normal(mean=w_rec, std=w_rec * 0.1), min=0.0, max=1000.0
    )
    params_syn_rec_exc["synapse_model"] = "eprop_synapse_bsshslm_2020_exc"

    params_syn_rec_inh = copy.deepcopy(params_syn_base)
    params_syn_rec_inh["weight"] = nest.math.redraw(
        nest.random.normal(mean=-w_rec * g, std=g * w_rec * 0.1), min=-1000.0, max=0.0
    )
    params_syn_rec_inh["synapse_model"] = "eprop_synapse_bsshslm_2020_inh"

    # Define the parameters for the feedback connections from readout neurons to recurrent neurons
    params_syn_feedback = {
        "synapse_model": "eprop_learning_signal_connection_bsshslm_2020",
        "delay": duration["step"],
        "weight": nest.math.redraw(
            nest.random.normal(mean=w_rec, std=w_rec * 0.1), min=0.0, max=1000.0
        ),
    }

    # Define the parameters for the target signal connections to readout neurons
    params_syn_rate_target = {
        "synapse_model": "rate_connection_delayed",
        "delay": duration["step"],
        "receptor_type": 2,  # receptor type over which readout neurons receive target signals
    }

    # Define the parameters for the static synapses used for recording
    params_syn_static = {
        "synapse_model": "static_synapse",
        "delay": duration["step"],
    }

    ####################

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

    # Connect each Poisson generator to a proportion of the excitatory and inhibitory populations
    if plastic_input_to_rec:
        # Fully connect all input neurons to all recurrent neurons with plastic synapses
        nest.Connect(
            gen_poisson_in,
            nrns_rec,
            params_conn_all_to_all,
            params_syn_rec_exc,
        )
    else:
        for i, poisson_node in enumerate(gen_poisson_in):
            # Connect to a proportion of excitatory neurons
            nest.Connect(
                poisson_node,
                nrns_rec_exc[
                    int(i * n_rec_exc / n_in) : int((i + 1) * n_rec_exc / n_in)
                ],
                params_conn_all_to_all,
                params_syn_input,
            )
            # Connect to a proportion of inhibitory neurons
            nest.Connect(
                poisson_node,
                nrns_rec_inh[
                    int(i * n_rec_inh / n_in) : int((i + 1) * n_rec_inh / n_in)
                ],
                params_conn_all_to_all,
                params_syn_input,
            )

    # Connect recurrent neurons to themselves
    nest.Connect(
        nrns_rec_exc, nrns_rec, params_conn_bernoulli, params_syn_rec_exc
    )  # Excitory to all
    nest.Connect(
        nrns_rec_inh, nrns_rec, params_conn_bernoulli, params_syn_rec_inh
    )  # Inhibitory to all

    # Connect recurrent neurons to readout neurons
    nest.Connect(
        nrns_rec_exc, nrns_out, params_conn_all_to_all, params_syn_rec_exc
    )  # Excitory to all readout neurons

    # Connect readout neurons to recurrent neurons
    nest.Connect(
        nrns_out, nrns_rec, params_conn_all_to_all, params_syn_feedback
    )  # Readout to all neurons

    # Connect readout neurons to target signal generator
    nest.Connect(
        gen_rate_target[0], nrns_out[0], params_conn_one_to_one, params_syn_rate_target
    )  # Readout 1 to target signal generator
    nest.Connect(
        gen_rate_target[1], nrns_out[1], params_conn_one_to_one, params_syn_rate_target
    )  # Readout 2 to target signal generator

    # Connect recorders to neurons
    nest.Connect(nrns_rec, sr, params_conn_all_to_all, params_syn_static)
    nest.Connect(mm_rec, nrns_rec_record, params_conn_all_to_all, params_syn_static)
    nest.Connect(mm_out, nrns_out, params_conn_all_to_all, params_syn_static)

    # %% ###########################################################################################################
    # Load training dataset
    # ~~~~~~~~~~~~~~~~~~~~~
    # We load the trajectory data from a file. The trajectory data is a one-dimensional signal that the network
    # should learn to reproduce. We resample the trajectory data to match the new resolution.
    # Load the training dataset and extract the trajectory number
    dataset_path = (
        Path(__file__).resolve().parent.parent
        / "dataset_motor_training"
        / "dataset_spikes.gdf"
    )
    training_dataset = load_data_file(str(dataset_path))

    # In the dataset, each batch contains 10 samples, so if we select 2 batches and 5 samples per batch,
    # we get 10 samples in total.
    # Create a list of indices for the selected batches and samples
    samples_per_batch = 10  # Number of samples per batch
    sample_ids = []

    for batch in range(n_batch):
        start_index = (
            batch * samples_per_batch
        )  # Calculate the starting index of the batch
        sample_ids.extend(range(start_index, start_index + n_samples))

    trajectories = []
    trajectory_original_resolution = (
        0.1  # Original resolution of the trajectory data in ms
    )
    desired_targets_list = {"pos": [], "neg": []}

    # Iterate over the sample IDs and load the corresponding data
    for sample_id in sample_ids:
        trajectory_num = int(training_dataset[sample_id][0][0])
        id_pos = training_dataset[sample_id][1]
        time_pos = training_dataset[sample_id][2]
        id_neg = training_dataset[sample_id][3]
        time_neg = training_dataset[sample_id][4]

        # Load the trajectory data used as input for the network
        trajectory_file = dataset_path.parent / f"trajectory{trajectory_num}.txt"
        trajectory_data = np.loadtxt(trajectory_file)  # load the trajectory data

        # Resample trajectory data to match the number of time steps per sequence
        # Calculate the number of points to sample
        original_num_points = len(trajectory_data)
        original_duration = original_num_points * trajectory_original_resolution
        # Generate new time points for resampling
        resampled_time = np.linspace(
            0, original_duration, n_timesteps_per_sequence, endpoint=False
        )
        original_time = np.arange(original_num_points) * trajectory_original_resolution
        # Interpolate to get resampled trajectory
        trajectory_data_resampled = np.interp(
            resampled_time, original_time, trajectory_data
        )

        # Store the trajectory data
        trajectories.append(trajectory_data_resampled)

        # Counts the number of output spikes per time step (bin edges must match n_timesteps_per_sequence)
        desired_targets = {
            "pos": np.histogram(
                time_pos,
                bins=n_timesteps_per_sequence,
                range=(0, duration["sequence"]),
            )[0],
            "neg": np.histogram(
                time_neg,
                bins=n_timesteps_per_sequence,
                range=(0, duration["sequence"]),
            )[0],
        }

        # Smooth the target signals
        desired_targets["pos"] = np.convolve(
            desired_targets["pos"], np.ones(20) / 10, mode="same"
        )
        desired_targets["neg"] = np.convolve(
            desired_targets["neg"], np.ones(20) / 10, mode="same"
        )

        # Store the desired targets
        desired_targets_list["pos"].append(desired_targets["pos"])
        desired_targets_list["neg"].append(desired_targets["neg"])

    # Get the length of the trajectory sample for later use
    trajectory_sample = trajectories[0]
    length_trajectory_sample = len(trajectory_sample)
    # Ensure this matches n_timesteps_per_sequence
    assert length_trajectory_sample == n_timesteps_per_sequence

    # %% ###########################################################################################################
    # Create input
    # ~~~~~~~~~~~~

    # Radial basis function (RBF) parameters for encoding the trajectory data
    def gaussian_rbf(x, center, width):
        return np.exp(-((x - center) ** 2) / (2 * width**2))

    # Prepare RBF inputs for all samples (trajectories)
    rbf_inputs_list = []
    for trajectory_sample in trajectories:
        rbf_inputs = np.zeros((n_timesteps_per_sequence, num_centers))
        for i, center in enumerate(centers):
            rbf_inputs[:, i] = gaussian_rbf(trajectory_sample, center, width)
        rbf_inputs_list.append(rbf_inputs)

    # Stack all RBF inputs: shape = (n_batch * n_samples * n_timesteps_per_sequence, num_centers)
    rate_based_rbf_inputs = np.vstack(rbf_inputs_list) * scale_rate

    # Calculate total number of input time steps
    total_input_steps = n_timesteps_per_sequence * n_batch * n_samples * n_iter

    # Time vector for input rates
    in_rate_times = (
        np.arange(total_input_steps) * duration["step"] + duration["offset_gen"]
    )

    # Tile the RBF input for all iterations
    tiled_rbf_inputs = np.tile(
        rate_based_rbf_inputs, (n_iter, 1)
    )  # shape: (total_input_steps, num_centers)

    params_gen_poisson_in = [
        {
            "rate_times": in_rate_times,
            "rate_values": tiled_rbf_inputs[:, n_center],
        }
        for n_center in range(num_centers)
    ]

    nest.SetStatus(gen_poisson_in, params_gen_poisson_in)

    # %% ###########################################################################################################
    # Create output
    # ~~~~~~~~~~~~~

    # Concatenate the desired targets in desired_targets_list
    concatenated_desired_targets = {
        "pos": np.concatenate(desired_targets_list["pos"]),
        "neg": np.concatenate(desired_targets_list["neg"]),
    }

    # The length of the target sample should match n_timesteps_per_sequence * n_batch * n_samples
    length_target_sample = len(concatenated_desired_targets["pos"])
    assert length_target_sample == n_timesteps_per_sequence * n_batch * n_samples

    target_amp_times = (
        np.arange(length_target_sample * n_iter) * duration["step"]
        + duration["offset_gen"]
    )

    params_gen_rate_target = [
        {
            "amplitude_times": target_amp_times,
            "amplitude_values": np.tile(
                concatenated_desired_targets[key] * 1e1, n_iter
            ),
        }
        for key in concatenated_desired_targets.keys()
    ]

    ####################

    nest.SetStatus(gen_rate_target, params_gen_rate_target)

    # %% ###########################################################################################################
    # Force final update
    # ~~~~~~~~~~~~~~~~~~
    # Synapses only get active, that is, the correct weight update calculated and applied, when they transmit a
    # spike. To still be able to read out the correct weights at the end of the simulation, we force spiking of the
    # presynaptic neuron and thus an update of all synapses, including those that have not transmitted a spike in
    # the last update interval, by sending a strong spike to all neurons that form the presynaptic side of an eprop
    # synapse. This step is required purely for technical reasons.

    gen_spk_final_update = nest.Create(
        "spike_generator", 1, {"spike_times": [duration["task"] + duration["delays"]]}
    )

    nest.Connect(gen_spk_final_update, nrns_rec, "all_to_all", {"weight": 1000.0})

    # %% ###########################################################################################################
    # Read out pre-training weights
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Before we begin training, we read out the initial weight matrices so that we can eventually compare them to
    # the optimized weights.

    def get_weights(pop_pre, pop_post):
        conns = nest.GetConnections(pop_pre, pop_post).get(
            ["source", "target", "weight"]
        )
        conns["senders"] = np.array(conns["source"]) - np.min(conns["source"])
        conns["targets"] = np.array(conns["target"]) - np.min(conns["target"])

        conns["weight_matrix"] = np.zeros((len(pop_post), len(pop_pre)))
        conns["weight_matrix"][conns["targets"], conns["senders"]] = conns["weight"]
        return conns

    weights_pre_train = {
        "in_rec": get_weights(nrns_rec, nrns_rec),
        "rec_rec": get_weights(nrns_rec, nrns_rec),
        "rec_out": get_weights(nrns_rec, nrns_out),
    }

    # %% ###########################################################################################################
    # Simulate
    # ~~~~~~~~
    # We train the network by simulating for a set simulation time, determined by the number of iterations and the
    # batch size and the length of one sequence.

    nest.Simulate(duration["sim"])

    # %% ###########################################################################################################
    # Read out post-training weights
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # After the training, we can read out the optimized final weights.

    weights_post_train = {
        "in_rec": get_weights(nrns_rec, nrns_rec),
        "rec_rec": get_weights(nrns_rec, nrns_rec),
        "rec_out": get_weights(nrns_rec, nrns_out),
    }

    # %% ###########################################################################################################
    # Read out recorders
    # ~~~~~~~~~~~~~~~~~~
    # We can also retrieve the recorded history of the dynamic variables and weights, as well as detected spikes.

    events_mm_rec = mm_rec.get("events")
    events_mm_out = mm_out.get("events")
    events_sr = sr.get("events")
    events_wr = wr.get("events")

    # %% ###########################################################################################################
    # Evaluate training error
    # ~~~~~~~~~~~~~~~~~~~~~~~
    # We evaluate the network's training error by calculating a loss - in this case, the mean squared error between
    # the integrated recurrent network activity and the target rate.

    readout_signal = events_mm_out["readout_signal"]
    target_signal = events_mm_out["target_signal"]

    error = (readout_signal - target_signal) ** 2
    loss = 0.5 * np.add.reduceat(
        error, np.arange(0, int(duration["task"]), int(duration["sequence"]))
    )

    # %% ###########################################################################################################
    # Plot results
    # ~~~~~~~~~~~~
    # Then, we plot a series of plots.

    # Plotting flag from config
    plotting_cfg = config.get("plotting", {})
    do_plotting = plotting_cfg.get("do_plotting", True)

    if plot_results and do_plotting:
        # Use scenario-specific directory for output files
        out_dir = result_dir if result_dir else "."
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

        # Save loss, full config, weights, and signals for later comparison
        if result_dir:
            import json

            def make_json_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(v) for v in obj]
                elif isinstance(obj, Path):
                    return str(obj)
                else:
                    return obj

            config_serializable = make_json_serializable(config)
            np.savez(
                os.path.join(out_dir, "results.npz"),
                loss=loss,
                weights_pre_train=make_json_serializable(weights_pre_train),
                weights_post_train=make_json_serializable(weights_post_train),
                readout_signal=readout_signal,
                target_signal=target_signal,
            )
            with open(os.path.join(out_dir, "config.json"), "w") as f:
                json.dump(config_serializable, f, indent=2)


# Define a default color cycle for plotting
colors = {
    "blue": "#1f77b4",
    "red": "#d62728",
    "white": "#ffffff",
}

# Define results directory for saving outputs
results_dir = os.path.join(os.path.dirname(__file__), "sim_results")
os.makedirs(results_dir, exist_ok=True)

# ---
# Usage Examples:
#
# 1. Set both excitatory and inhibitory learning rates simultaneously:
#    python eprop-reaching-task.py --learning-rate 0.001
#
# 2. Scan another parameter while setting both learning rates:
#    python eprop-reaching-task.py --scan-param neurons.n_rec --scan-values 100,200 --learning-rate 0.001
#
# 3. Scan learning rates, number of RBF centers, and number of neurons together (multi-parameter grid search):
#    python eprop-reaching-task.py --scan-param learning_rate_exc,learning_rate_inh,rbf.num_centers,neurons.n_rec \
#        --scan-values "0.001,0.01;0.001,0.01;10,20;100,200"
#    # Note: Use quotes around --scan-values to avoid shell parsing issues with semicolons.
#
# 4. Use --learning-rate to set both rates and scan RBF centers and neurons:
#    python eprop-reaching-task.py --scan-param rbf.num_centers,neurons.n_rec \
#        --scan-values "10,20;100,200" --learning_rate 0.001
#
# 5. Scan a single parameter:
#    python eprop-reaching-task.py --scan-param learning_rate_exc --scan-values 0.001,0.01,0.1
#
# 6. Scan multiple parameters (grid search):
#    python eprop-reaching-task.py --scan-param learning_rate_exc,neurons.n_rec --scan-values "0.001,0.01;100,200"
#    # This will run all combinations of the provided values.
#
# 7. Make input-to-recurrent connections plastic and fully connected:
#    python eprop-reaching-task.py --plastic-input-to-rec
#
# - Separate parameter names with commas, and value lists with semicolons.
# - All parameter names must match the config structure (e.g., neurons.n_rec, rbf.num_centers).
# - Always quote --scan-values if using semicolons to avoid shell parsing errors.
# ---

if __name__ == "__main__":
    # Parse command-line arguments for parameter scan and plotting options
    import argparse

    parser = argparse.ArgumentParser(
        description="Run e-prop reaching task parameter scan."
    )
    parser.add_argument(
        "--scan-param",
        type=str,
        default=None,
        help="Comma-separated list of parameters to scan (dot notation for nested keys)",
    )
    parser.add_argument(
        "--scan-values",
        type=str,
        default=None,
        help="Semicolon-separated lists of comma-separated values for each parameter (e.g. 0.1,0.2;10,20)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Set both exc and inh learning rates simultaneously",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument(
        "--plastic-input-to-rec",
        action="store_true",
        help="Make input-to-recurrent connections plastic and fully connected",
    )
    args = parser.parse_args()

    # Prepare parameter names and values for grid search
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
    plot_results_flag = not args.no_plot

    scenarios = []
    # Run parameter scan if parameters and values are provided
    if scan_params and scan_values and len(scan_params) == len(scan_values):
        for combo in itertools.product(*scan_values):
            param_dict = dict(zip(scan_params, combo))
            # If --learning-rate is set, override both exc and inh learning rates
            if args.learning_rate is not None:
                param_dict["learning_rate_exc"] = args.learning_rate
                param_dict["learning_rate_inh"] = args.learning_rate
            folder_name = "_".join(
                f"{k.replace('.', '_')}_{v}" for k, v in param_dict.items()
            )
            sim_dir = os.path.join(results_dir, folder_name)
            os.makedirs(sim_dir, exist_ok=True)
            print(f"Running scenario: {param_dict}")
            run_simulation(
                **param_dict,
                result_dir=sim_dir,
                plot_results=plot_results_flag,
                plastic_input_to_rec=args.plastic_input_to_rec,
            )
            scenarios.append(folder_name)
    else:
        # Run a single default scenario if no parameter scan is specified
        param_dict = {}
        if args.learning_rate is not None:
            param_dict["learning_rate_exc"] = args.learning_rate
            param_dict["learning_rate_inh"] = args.learning_rate
        sim_dir = os.path.join(results_dir, "default")
        os.makedirs(sim_dir, exist_ok=True)
        run_simulation(
            result_dir=sim_dir,
            plot_results=plot_results_flag,
            plastic_input_to_rec=args.plastic_input_to_rec,
            **param_dict,
        )
        scenarios.append("default")

    # Plot and compare all loss curves from the results directory
    plot_all_loss_curves(results_dir, showfig=False)
