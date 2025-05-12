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

import matplotlib as mpl
import matplotlib.pyplot as plt
import nest
import copy
import numpy as np
from cycler import cycler
from IPython.display import Image
import yaml

# Import the function to load the dataset
import sys
from pathlib import Path

# Add the parent directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent / "dataset_motor_training"))
from load_dataset import load_data_file

# Load parameters from YAML config file
config_path = Path(__file__).resolve().parent / "config" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

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
duration = {
    "step": step_ms,
    "sequence": task_cfg["sequence"] * step_ms,
    "learning_window": task_cfg["sequence"] * step_ms,
    "task": n_iter * n_batch * task_cfg["sequence"] * n_samples * step_ms,
    "offset_gen": task_cfg["offset_gen"] * step_ms,
    "delay_in_rec": task_cfg["delay_in_rec"] * step_ms,
    "delay_rec_out": task_cfg["delay_rec_out"] * step_ms,
    "delay_out_norm": task_cfg["delay_out_norm"] * step_ms,
    "extension_sim": task_cfg["extension_sim"] * step_ms,
}
duration["delays"] = duration["delay_in_rec"] + duration["delay_rec_out"] + duration["delay_out_norm"]
duration["total_offset"] = duration["offset_gen"] + duration["delays"]
duration["sim"] = duration["task"] + duration["total_offset"] + duration["extension_sim"]

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
num_centers = config["rbf"]["num_centers"]
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
n_rec = config["neurons"]["n_rec"]
n_out = config["neurons"]["n_out"]
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

# Recording parameters from config
n_record = config["recording"]["n_record"]
n_record_w = config["recording"]["n_record_w"]

if n_record == 0 or n_record_w == 0:
    raise ValueError("n_record and n_record_w >= 1 required")

params_mm_rec = config["recording"]["mm_rec"]
params_mm_rec["interval"] = duration["step"]
params_mm_rec["start"] = duration["offset_gen"] + duration["delay_in_rec"]
params_mm_rec["stop"] = duration["offset_gen"] + duration["delay_in_rec"] + duration["task"]

params_mm_out = config["recording"]["mm_out"]
params_mm_out["interval"] = duration["step"]
params_mm_out["start"] = duration["total_offset"]
params_mm_out["stop"] = duration["total_offset"] + duration["task"]

params_wr = {
    "senders": nrns_rec[:n_record_w],  # limit senders to subsample weights to record
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

params_common_syn_eprop = {
    "optimizer": config["synapses"]["exc"]["optimizer"],
    "average_gradient": config["synapses"]["average_gradient"],
    "weight_recorder": wr,
}

params_syn_eprop_exc = {
    "optimizer": config["synapses"]["exc"]["optimizer"],
    "average_gradient": config["synapses"]["average_gradient"],
    "weight_recorder": wr,
}

params_syn_eprop_inh = {
    "optimizer": config["synapses"]["inh"]["optimizer"],
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
    "receptor_type": 2,  # receptor type over which readout neurons receive target signals
}

params_syn_static = {
    "synapse_model": "static_synapse",
    "delay": duration["step"],
}

####################

nest.SetDefaults("eprop_synapse_bsshslm_2020", params_common_syn_eprop)
nest.CopyModel(
    "eprop_synapse_bsshslm_2020", "eprop_synapse_bsshslm_2020_exc", params_syn_eprop_exc
)
nest.CopyModel(
    "eprop_synapse_bsshslm_2020", "eprop_synapse_bsshslm_2020_inh", params_syn_eprop_inh
)

# Connect each Poisson generator to a proportion of the excitatory and inhibitory populations
for i, poisson_node in enumerate(gen_poisson_in):
    # Connect to a proportion of excitatory neurons
    nest.Connect(
        poisson_node,
        nrns_rec_exc[int(i * n_rec_exc / n_in) : int((i + 1) * n_rec_exc / n_in)],
        params_conn_all_to_all,
        params_syn_input,
    )
    # Connect to a proportion of inhibitory neurons
    nest.Connect(
        poisson_node,
        nrns_rec_inh[int(i * n_rec_inh / n_in) : int((i + 1) * n_rec_inh / n_in)],
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

sample_ids = list(range(n_samples))  # indices of the samples to load
trajectories = []
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

    # Resample trajectory data to match the code resolution of 1.0 ms
    trajectory_data = trajectory_data[::10]

    # Store the trajectory data
    trajectories.append(trajectory_data)

    # Counts the number of output spikes per time step
    desired_targets = {
        "pos": np.histogram(
            time_pos, bins=int(duration["sequence"]), range=(0, duration["sequence"])
        )[0],
        "neg": np.histogram(
            time_neg, bins=int(duration["sequence"]), range=(0, duration["sequence"])
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

# %% ###########################################################################################################
# Create input
# ~~~~~~~~~~~~

# Radial basis function (RBF) parameters for encoding the trajectory data
def gaussian_rbf(x, center, width):
    return np.exp(-((x - center) ** 2) / (2 * width**2))

rbf_inputs_list = []
for trajectory_sample in trajectories:
    rbf_inputs = np.zeros((len(trajectory_sample), num_centers))
    for i, center in enumerate(centers):
        rbf_inputs[:, i] = gaussian_rbf(trajectory_sample, center, width)
    rbf_inputs_list.append(rbf_inputs)

rate_based_rbf_inputs = np.vstack(rbf_inputs_list) * scale_rate  # Combine all trajectories

# Create Poisson generator parameters
params_gen_poisson_in = [
    {
        "rate_times": np.arange(0.0, duration["task"], duration["step"])
        + duration["offset_gen"],
        "rate_values": np.tile(rate_based_rbf_inputs[:, n_center], n_iter * n_batch),
    }
    for n_center in range(num_centers)
]

# Assign the parameters to the Poisson generators
nest.SetStatus(gen_poisson_in, params_gen_poisson_in)

# %% ###########################################################################################################
# Create output
# ~~~~~~~~~~~~~

# The desired target signal is the output data sent by the collaborators (Alberto, Claudia)

# Concatenate the desired targets in desired_targets_list
concatenated_desired_targets = {
    "pos": np.concatenate(desired_targets_list["pos"]),
    "neg": np.concatenate(desired_targets_list["neg"]),
}

params_gen_rate_target = []

params_gen_rate_target = [
    {
        "amplitude_times": np.arange(0.0, duration["task"], duration["step"])
        + duration["total_offset"],
        "amplitude_values": np.tile(concatenated_desired_targets[key] * 1e1, n_iter * n_batch),
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
    conns = nest.GetConnections(pop_pre, pop_post).get(["source", "target", "weight"])
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
loss = 0.5 * np.add.reduceat(error, np.arange(0, int(duration["task"]), int(duration["sequence"])))

# %% ###########################################################################################################
# Plot results
# ~~~~~~~~~~~~
# Then, we plot a series of plots.

# Plotting flag from config
plotting_cfg = config.get("plotting", {})
do_plotting = plotting_cfg.get("do_plotting", True)

if not do_plotting:
    exit()

colors = {
    "blue": "#2854c5ff",
    "red": "#e04b40ff",
    "white": "#ffffffff",
}

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.prop_cycle": cycler(color=[colors["blue"], colors["red"]]),
    }
)

# %% ###########################################################################################################
# Plot training error
# ...................
# We begin with a plot visualizing the training error of the network: the loss plotted against the iterations.

fig, ax = plt.subplots()

ax.plot(range(1, n_samples*n_iter + 1), loss)
ax.set_ylabel(r"$E = \frac{1}{2} \sum_{t,k} \left( y_k^t -y_k^{*,t}\right)^2$")
ax.set_xlabel("training iteration")
ax.set_xlim(1, n_samples*n_iter)
ax.xaxis.get_major_locator().set_params(integer=True)

fig.tight_layout()
fig.savefig("training_error.png")  # Save the figure

# %% ###########################################################################################################
# Plot spikes and dynamic variables
# .................................
# This plotting routine shows how to plot all of the recorded dynamic variables and spikes across time. We take
# one snapshot in the first iteration and one snapshot at the end.


def plot_recordable(ax, events, recordable, ylabel, xlims):
    for sender in set(events["senders"]):
        idc_sender = events["senders"] == sender
        idc_times = (events["times"][idc_sender] > xlims[0]) & (
            events["times"][idc_sender] < xlims[1]
        )
        ax.plot(
            events["times"][idc_sender][idc_times],
            events[recordable][idc_sender][idc_times],
            lw=0.5,
        )
    ax.set_ylabel(ylabel)
    margin = np.abs(np.max(events[recordable]) - np.min(events[recordable])) * 0.1
    # ax.set_ylim(np.min(events[recordable]) - margin, np.max(events[recordable]) + margin)


def plot_spikes(ax, events, nrns, ylabel, xlims):
    idc_times = (events["times"] > xlims[0]) & (events["times"] < xlims[1])
    idc_sender = np.isin(events["senders"][idc_times], nrns.tolist())
    senders_subset = events["senders"][idc_times][idc_sender]
    times_subset = events["times"][idc_times][idc_sender]

    ax.scatter(times_subset, senders_subset, s=0.1)
    ax.set_ylabel(ylabel)
    if senders_subset.size > 0:
        margin = np.abs(np.max(senders_subset) - np.min(senders_subset)) * 0.1
        ax.set_ylim(np.min(senders_subset) - margin, np.max(senders_subset) + margin)
    else:
        ax.set_ylim(0, 1)


for xlims in [
    (0, duration["sequence"] * n_samples),
    (duration["task"] - duration["sequence"] * n_samples, duration["task"]),
]:
    fig, axs = plt.subplots(8, 1, sharex=True, figsize=(4, 12))

    plot_spikes(axs[0], events_sr, nrns_rec, r"$z_j$" + "\n", xlims)

    plot_recordable(axs[1], events_mm_rec, "V_m", r"$v_j$" + "\n(mV)", xlims)
    plot_recordable(
        axs[2], events_mm_rec, "surrogate_gradient", r"$\psi_j$" + "\n", xlims
    )
    plot_recordable(
        axs[3], events_mm_rec, "learning_signal", r"$L_j$" + "\n(pA)", xlims
    )

    plot_recordable(axs[4], events_mm_out, "V_m", r"$v_k$" + "\n(mV)", xlims)
    plot_recordable(axs[5], events_mm_out, "target_signal", r"$y^*_k$" + "\n", xlims)
    plot_recordable(axs[6], events_mm_out, "readout_signal", r"$y_k$" + "\n", xlims)
    plot_recordable(axs[7], events_mm_out, "error_signal", r"$y_k-y^*_k$" + "\n", xlims)

    axs[-1].set_xlabel(r"$t$ (ms)")
    axs[-1].set_xlim(*xlims)

    for ax in axs:
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
        for line in ax.get_lines():
            line.set_linewidth(1.5)  # Increase the linewidth

    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(
        f"spikes_and_dynamic_variables_{xlims[0]}_{xlims[1]}.png", dpi=300
    )  # Save the figure

# %% ###########################################################################################################
# Plot weight time courses
# ........................
# Similarly, we can plot the weight histories. Note that the weight recorder, attached to the synapses, works
# differently than the other recorders. Since synapses only get activated when they transmit a spike, the weight
# recorder only records the weight in those moments. That is why the first weight registrations do not start in
# the first time step and we add the initial weights manually.


def plot_weight_time_course(ax, events, nrns_senders, nrns_targets, label, ylabel):
    for sender in nrns_senders.tolist():
        for target in nrns_targets.tolist():
            idc_syn = (events["senders"] == sender) & (events["targets"] == target)
            idc_syn_pre = (weights_pre_train[label]["source"] == sender) & (
                weights_pre_train[label]["target"] == target
            )

            times = [0.0] + events["times"][idc_syn].tolist()
            weights = [weights_pre_train[label]["weight"][idc_syn_pre]] + events[
                "weights"
            ][idc_syn].tolist()

            ax.step(times, weights, c=colors["blue"])
        ax.set_ylabel(ylabel)


fig, axs = plt.subplots(2, 1, sharex=True, figsize=(3, 4))

plot_weight_time_course(
    axs[0],
    events_wr,
    nrns_rec[:n_record_w],
    nrns_rec[:n_record_w],
    "rec_rec",
    r"$W_\text{rec}$ (pA)",
)
plot_weight_time_course(
    axs[1],
    events_wr,
    nrns_rec[:n_record_w],
    nrns_out,
    "rec_out",
    r"$W_\text{out}$ (pA)",
)

axs[-1].set_xlabel(r"$t$ (ms)")
axs[-1].set_xlim(0, duration["task"])

fig.align_ylabels()
fig.tight_layout()
fig.savefig("weight_time_courses.png")  # Save the figure

# %% ###########################################################################################################
# Plot weight matrices
# ....................
# If one is not interested in the time course of the weights, it is possible to read out only the initial and
# final weights, which requires less computing time and memory than the weight recorder approach. Here, we plot
# the corresponding weight matrices before and after the optimization.

cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "cmap", ((0.0, colors["blue"]), (0.5, colors["white"]), (1.0, colors["red"]))
)

fig, axs = plt.subplots(2, 2, sharex="col", sharey="row")

all_w_extrema = []

for k in weights_pre_train.keys():
    w_pre = weights_pre_train[k]["weight"]
    w_post = weights_post_train[k]["weight"]
    all_w_extrema.append([np.min(w_pre), np.max(w_pre), np.min(w_post), np.max(w_post)])

# Center the color map at zero
vmin = np.min(all_w_extrema)
vmax = np.max(all_w_extrema)
norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

args = {"cmap": cmap, "norm": norm}

for i, weights in zip([0, 1], [weights_pre_train, weights_post_train]):
    axs[0, i].pcolormesh(weights["rec_rec"]["weight_matrix"], **args)
    cmesh = axs[1, i].pcolormesh(weights["rec_out"]["weight_matrix"], **args)

    axs[1, i].set_xlabel("recurrent\nneurons")

axs[0, 0].set_ylabel("recurrent\nneurons")
axs[1, 0].set_ylabel("readout\nneurons")
fig.align_ylabels(axs[:, 0])

axs[0, 0].text(0.5, 1.1, "pre-training", transform=axs[0, 0].transAxes, ha="center")
axs[0, 1].text(0.5, 1.1, "post-training", transform=axs[0, 1].transAxes, ha="center")

axs[1, 0].yaxis.get_major_locator().set_params(integer=True)

cbar = plt.colorbar(
    cmesh, cax=axs[1, 1].inset_axes([1.1, 0.2, 0.05, 0.8]), label="weight (pA)"
)

fig.tight_layout()
fig.savefig("weight_matrices.png")  # Save the figure

# plt.show()
