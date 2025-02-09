# -*- coding: utf-8 -*-
#
# eprop_supervised_classification_evidence-accumulation.py
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
Tutorial on learning to perform a 2D reaching task with e-prop
-------------------------------------------------------------

Training a model using supervised e-prop plasticity to perform a 2D reaching task.

Description
~~~~~~~~~~~

This script demonstrates supervised learning of a 2D reaching task with the eligibility propagation (e-prop)
plasticity mechanism by Bellec et al. [1]_.

The task involves controlling a simulated arm to reach a target position in a 2D plane. The arm receives
input signals that represent the target position and must learn to generate appropriate motor commands to
reach the target.

Learning in the neural network model is achieved by optimizing the connection weights with e-prop plasticity.
The network architecture consists of input neurons representing the target position, recurrent neurons for
processing, and output neurons representing the motor commands. The training error is assessed by the
difference between the actual and target positions.

References
~~~~~~~~~~

.. [1] Bellec G, Scherr F, Subramoney F, Hajek E, Salaj D, Legenstein R, Maass W (2020). A solution to the
       learning dilemma for recurrent networks of spiking neurons. Nature Communications, 11:3625.
       https://doi.org/10.1038/s41467-020-17236-y
"""  # pylint: disable=line-too-long # noqa: E501

# %% ###########################################################################################################
# Import libraries
# ~~~~~~~~~~~~~~~~
# We begin by importing all libraries required for the simulation, analysis, and visualization.

import matplotlib as mpl
import matplotlib.pyplot as plt
import nest
import numpy as np
from cycler import cycler
from IPython.display import Image

# %% ###########################################################################################################
# Schematic of network architecture
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This figure, identical to the one in the description, shows the required network architecture in the center,
# the input and output of the pattern generation task above, and lists of the required NEST device, neuron, and
# synapse models below. The connections that must be established are numbered 1 to 7.

# %% ###########################################################################################################
# Setup
# ~~~~~

# %% ###########################################################################################################
# Initialize random generator
# ...........................
# We seed the numpy random generator, which will generate random initial weights as well as random input and
# output.

rng_seed = 1  # numpy random seed
np.random.seed(rng_seed)  # fix numpy random seed

# %% ###########################################################################################################
# Define timing of task
# .....................
# The task's temporal structure is defined, with time steps and durations in milliseconds.

n_batch = 1  # batch size
n_iter = 50  # number of iterations

n_input_symbols = 2  # number of input populations (x and y coordinates)
steps = {
    "target": 100,  # time steps for target presentation
    "movement": 200,  # time steps for movement
}

steps["sequence"] = steps["target"] + steps["movement"]  # time steps of one full sequence
steps["task"] = n_iter * n_batch * steps["sequence"]  # time steps of task

steps.update(
    {
        "offset_gen": 1,  # offset since generator signals start from time step 1
        "delay_in_rec": 1,  # connection delay between input and recurrent neurons
        "delay_rec_out": 1,  # connection delay between recurrent and output neurons
        "delay_out_norm": 1,  # connection delay between output neurons for normalization
        "extension_sim": 1,  # extra time step to close right-open simulation time interval in Simulate()
    }
)

steps["delays"] = steps["delay_in_rec"] + steps["delay_rec_out"] + steps["delay_out_norm"]  # time steps of delays
steps["total_offset"] = steps["offset_gen"] + steps["delays"]  # time steps of total offset
steps["sim"] = steps["task"] + steps["total_offset"] + steps["extension_sim"]  # time steps of simulation

duration = {"step": 1.0}  # ms, temporal resolution of the simulation
duration.update({key: value * duration["step"] for key, value in steps.items()})  # ms, durations

# %% ###########################################################################################################
# Set up simulation
# .................
# As last step of the setup, we reset the NEST kernel to remove all existing NEST simulation settings and
# objects and set some NEST kernel parameters, some of which are e-prop-related.

params_setup = {
    "eprop_learning_window": duration["movement"],
    "eprop_reset_neurons_on_update": True,  # if True, reset dynamic variables at start of each update interval
    "eprop_update_interval": duration["sequence"],  # ms, time interval for updating the synaptic weights
    "print_time": False,  # if True, print time progress bar during simulation, set False if run as code cell
    "resolution": duration["step"],
    "total_num_virtual_procs": 1,  # number of virtual processes, set in case of distributed computing
}

####################

nest.ResetKernel()
nest.set(**params_setup)

# %% ###########################################################################################################
# Create neurons
# ~~~~~~~~~~~~~~
# We proceed by creating a certain number of input, recurrent, and readout neurons and setting their parameters.
# Additionally, we already create an input spike generator and an output target rate generator, which we will
# configure later. Within the recurrent network, alongside a population of regular neurons, we introduce a
# population of adaptive neurons, to enhance the network's memory retention.

n_in = 2  # number of input neurons (x and y coordinates)
n_ad = 50  # number of adaptive neurons
n_reg = 50  # number of regular neurons
n_rec = n_ad + n_reg  # number of recurrent neurons
n_out = 2  # number of readout neurons

params_nrn_reg = {
    "C_m": 1.0,  # pF, membrane capacitance - takes effect only if neurons get current input (here not the case)
    "c_reg": 2.0,  # firing rate regularization scaling - double the TF c_reg for technical reasons
    "E_L": 0.0,  # mV, leak / resting membrane potential
    "f_target": 10.0,  # spikes/s, target firing rate for firing rate regularization
    "gamma": 0.3,  # scaling of the pseudo derivative
    "I_e": 0.0,  # pA, external current input
    "regular_spike_arrival": True,  # If True, input spikes arrive at end of time step, if False at beginning
    "surrogate_gradient_function": "piecewise_linear",  # surrogate gradient / pseudo-derivative function
    "t_ref": 5.0,  # ms, duration of refractory period
    "tau_m": 20.0,  # ms, membrane time constant
    "V_m": 0.0,  # mV, initial value of the membrane voltage
    "V_th": 0.6,  # mV, spike threshold membrane voltage
}

params_nrn_ad = {
    "adapt_tau": 2000.0,  # ms, time constant of adaptive threshold
    "adaptation": 0.0,  # initial value of the spike threshold adaptation
    "C_m": 1.0,
    "c_reg": 2.0,
    "E_L": 0.0,
    "f_target": 10.0,
    "gamma": 0.3,
    "I_e": 0.0,
    "regular_spike_arrival": True,
    "surrogate_gradient_function": "piecewise_linear",
    "t_ref": 5.0,
    "tau_m": 20.0,
    "V_m": 0.0,
    "V_th": 0.6,
}

params_nrn_ad["adapt_beta"] = 1.7 * (
    (1.0 - np.exp(-duration["step"] / params_nrn_ad["adapt_tau"]))
    / (1.0 - np.exp(-duration["step"] / params_nrn_ad["tau_m"]))
)  # prefactor of adaptive threshold

params_nrn_out = {
    "C_m": 1.0,
    "E_L": 0.0,
    "I_e": 0.0,
    "loss": "cross_entropy",  # loss function
    "regular_spike_arrival": False,
    "tau_m": 20.0,
    "V_m": 0.0,
}

####################

# Intermediate parrot neurons required between input spike generators and recurrent neurons,
# since devices cannot establish plastic synapses for technical reasons

gen_spk_in = nest.Create("spike_generator", n_in)
nrns_in = nest.Create("parrot_neuron", n_in)

# The suffix _bsshslm_2020 follows the NEST convention to indicate in the model name the paper
# that introduced it by the first letter of the authors' last names and the publication year.

nrns_reg = nest.Create("eprop_iaf_bsshslm_2020", n_reg, params_nrn_reg)
nrns_ad = nest.Create("eprop_iaf_adapt_bsshslm_2020", n_ad, params_nrn_ad)
nrns_out = nest.Create("eprop_readout_bsshslm_2020", n_out, params_nrn_out)
gen_rate_target = nest.Create("step_rate_generator", n_out)

nrns_rec = nrns_reg + nrns_ad

# %% ###########################################################################################################
# Create recorders
# ~~~~~~~~~~~~~~~~
# We also create recorders, which, while not required for the training, will allow us to track various dynamic
# variables of the neurons, spikes, and changes in synaptic weights. To save computing time and memory, the
# recorders, the recorded variables, neurons, and synapses can be limited to the ones relevant to the
# experiment, and the recording interval can be increased (see the documentation on the specific recorders). By
# default, recordings are stored in memory but can also be written to file.

n_record = 1  # number of neurons per type to record dynamic variables from - this script requires n_record >= 1
n_record_w = 3  # number of senders and targets to record weights from - this script requires n_record_w >=1

if n_record == 0 or n_record_w == 0:
    raise ValueError("n_record and n_record_w >= 1 required")

n_record_w = min(n_record_w, len(nrns_in), len(nrns_rec))  # Ensure n_record_w does not exceed lengths

params_mm_reg = {
    "interval": duration["step"],  # interval between two recorded time points
    "record_from": ["V_m", "surrogate_gradient", "learning_signal"],  # dynamic variables to record
    "start": duration["offset_gen"] + duration["delay_in_rec"],  # start time of recording
    "stop": duration["offset_gen"] + duration["delay_in_rec"] + duration["task"],  # stop time of recording
}

params_mm_ad = {
    "interval": duration["step"],
    "record_from": params_mm_reg["record_from"] + ["V_th_adapt", "adaptation"],
    "start": duration["offset_gen"] + duration["delay_in_rec"],
    "stop": duration["offset_gen"] + duration["delay_in_rec"] + duration["task"],
}

params_mm_out = {
    "interval": duration["step"],
    "record_from": ["V_m", "readout_signal", "readout_signal_unnorm", "target_signal", "error_signal"],
    "start": duration["total_offset"],
    "stop": duration["total_offset"] + duration["task"],
}

params_wr = {
    "senders": nrns_in[:n_record_w] + nrns_rec[:n_record_w],  # limit senders to subsample weights to record
    "targets": nrns_rec[:n_record_w] + nrns_out,  # limit targets to subsample weights to record from
    "start": duration["total_offset"],
    "stop": duration["total_offset"] + duration["task"],
}

params_sr = {
    "start": duration["total_offset"],
    "stop": duration["total_offset"] + duration["task"],
}

####################

mm_reg = nest.Create("multimeter", params_mm_reg)
mm_ad = nest.Create("multimeter", params_mm_ad)
mm_out = nest.Create("multimeter", params_mm_out)
sr = nest.Create("spike_recorder", params_sr)
wr = nest.Create("weight_recorder", params_wr)

nrns_reg_record = nrns_reg[:n_record]
nrns_ad_record = nrns_ad[:n_record]

# %% ###########################################################################################################
# Create connections
# ~~~~~~~~~~~~~~~~~~
# Now, we define the connectivity and set up the synaptic parameters, with the synaptic weights drawn from
# normal distributions. After these preparations, we establish the enumerated connections of the core network,
# as well as additional connections to the recorders.

params_conn_all_to_all = {"rule": "all_to_all", "allow_autapses": False}
params_conn_one_to_one = {"rule": "one_to_one"}


def calculate_glorot_dist(fan_in, fan_out):
    glorot_scale = 1.0 / max(1.0, (fan_in + fan_out) / 2.0)
    glorot_limit = np.sqrt(3.0 * glorot_scale)
    glorot_distribution = np.random.uniform(low=-glorot_limit, high=glorot_limit, size=(fan_in, fan_out))
    return glorot_distribution


dtype_weights = np.float32  # data type of weights - for reproducing TF results set to np.float32
weights_in_rec = np.array(np.random.randn(n_in, n_rec).T / np.sqrt(n_in), dtype=dtype_weights)
weights_rec_rec = np.array(np.random.randn(n_rec, n_rec).T / np.sqrt(n_rec), dtype=dtype_weights)
np.fill_diagonal(weights_rec_rec, 0.0)  # since no autapses set corresponding weights to zero
weights_rec_out = np.array(calculate_glorot_dist(n_rec, n_out).T, dtype=dtype_weights)
weights_out_rec = np.array(np.random.randn(n_rec, n_out), dtype=dtype_weights)

params_common_syn_eprop = {
    "optimizer": {
        "type": "adam",  # algorithm to optimize the weights
        "batch_size": n_batch,
        "beta_1": 0.9,  # exponential decay rate for 1st moment estimate of Adam optimizer
        "beta_2": 0.999,  # exponential decay rate for 2nd moment raw estimate of Adam optimizer
        "epsilon": 1e-8,  # small numerical stabilization constant of Adam optimizer
        "eta": 5e-3,  # learning rate
        "Wmin": -100.0,  # pA, minimal limit of the synaptic weights
        "Wmax": 100.0,  # pA, maximal limit of the synaptic weights
    },
    "average_gradient": True,  # if True, average the gradient over the learning window
    "weight_recorder": wr,
}

params_syn_base = {
    "synapse_model": "eprop_synapse_bsshslm_2020",
    "delay": duration["step"],  # ms, dendritic delay
    "tau_m_readout": params_nrn_out["tau_m"],  # ms, for technical reasons pass readout neuron membrane time constant
}

params_syn_in = params_syn_base.copy()
params_syn_in["weight"] = weights_in_rec  # pA, initial values for the synaptic weights

params_syn_rec = params_syn_base.copy()
params_syn_rec["weight"] = weights_rec_rec

params_syn_out = params_syn_base.copy()
params_syn_out["weight"] = weights_rec_out


params_syn_feedback = {
    "synapse_model": "eprop_learning_signal_connection_bsshslm_2020",
    "delay": duration["step"],
    "weight": weights_out_rec,
}

params_syn_out_out = {
    "synapse_model": "rate_connection_delayed",
    "delay": duration["step"],
    "receptor_type": 1,  # receptor type of readout neuron to receive other readout neuron's signals for softmax
    "weight": 1.0,  # pA, weight 1.0 required for correct softmax computation for technical reasons
}

params_syn_rate_target = {
    "synapse_model": "rate_connection_delayed",
    "delay": duration["step"],
    "receptor_type": 2,  # receptor type over which readout neuron receives target signal
}

params_syn_static = {
    "synapse_model": "static_synapse",
    "delay": duration["step"],
}

params_init_optimizer = {
    "optimizer": {
        "m": 0.0,  # initial 1st moment estimate m of Adam optimizer
        "v": 0.0,  # initial 2nd moment raw estimate v of Adam optimizer
    }
}

####################

nest.SetDefaults("eprop_synapse_bsshslm_2020", params_common_syn_eprop)

nest.Connect(gen_spk_in, nrns_in, params_conn_one_to_one, params_syn_static)  # connection 1
nest.Connect(nrns_in, nrns_rec, params_conn_all_to_all, params_syn_in)  # connection 2
nest.Connect(nrns_rec, nrns_rec, params_conn_all_to_all, params_syn_rec)  # connection 3
nest.Connect(nrns_rec, nrns_out, params_conn_all_to_all, params_syn_out)  # connection 4
nest.Connect(nrns_out, nrns_rec, params_conn_all_to_all, params_syn_feedback)  # connection 5
nest.Connect(gen_rate_target, nrns_out, params_conn_one_to_one, params_syn_rate_target)  # connection 6
nest.Connect(nrns_out, nrns_out, params_conn_all_to_all, params_syn_out_out)  # connection 7

nest.Connect(nrns_in + nrns_rec, sr, params_conn_all_to_all, params_syn_static)

nest.Connect(mm_reg, nrns_reg_record, params_conn_all_to_all, params_syn_static)
nest.Connect(mm_ad, nrns_ad_record, params_conn_all_to_all, params_syn_static)
nest.Connect(mm_out, nrns_out, params_conn_all_to_all, params_syn_static)

# After creating the connections, we can individually initialize the optimizer's
# dynamic variables for single synapses (here exemplarily for two connections).

nest.GetConnections(nrns_rec[0], nrns_rec[1:3]).set([params_init_optimizer] * 2)

# %% ###########################################################################################################
# Create input and output
# ~~~~~~~~~~~~~~~~~~~~~~~
# Generate input as target positions and output as motor commands.

def generate_reaching_task_input_output(n_batch, n_in, steps):
    input_positions = np.random.rand(n_batch, 2)  # random target positions in 2D plane
    input_spike_probs = np.zeros((n_batch, steps["sequence"], n_in))

    for b_idx in range(n_batch):
        input_spike_probs[b_idx, :steps["target"], :2] = input_positions[b_idx]

    input_spike_bools = input_spike_probs > np.random.rand(input_spike_probs.size).reshape(input_spike_probs.shape)
    input_spike_bools[:, 0, :] = 0  # remove spikes in 0th time step of every sequence for technical reasons

    target_positions = input_positions  # target positions are the same as input positions

    return input_spike_bools, target_positions

input_spike_prob = 0.04  # spike probability of frozen input noise
dtype_in_spks = np.float32  # data type of input spikes - for reproducing TF results set to np.float32

input_spike_bools_list = []
target_positions_list = []

for iteration in range(n_iter):
    input_spike_bools, target_positions = generate_reaching_task_input_output(n_batch, n_in, steps)
    input_spike_bools_list.append(input_spike_bools)
    target_positions_list.extend(target_positions.tolist())

input_spike_bools_arr = np.array(input_spike_bools_list).reshape(steps["task"], n_in)
timeline_task = np.arange(0.0, duration["task"], duration["step"]) + duration["offset_gen"]

params_gen_spk_in = [
    {"spike_times": timeline_task[input_spike_bools_arr[:, nrn_in_idx]].astype(dtype_in_spks)}
    for nrn_in_idx in range(n_in)
]

target_rate_changes = np.zeros((n_out, n_batch * n_iter))
target_rate_changes[:, np.arange(n_batch * n_iter)] = np.array(target_positions_list).T

params_gen_rate_target = [
    {
        "amplitude_times": np.arange(0.0, duration["task"], duration["sequence"]) + duration["total_offset"],
        "amplitude_values": target_rate_changes[nrn_out_idx],
    }
    for nrn_out_idx in range(n_out)
]

nest.SetStatus(gen_spk_in, params_gen_spk_in)
nest.SetStatus(gen_rate_target, params_gen_rate_target)

# %% ###########################################################################################################
# Force final update
# ~~~~~~~~~~~~~~~~~~
# Synapses only get active, that is, the correct weight update calculated and applied, when they transmit a
# spike. To still be able to read out the correct weights at the end of the simulation, we force spiking of the
# presynaptic neuron and thus an update of all synapses, including those that have not transmitted a spike in
# the last update interval, by sending a strong spike to all neurons that form the presynaptic side of an eprop
# synapse. This step is required purely for technical reasons.

gen_spk_final_update = nest.Create("spike_generator", 1, {"spike_times": [duration["task"] + duration["delays"]]})

nest.Connect(gen_spk_final_update, nrns_in + nrns_rec, "all_to_all", {"weight": 1000.0})

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
    "in_rec": get_weights(nrns_in, nrns_rec),
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
    "in_rec": get_weights(nrns_in, nrns_rec),
    "rec_rec": get_weights(nrns_rec, nrns_rec),
    "rec_out": get_weights(nrns_rec, nrns_out),
}

# %% ###########################################################################################################
# Read out recorders
# ~~~~~~~~~~~~~~~~~~
# We can also retrieve the recorded history of the dynamic variables and weights, as well as detected spikes.

events_mm_reg = mm_reg.get("events")
events_mm_ad = mm_ad.get("events")
events_mm_out = mm_out.get("events")
events_sr = sr.get("events")
events_wr = wr.get("events")

# %% ###########################################################################################################
# Evaluate training error
# ~~~~~~~~~~~~~~~~~~~~~~~
# Evaluate the network's training error by calculating the mean squared error between the actual and target positions.

readout_signal = events_mm_out["readout_signal"]
target_signal = events_mm_out["target_signal"]
senders = events_mm_out["senders"]

readout_signal = np.array([readout_signal[senders == i] for i in set(senders)])
target_signal = np.array([target_signal[senders == i] for i in set(senders)])

readout_signal = readout_signal.reshape((n_out, n_iter, n_batch, steps["sequence"]))
readout_signal = readout_signal[:, :, :, -steps["movement"]:]

target_signal = target_signal.reshape((n_out, n_iter, n_batch, steps["sequence"]))
target_signal = target_signal[:, :, :, -steps["movement"]:]

# Calculate the mean squared error for each iteration
loss = np.mean((readout_signal - target_signal) ** 2, axis=(0, 2, 3))

# %% ###########################################################################################################
# Plot results
# ~~~~~~~~~~~~
# Adjust plots to show the trajectory of an example.

do_plotting = True  # if True, plot the results

if not do_plotting:
    exit()

colors = {
    "blue": "#2854c5ff",
    "red": "#e04b40ff",
    "white": "#ffffffff",
}

plt.rcParams.update(
    {
        "font.sans-serif": "Arial",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.prop_cycle": cycler(color=[colors["blue"], colors["red"]]),
    }
)

# Plot training error
# ...................
# Plot the training error of the network: the mean squared error.

fig, ax = plt.subplots()

ax.plot(range(1, n_iter + 1), loss)
ax.set_ylabel("Mean Squared Error")
ax.set_xlabel("Training Iteration")
ax.set_xlim(1, n_iter)
ax.xaxis.get_major_locator().set_params(integer=True)

fig.tight_layout()

# Plot trajectory
# ...............
# Plot the trajectory of an example.

# Extract the readout signals for the final iteration
final_iteration_idx = -1  # Index for the final iteration
final_readout_signal = readout_signal[:, final_iteration_idx, :, :].reshape(n_out, -1)

# Plot the trajectory for the final iteration
fig, ax = plt.subplots()

example_idx = 0  # Index of the example to plot
target_position = target_positions_list[example_idx]

ax.plot(final_readout_signal[0], final_readout_signal[1], '.', label="Trajectory")
ax.scatter(target_position[0], target_position[1], color="red", label="Target")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.legend()

fig.tight_layout()

# Plot raster plots
# ~~~~~~~~~~~~~~~~~
# Plot the raster plots of spikes recorded during the simulation.

fig, ax = plt.subplots()

# Plot spikes for each population in different colors
input_spikes = events_sr["senders"] < n_in
rec_spikes = (events_sr["senders"] >= n_in) & (events_sr["senders"] < n_in + n_rec)
output_spikes = events_sr["senders"] >= n_in + n_rec

ax.plot(events_sr["times"][input_spikes], events_sr["senders"][input_spikes], '.', markersize=2, color='r', label='Input Neurons')
ax.plot(events_sr["times"][rec_spikes], events_sr["senders"][rec_spikes], '.', markersize=2, color='g', label='Recurrent Neurons')
ax.plot(events_sr["times"][output_spikes], events_sr["senders"][output_spikes], '.', markersize=2, color='b', label='Output Neurons')

ax.set_xlabel("Time (ms)")
ax.set_ylabel("Neuron ID")
ax.set_title("Raster Plot of Spikes")
ax.legend()

fig.tight_layout()
plt.show()
