"""
trained_weights_net.py
----------------------

This script loads trained weights from a previously trained e-prop motor control network
and runs a test simulation using NEST. It reproduces the network setup and input encoding
used in the main training script (eprop-reaching-task.py) for consistency.

Key steps:
- Loads configuration parameters from config.yaml.
- Loads trained weights from .npz file.
- Loads and resamples trajectory data as input.
- Sets up NEST simulation with neuron populations and connections.
- Applies loaded weights to recurrent and output connections.
- Runs the simulation and plots spike raster and loaded weight matrices.

Input connections:
- Each rb_neuron is connected to a group of excitatory and inhibitory recurrent neurons,
  matching the grouping logic in eprop-reaching-task.py.

This script is intended for post-training evaluation and visualization.

Author: Renan Oliveira Shimoura
"""

import numpy as np
import nest
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import sys
import yaml

mpl.use("Agg")

# Add dataset path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent / "dataset_motor_training"))
from load_dataset import load_data_file

# --------------------------------------------------------------------------------------
# Load configuration parameters from config.yaml
# --------------------------------------------------------------------------------------

config_path = Path(__file__).resolve().parent / "config" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Use same variable names as eprop-reaching-task.py for consistency
n_rec = int(config["neurons"]["n_rec"])
n_out = int(config["neurons"]["n_out"])
step_ms = float(config["simulation"]["step"])
sequence = float(config["task"]["sequence"])
silent_period = float(config["task"]["silent_period"])
n_iter = 1  # Only one iteration for testing
num_centers = int(config["rbf"]["num_centers"])
scale_rate = float(config["rbf"]["scale_rate"])

# Duration matches eprop-reaching-task.py
n_timesteps_per_sequence = int(round((sequence + silent_period) / step_ms))
n_samples_per_trajectory_to_use = int(config["task"]["n_samples_per_trajectory_to_use"])
trajectory_ids_to_use = config["task"]["trajectory_ids_to_use"]
n_samples = len(trajectory_ids_to_use) * n_samples_per_trajectory_to_use
duration_task = n_timesteps_per_sequence * n_samples * n_iter * step_ms
duration = duration_task + step_ms

# --------------------------------------------------------------------------------------
# Load trained weights from .npz file
# --------------------------------------------------------------------------------------

weights_path = (
    Path(__file__).resolve().parent
    / "sim_results"
    / "default_plastic_False_manualRBF_False"
    / "trained_weights.npz"
)
if not weights_path.exists():
    raise FileNotFoundError(f"Trained weights file not found: {weights_path}")

weights = np.load(weights_path)
rec_rec_weights = weights.get("rec_rec")
rec_out_weights = weights.get("rec_out")

if rec_rec_weights is None or rec_out_weights is None:
    raise KeyError("Missing 'rec_rec' or 'rec_out' arrays in trained_weights.npz")

# --------------------------------------------------------------------------------------
# Load and resample trajectory data
# --------------------------------------------------------------------------------------

dataset_path = (
    Path(__file__).resolve().parent.parent
    / "dataset_motor_training"
    / "dataset_spikes.gdf"
)
training_dataset = load_data_file(str(dataset_path))
sample_ids = [
    tid * config["task"]["samples_per_trajectory_in_dataset"] + j
    for tid in trajectory_ids_to_use
    for j in range(n_samples_per_trajectory_to_use)
]
assert len(sample_ids) == n_samples

# Resample each trajectory to match simulation time steps
trajectories = []
for idx, sample_id in enumerate(sample_ids):
    traj_num = int(training_dataset[sample_id][0][0])
    traj_file = dataset_path.parent / f"trajectory{traj_num}.txt"
    traj_data = np.loadtxt(traj_file)
    orig_num_pts, orig_dur = len(traj_data), len(traj_data) * 0.1  # 0.1ms resolution
    resampled_time = np.linspace(0, orig_dur, int(sequence / step_ms), endpoint=False)
    orig_time = np.arange(orig_num_pts) * 0.1
    trajectory_signal = np.interp(resampled_time, orig_time, traj_data)
    # Prepend zeros for silent period
    if silent_period > 0:
        silent_steps = int(silent_period / step_ms)
        trajectory_signal = np.concatenate((np.zeros(silent_steps), trajectory_signal))
    trajectories.append(trajectory_signal)

# Concatenate and tile for all iterations
input_spk_rate = np.concatenate(trajectories) * scale_rate
input_spk_rate = np.tile(input_spk_rate, n_iter)
in_rate_times = np.arange(len(input_spk_rate)) * step_ms + step_ms

# --------------------------------------------------------------------------------------
# Set up NEST simulation
# --------------------------------------------------------------------------------------

nest.ResetKernel()
nest.SetKernelStatus({"resolution": step_ms, "rng_seed": 1234})


# ----------------------------------------------------------------------------------------
# Create neuron populations and input generator
# ----------------------------------------------------------------------------------------

# Create neuron populations and input generator
nest.Install("motor_neuron_module")
n_rb = num_centers
nrns_rb = nest.Create("rb_neuron", n_rb)
params_rb_neuron = config["neurons"]["rb"].copy()
params_rb_neuron["simulation_steps"] = int(duration / step_ms + 1)
params_rb_neuron["sdev"] = scale_rate * config["rbf"]["width"]
params_rb_neuron["max_peak_rate"] = scale_rate / step_ms
nest.SetStatus(nrns_rb, params_rb_neuron)

# Set the desired center for each rb_neuron's receptive field
angle_centers = np.linspace(0.0, np.pi, n_rb)
desired_rates = angle_centers * scale_rate
for i, nrn in enumerate(nrns_rb):
    nest.SetStatus(nrn, {"desired": desired_rates[i]})

gen_poisson_in = nest.Create("inhomogeneous_poisson_generator")
nrns_rec = nest.Create("eprop_iaf_bsshslm_2020", n_rec)
nrns_out = nest.Create("eprop_readout_bsshslm_2020", n_out)
spike_recorder = nest.Create("spike_recorder")
spike_recorder_rb = nest.Create("spike_recorder")  # Recorder for input neurons

# Set input rates for Poisson generator
nest.SetStatus(
    gen_poisson_in, {"rate_times": in_rate_times, "rate_values": input_spk_rate}
)

# --------------------------------------------------------------------------------------
# Connect input to recurrent neurons via rb_neuron
# --------------------------------------------------------------------------------------

params_conn_all_to_all = {"rule": "all_to_all", "allow_autapses": False}
params_conn_one_to_one = {"rule": "one_to_one", "allow_autapses": False}
params_syn_static = {"synapse_model": "static_synapse", "weight": 1.0, "delay": step_ms}

# Load trained input weights if present
rb_rec_weights = weights.get("rb_rec")  # shape: (n_rec, n_rb) if present

# Poisson generator to rb_neuron
nest.Connect(gen_poisson_in, nrns_rb, params_conn_all_to_all, params_syn_static)

exc_ratio = float(config["neurons"]["exc_ratio"])
n_rec_exc = int(n_rec * exc_ratio)
n_rec_inh = n_rec - n_rec_exc
nrns_rec_exc = nrns_rec[:n_rec_exc]
nrns_rec_inh = nrns_rec[n_rec_exc:]

for i, rb in enumerate(nrns_rb):
    exc_start = int(i * n_rec_exc / num_centers)
    exc_end = int((i + 1) * n_rec_exc / num_centers)
    inh_start = int(i * n_rec_inh / num_centers)
    inh_end = int((i + 1) * n_rec_inh / num_centers)
    group_exc = nrns_rec_exc[exc_start:exc_end]
    group_inh = nrns_rec_inh[inh_start:inh_end]

    # Use trained input weights
    weights_exc = rb_rec_weights[exc_start:exc_end, i]
    nest.Connect(
        rb.tolist() * len(group_exc),
        group_exc,
        params_conn_one_to_one,
        {
            "synapse_model": "static_synapse",
            "weight": weights_exc,
            "delay": [step_ms] * len(group_exc),
        },
    )
    weights_inh = rb_rec_weights[n_rec_exc + inh_start : n_rec_exc + inh_end, i]
    nest.Connect(
        rb.tolist() * len(group_inh),
        group_inh,
        params_conn_one_to_one,
        {
            "synapse_model": "static_synapse",
            "weight": weights_inh,
            "delay": [step_ms] * len(group_inh),
        },
    )

# --------------------------------------------------------------------------------------
# Connect recurrent and output neurons using loaded weights
# --------------------------------------------------------------------------------------

# Connect rec to rec with trained weights
for i, pre in enumerate(nrns_rec):
    for j, post in enumerate(nrns_rec):
        nest.Connect(
            pre,
            post,
            syn_spec={
                "synapse_model": "static_synapse",
                "weight": rec_rec_weights[j, i],
                "delay": step_ms,
            },
        )

# Connect rec to out with trained weights
for i, pre in enumerate(nrns_rec):
    for j, post in enumerate(nrns_out):
        nest.Connect(
            pre,
            post,
            syn_spec={
                "synapse_model": "static_synapse",
                "weight": rec_out_weights[j, i],
                "delay": step_ms,
            },
        )

# --------------------------------------------------------------------------------------
# Record spikes and run simulation
# --------------------------------------------------------------------------------------

nest.Connect(nrns_rec, spike_recorder)
nest.Connect(nrns_rb, spike_recorder_rb)

nest.Simulate(duration)

# --------------------------------------------------------------------------------------
# Extract weights from the network for comparison
# --------------------------------------------------------------------------------------


def get_weights(pop_pre, pop_post):
    conns = nest.GetConnections(pop_pre, pop_post).get(["source", "target", "weight"])
    if not conns["source"]:
        return np.zeros((len(pop_post), len(pop_pre)))
    senders = np.array(conns["source"]) - np.min(conns["source"])
    targets = np.array(conns["target"]) - np.min(conns["target"])
    weight_matrix = np.zeros((len(pop_post), len(pop_pre)))
    weight_matrix[targets, senders] = conns["weight"]
    return weight_matrix


rec_rec_weights_extracted = get_weights(nrns_rec, nrns_rec) # recurrent to recurrent
rec_out_weights_extracted = get_weights(nrns_rec, nrns_out) # recurrent to output
rb_rec_weights_extracted = get_weights(nrns_rb, nrns_rec)  # input to recurrent

# --------------------------------------------------------------------------------------
# Plot results: spike raster and loaded weights vs extracted weights
# --------------------------------------------------------------------------------------

events = spike_recorder.get("events")
events_rb = spike_recorder_rb.get("events")

fig_raster, axs_raster = plt.subplots(
    2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [1, 5]}, sharex=True
)
axs_raster[0].scatter(events_rb["times"], events_rb["senders"], s=2, color="#1f77b4")
axs_raster[0].set_ylabel("Input Neuron ID")
axs_raster[0].set_title("Raster Plot of Input (rb_neuron) Neurons")
axs_raster[1].scatter(events["times"], events["senders"], s=2)
axs_raster[1].set_xlabel("Time (ms)")
axs_raster[1].set_ylabel("Recurrent Neuron ID")
axs_raster[1].set_title("Raster Plot of Recurrent Neurons")
plt.tight_layout()
plt.savefig("./sim_results/trained_weights_raster_plot.png")

# Plot loaded weights and extracted weights for verification
colors = {"blue": "#1f77b4", "red": "#d62728", "white": "#ffffff"}
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "cmap", ((0.0, colors["red"]), (0.5, colors["white"]), (1.0, colors["blue"]))
)
vmin = min(
    np.min(rec_rec_weights),
    np.min(rec_out_weights),
    np.min(rec_rec_weights_extracted),
    np.min(rec_out_weights_extracted),
)
vmax = max(
    np.max(rec_rec_weights),
    np.max(rec_out_weights),
    np.max(rec_rec_weights_extracted),
    np.max(rec_out_weights_extracted),
)
norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
pc0 = axs[0, 0].pcolormesh(rec_rec_weights, cmap=cmap, norm=norm)
axs[0, 0].set_title("Loaded rec_rec weights")
axs[0, 0].set_xlabel("Presynaptic neuron")
axs[0, 0].set_ylabel("Postsynaptic neuron")
plt.colorbar(pc0, ax=axs[0, 0])
pc1 = axs[0, 1].pcolormesh(rec_out_weights, cmap=cmap, norm=norm)
axs[0, 1].set_title("Loaded rec_out weights")
axs[0, 1].set_xlabel("Presynaptic neuron")
axs[0, 1].set_ylabel("Postsynaptic neuron")
plt.colorbar(pc1, ax=axs[0, 1])
pc2 = axs[1, 0].pcolormesh(rec_rec_weights_extracted, cmap=cmap, norm=norm)
axs[1, 0].set_title("Extracted rec_rec weights")
axs[1, 0].set_xlabel("Presynaptic neuron")
axs[1, 0].set_ylabel("Postsynaptic neuron")
plt.colorbar(pc2, ax=axs[1, 0])
pc3 = axs[1, 1].pcolormesh(rec_out_weights_extracted, cmap=cmap, norm=norm)
axs[1, 1].set_title("Extracted rec_out weights")
axs[1, 1].set_xlabel("Presynaptic neuron")
axs[1, 1].set_ylabel("Postsynaptic neuron")
plt.colorbar(pc3, ax=axs[1, 1])
plt.tight_layout()
plt.savefig("./sim_results/trained_weights_comparison.png")

# --- Plot input-to-recurrent weights (rb_neuron to rec) ---

fig_rbrec, axs_rbrec = plt.subplots(1, 2, figsize=(12, 5))
vmin_rb = -np.max(rb_rec_weights_extracted)
vmax_rb = np.max(rb_rec_weights_extracted)
norm_rb = mpl.colors.TwoSlopeNorm(vmin=vmin_rb, vcenter=0, vmax=vmax_rb)
if rb_rec_weights is not None:
    pc_rb_loaded = axs_rbrec[0].pcolormesh(rb_rec_weights, cmap=cmap, norm=norm_rb)
    axs_rbrec[0].set_title("Loaded rb_rec weights (input to recurrent)")
    axs_rbrec[0].set_xlabel("Input neuron (rb_neuron)")
    axs_rbrec[0].set_ylabel("Recurrent neuron")
    plt.colorbar(pc_rb_loaded, ax=axs_rbrec[0])
else:
    axs_rbrec[0].set_title("Loaded rb_rec weights not present")
    axs_rbrec[0].axis("off")

pc_rb_extracted = axs_rbrec[1].pcolormesh(
    rb_rec_weights_extracted, cmap=cmap, norm=norm_rb
)
axs_rbrec[1].set_title("Extracted rb_rec weights (input to recurrent)")
axs_rbrec[1].set_xlabel("Input neuron (rb_neuron)")
axs_rbrec[1].set_ylabel("Recurrent neuron")
plt.colorbar(pc_rb_extracted, ax=axs_rbrec[1])
plt.tight_layout()
plt.savefig("./sim_results/trained_weights_rb_rec_comparison.png")
