import numpy as np
import nest
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import yaml

# Add dataset path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent / "dataset_motor_training"))
from load_dataset import load_data_file

# Parameters (adjust as needed)

# Load config.yaml for all parameters

config_path = Path(__file__).resolve().parent / "config" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


# Use same variable names as eprop-reaching-task.py
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

# Load trained weights safely
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

# Load trajectory

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

# Load and resample trajectory as in eprop-reaching-task.py
trajectories = []
for idx, sample_id in enumerate(sample_ids):
    traj_num = int(training_dataset[sample_id][0][0])
    traj_file = dataset_path.parent / f"trajectory{traj_num}.txt"
    traj_data = np.loadtxt(traj_file)
    orig_num_pts, orig_dur = len(traj_data), len(traj_data) * 0.1
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

# Set up NEST
nest.ResetKernel()
nest.SetKernelStatus({"resolution": step_ms})

# Create neuron populations and input generator
# Install custom neuron module
nest.Install("motor_neuron_module")
n_rb = num_centers
nrns_rb = nest.Create("rb_neuron", n_rb)
params_rb_neuron = config["neurons"]["rb"].copy()
params_rb_neuron["simulation_steps"] = int(duration / step_ms + 1)
params_rb_neuron["sdev"] = scale_rate * config["rbf"]["width"]
params_rb_neuron["max_peak_rate"] = scale_rate / step_ms
nest.SetStatus(nrns_rb, params_rb_neuron)

# Create neuron populations and input generator
gen_poisson_in = nest.Create("inhomogeneous_poisson_generator")
nrns_rec = nest.Create("eprop_iaf_bsshslm_2020", n_rec)
nrns_out = nest.Create("eprop_readout_bsshslm_2020", n_out)
spike_recorder = nest.Create("spike_recorder")

# Set input rates
nest.SetStatus(
    gen_poisson_in, {"rate_times": in_rate_times, "rate_values": input_spk_rate}
)

# Connect input to rec
params_conn_all_to_all = {"rule": "all_to_all", "allow_autapses": False}
params_syn_static = {"synapse_model": "static_synapse", "weight": 1.0, "delay": step_ms}
w_input = float(config["synapses"]["w_input"])
params_syn_rb_to_rec = {
    "synapse_model": "static_synapse",
    "weight": w_input,
    "delay": step_ms,
}

# Poisson generator to rb_neuron
nest.Connect(gen_poisson_in, nrns_rb, params_conn_all_to_all, params_syn_static)
# rb_neuron to recurrent neurons: connect each rb_neuron to a group of rec neurons
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
    if group_exc:
        nest.Connect(rb, group_exc, params_conn_all_to_all, params_syn_rb_to_rec)
    if group_inh:
        nest.Connect(rb, group_inh, params_conn_all_to_all, params_syn_rb_to_rec)

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

# Record spikes
nest.Connect(nrns_rec, spike_recorder)

# Run simulation
nest.Simulate(duration)

# Get and plot raster
events = spike_recorder.get("events")
plt.figure(figsize=(10, 6))
plt.scatter(events["times"], events["senders"], s=2)
plt.xlabel("Time (ms)")
plt.ylabel("Neuron ID")
plt.title("Raster Plot of Recurrent Neurons")
plt.tight_layout()

# Plot loaded weights for verification
# Use the same color scheme as in plot_weight_matrices
import matplotlib as mpl

colors = {"blue": "#1f77b4", "red": "#d62728", "white": "#ffffff"}
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "cmap", ((0.0, colors["red"]), (0.5, colors["white"]), (1.0, colors["blue"]))
)
vmin = min(np.min(rec_rec_weights), np.min(rec_out_weights))
vmax = max(np.max(rec_rec_weights), np.max(rec_out_weights))
norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
pc0 = axs[0].pcolormesh(rec_rec_weights, cmap=cmap, norm=norm)
axs[0].set_title("Loaded rec_rec weights")
axs[0].set_xlabel("Presynaptic neuron")
axs[0].set_ylabel("Postsynaptic neuron")
plt.colorbar(pc0, ax=axs[0])
pc1 = axs[1].pcolormesh(rec_out_weights, cmap=cmap, norm=norm)
axs[1].set_title("Loaded rec_out weights")
axs[1].set_xlabel("Presynaptic neuron")
axs[1].set_ylabel("Postsynaptic neuron")
plt.colorbar(pc1, ax=axs[1])
plt.tight_layout()
plt.show()
