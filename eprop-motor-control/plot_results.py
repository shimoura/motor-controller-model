import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from cycler import cycler

def plot_all_loss_curves(results_dir, metric_fn=None, savefig=True, showfig=True):
    """
    Plot all loss curves from *_results.npz files in the given directory and its subdirectories.
    Args:
        results_dir: Directory containing subfolders with *_results.npz files
        metric_fn: Function to extract a metric from the loss array (default: final loss)
        savefig: Whether to save the figure as 'all_loss_curves.png' in results_dir
        showfig: Whether to display the figure interactively
    """
    import glob
    if metric_fn is None:
        metric_fn = lambda loss: loss[-1]  # Default: final loss

    metrics = []
    # Recursively find all *_results.npz files
    npz_files = glob.glob(os.path.join(results_dir, '**', '*_results.npz'), recursive=True)
    for fpath in npz_files:
        data = np.load(fpath)
        loss = data['loss']
        metric = metric_fn(loss)
        label = os.path.relpath(fpath, results_dir).replace('_results.npz', '')
        metrics.append((label, metric, loss))

    if not metrics:
        print("No *_results.npz files found in", results_dir)
        return

    metrics.sort(key=lambda x: x[1])
    print("Best scenario:", metrics[0][0], "with loss:", metrics[0][1])
    print("\nAll results:")
    for label, metric, _ in metrics:
        print(f"{label}: {metric}")

    plt.figure(figsize=(12, 8))
    for label, _, loss in metrics:
        plt.plot(loss, label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("All Loss Curves")
    plt.legend()
    if savefig:
        plt.savefig(os.path.join(results_dir, 'all_loss_curves.png'), dpi=300)
    if showfig:
        plt.show()
    plt.close()

def plot_training_error(loss, n_iter, n_samples, n_batch, out_path):
    """
    Plot the training error (loss curve) for a single simulation run.
    Args:
        loss: Array of loss values per iteration
        n_iter: Number of training iterations
        n_samples: Number of samples per batch
        n_batch: Number of batches
        out_path: Path to save the figure
    """
    fig, ax = plt.subplots()
    ax.plot(range(1, n_iter * n_samples * n_batch + 1), loss)
    ax.set_ylabel(r"$E = \frac{1}{2} \sum_{t,k} (y_k^t -y_k^{*,t})^2$")
    ax.set_xlabel("training iteration")
    ax.set_xlim(1, n_iter * n_samples * n_batch)
    ax.xaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_spikes_and_dynamics(events_sr, events_mm_rec, events_mm_out, nrns_rec, n_record, duration, colors, out_prefix):
    """
    Plot spikes and all recorded dynamic variables for a simulation run, showing pre- and post-training windows side by side.
    Args:
        events_sr: Spike recorder events
        events_mm_rec: Multimeter events for recurrent neurons
        events_mm_out: Multimeter events for output neurons
        nrns_rec: List of recurrent neuron IDs
        n_record: Number of recorded neurons
        duration: Dictionary of timing values
        colors: Color dictionary
        out_prefix: Prefix for output files
    """
    def plot_recordable(ax, events, recordable, ylabel, xlims):
        for sender in set(events["senders"]):
            idc_sender = events["senders"] == sender
            idc_times = (events["times"][idc_sender] > xlims[0]) & (events["times"][idc_sender] < xlims[1])
            ax.plot(
                events["times"][idc_sender][idc_times],
                events[recordable][idc_sender][idc_times],
                lw=0.5,
            )
        ax.set_ylabel(ylabel)
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
    # Define the two time windows: pre- and post-training
    xlims_list = [
        (0, duration["sequence"]),
        (duration["task"] - duration["sequence"], duration["task"]),
    ]
    fig, axs = plt.subplots(8, 2, sharex='col', figsize=(8, 12))
    # Left column: pre-training window
    plot_spikes(axs[0,0], events_sr, nrns_rec, r"$z_j$\n", xlims_list[0])
    plot_recordable(axs[1,0], events_mm_rec, "V_m", r"$v_j$\n(mV)", xlims_list[0])
    plot_recordable(axs[2,0], events_mm_rec, "surrogate_gradient", r"$\psi_j$\n", xlims_list[0])
    plot_recordable(axs[3,0], events_mm_rec, "learning_signal", r"$L_j$\n(pA)", xlims_list[0])
    plot_recordable(axs[4,0], events_mm_out, "V_m", r"$v_k$\n(mV)", xlims_list[0])
    plot_recordable(axs[5,0], events_mm_out, "target_signal", r"$y^*_k$\n", xlims_list[0])
    plot_recordable(axs[6,0], events_mm_out, "readout_signal", r"$y_k$\n", xlims_list[0])
    plot_recordable(axs[7,0], events_mm_out, "error_signal", r"$y_k-y^*_k$\n", xlims_list[0])
    # Right column: post-training window
    plot_spikes(axs[0,1], events_sr, nrns_rec, r"$z_j$\n", xlims_list[1])
    plot_recordable(axs[1,1], events_mm_rec, "V_m", r"$v_j$\n(mV)", xlims_list[1])
    plot_recordable(axs[2,1], events_mm_rec, "surrogate_gradient", r"$\psi_j$\n", xlims_list[1])
    plot_recordable(axs[3,1], events_mm_rec, "learning_signal", r"$L_j$\n(pA)", xlims_list[1])
    plot_recordable(axs[4,1], events_mm_out, "V_m", r"$v_k$\n(mV)", xlims_list[1])
    plot_recordable(axs[5,1], events_mm_out, "target_signal", r"$y^*_k$\n", xlims_list[1])
    plot_recordable(axs[6,1], events_mm_out, "readout_signal", r"$y_k$\n", xlims_list[1])
    plot_recordable(axs[7,1], events_mm_out, "error_signal", r"$y_k-y^*_k$\n", xlims_list[1])
    # Set labels and titles
    axs[0,0].set_title("Pre-training window")
    axs[0,1].set_title("Post-training window")
    for i in range(8):
        axs[i,0].label_outer()
        axs[i,1].label_outer()
    axs[-1,0].set_xlabel(r"$t$ (ms)")
    axs[-1,1].set_xlabel(r"$t$ (ms)")
    axs[-1,0].set_xlim(*xlims_list[0])
    axs[-1,1].set_xlim(*xlims_list[1])
    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_spikes_and_dynamic_variables_side_by_side.png", dpi=300)
    plt.close(fig)

def plot_weight_time_courses(events_wr, weights_pre_train, nrns_rec, nrns_out, n_record_w, colors, duration, out_path):
    """
    Plot the time course of selected synaptic weights during training.
    Args:
        events_wr: Weight recorder events
        weights_pre_train: Initial weights
        nrns_rec: List of recurrent neuron IDs
        nrns_out: List of output neuron IDs
        n_record_w: Number of recorded weights
        colors: Color dictionary
        duration: Dictionary of timing values
        out_path: Path to save the figure
    """
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
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    plot_weight_time_course(
        axs[0], events_wr, nrns_rec[:n_record_w], nrns_rec[:n_record_w], "rec_rec", r"$W_\text{rec}$ (pA)"
    )
    plot_weight_time_course(
        axs[1], events_wr, nrns_rec[:n_record_w], nrns_out, "rec_out", r"$W_\text{out}$ (pA)"
    )
    axs[-1].set_xlabel(r"$t$ (ms)")
    axs[-1].set_xlim(0, duration["task"])
    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_weight_matrices(weights_pre_train, weights_post_train, colors, out_path):
    """
    Plot the initial and final weight matrices for recurrent and output connections.
    Args:
        weights_pre_train: Initial weights
        weights_post_train: Final weights
        colors: Color dictionary
        out_path: Path to save the figure
    """
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "cmap", ((0.0, colors["blue"]), (0.5, colors["white"]), (1.0, colors["red"]))
    )
    fig, axs = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(10, 8))
    all_w_extrema = []
    for k in weights_pre_train.keys():
        w_pre = weights_pre_train[k]["weight"]
        w_post = weights_post_train[k]["weight"]
        all_w_extrema.append([np.min(w_pre), np.max(w_pre), np.min(w_post), np.max(w_post)])
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
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
