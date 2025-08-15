import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from cycler import cycler


def plot_all_loss_curves(
    results_dir,
    metric_fn=None,
    savefig=True,
    showfig=True,
    avg_last_n=10,
):
    """
    Plot all loss curves from *_results.npz files in the given directory and its subdirectories.

    Args:
        results_dir: Directory containing subfolders with *_results.npz files.
        metric_fn: Function to extract a metric from the loss array (default: avg of last n).
        savefig: Whether to save the figure as 'all_loss_curves.png' in results_dir.
        showfig: Whether to display the figure interactively.
        avg_last_n: Number of final iterations to average for the default metric.
    """
    import glob
    import numpy as np

    # If no custom metric function is provided, use a robust default metric.
    if metric_fn is None:

        def default_metric(loss):
            # Handle cases where the simulation run was very short.
            if len(loss) > avg_last_n:
                # Ideal case: average over the last N valid iterations.
                return np.mean(loss[-(avg_last_n + 1) : -1])
            elif len(loss) > 1:
                # Fallback for short runs: average all valid points.
                return np.mean(loss[:-1])
            else:
                # Return infinity for invalid runs so they are ranked last.
                return np.inf

        metric_fn = default_metric

    metrics = []
    # Recursively find all *results.npz files
    npz_files = glob.glob(
        os.path.join(results_dir, "**", "*results.npz"), recursive=True
    )
    for fpath in npz_files:
        data = np.load(fpath)
        loss = data["loss"]
        metric = metric_fn(loss)
        label = os.path.relpath(fpath, results_dir).replace("/results.npz", "")
        metrics.append((label, metric, loss))

    if not metrics:
        print("No *results.npz files found in", results_dir)
        return

    # The rest of the function remains the same, but is now more robust.
    metrics.sort(key=lambda x: x[1])

    print(f"--- Results (ranked by average of last {avg_last_n} iterations) ---")
    print("Best scenario:", metrics[0][0], "with loss:", metrics[0][1])
    print("\nAll results:")
    for label, metric, _ in metrics:
        print(f"{label}: {metric}")

    plt.figure(figsize=(12, 8))
    for label, _, loss in metrics:
        if len(loss) > 1:
            x_values = np.arange(1, len(loss))
            plt.plot(x_values, loss, label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("All Loss Curves")
    plt.legend()
    if savefig:
        plt.savefig(os.path.join(results_dir, "all_loss_curves.png"), dpi=300)
    if showfig:
        plt.show()
    plt.close()


def plot_training_error(loss, out_path, x=None, xlabel="training iteration"):
    """
    Plot the training error (loss curve) for a single simulation run.
    Args:
        loss: Array of loss values per iteration
        out_path: Path to save the figure
        x: Optional x-axis values (default: range(1, len(loss)+1))
        xlabel: Label for the x-axis (default: "training iteration")
    """
    loss = np.asarray(loss)
    if x is None:
        x = np.arange(1, len(loss) + 1)
    else:
        x = np.asarray(x)
        minlen = min(len(x), len(loss))
        x = x[:minlen]
        loss = loss[:minlen]
    fig, ax = plt.subplots(figsize=(4, 3))  # Changed figure size here
    ax.plot(x, loss)
    ax.set_ylabel(r"$E = \frac{1}{2} \sum_{t,k} (y_k^t -y_k^{*,t})^2$")
    ax.set_xlabel(xlabel)

    # Adjust x-axis limits to avoid showing the last point
    if len(x) > 1:
        ax.set_xlim(x[0], x[-2])

    ax.set_xlim(x[0], x[-1])
    ax.xaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_spikes_and_dynamics(
    events_sr,
    events_mm_rec,
    events_mm_out,
    nrns_rec,
    n_record,
    duration,
    colors,
    out_prefix,
):
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

    def plot_recordable(ax, events, recordable, ylabel, xlims, color_cycle=None):
        senders = np.unique(events["senders"])
        for idx, sender in enumerate(senders):
            idc_sender = events["senders"] == sender
            idc_times = (events["times"][idc_sender] > xlims[0]) & (
                events["times"][idc_sender] < xlims[1]
            )
            if np.any(idc_times):
                color = color_cycle[idx % len(color_cycle)] if color_cycle else None
                ax.plot(
                    events["times"][idc_sender][idc_times],
                    events[recordable][idc_sender][idc_times],
                    lw=1.5,
                    color=color,
                    alpha=0.8,
                )
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)

    def plot_spikes(ax, events, nrns, ylabel, xlims, color="black"):
        idc_times = (events["times"] > xlims[0]) & (events["times"] < xlims[1])
        idc_sender = np.isin(events["senders"][idc_times], nrns.tolist())
        senders_subset = events["senders"][idc_times][idc_sender]
        times_subset = events["times"][idc_times][idc_sender]
        if senders_subset.size > 0:
            ax.scatter(times_subset, senders_subset, s=2, color=color, alpha=0.7)
            margin = np.abs(np.max(senders_subset) - np.min(senders_subset)) * 0.1 + 1
            ax.set_ylim(
                np.min(senders_subset) - margin, np.max(senders_subset) + margin
            )
        else:
            ax.set_ylim(0, 1)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)

    # Define pre/post windows dynamically:
    n_trajectories = duration["n_trajectories"]
    pre_train_window = (0, n_trajectories * duration["total_sequence_with_silence"])
    post_train_window = (
        duration["task"] - n_trajectories * duration["total_sequence_with_silence"],
        duration["task"],
    )
    xlims_list = [pre_train_window, post_train_window]
    fig, axs = plt.subplots(8, 2, sharex="col", figsize=(6, 12), dpi=300)

    # Color cycles for better distinction
    rec_colors = [
        colors.get("blue", "#1f77b4"),
        colors.get("red", "#d62728"),
        colors.get("green", "#2ca02c"),
        colors.get("orange", "#ff7f0e"),
    ]
    out_colors = [
        colors.get("blue", "#1f77b4"),
        colors.get("red", "#d62728"),
        colors.get("pink", "#e377c2"),
    ]

    # Left column: pre-training window
    plot_spikes(
        axs[0, 0],
        events_sr,
        nrns_rec,
        r"$z_j$",
        xlims_list[0],
        color=colors.get("black", "black"),
    )
    plot_recordable(
        axs[1, 0], events_mm_rec, "V_m", r"$v_j$ (mV)", xlims_list[0], rec_colors
    )
    plot_recordable(
        axs[2, 0],
        events_mm_rec,
        "surrogate_gradient",
        r"$\psi_j$",
        xlims_list[0],
        rec_colors,
    )
    plot_recordable(
        axs[3, 0],
        events_mm_rec,
        "learning_signal",
        r"$L_j$ (pA)",
        xlims_list[0],
        rec_colors,
    )
    plot_recordable(
        axs[4, 0], events_mm_out, "V_m", r"$v_k$ (mV)", xlims_list[0], out_colors
    )
    plot_recordable(
        axs[5, 0], events_mm_out, "target_signal", r"$y^*_k$", xlims_list[0], out_colors
    )
    plot_recordable(
        axs[6, 0], events_mm_out, "readout_signal", r"$y_k$", xlims_list[0], out_colors
    )
    plot_recordable(
        axs[7, 0],
        events_mm_out,
        "error_signal",
        r"$y_k-y^*_k$",
        xlims_list[0],
        out_colors,
    )

    # Right column: post-training window
    plot_spikes(
        axs[0, 1],
        events_sr,
        nrns_rec,
        r"$z_j$",
        xlims_list[1],
        color=colors.get("black", "black"),
    )
    plot_recordable(
        axs[1, 1], events_mm_rec, "V_m", r"$v_j$ (mV)", xlims_list[1], rec_colors
    )
    plot_recordable(
        axs[2, 1],
        events_mm_rec,
        "surrogate_gradient",
        r"$\psi_j$",
        xlims_list[1],
        rec_colors,
    )
    plot_recordable(
        axs[3, 1],
        events_mm_rec,
        "learning_signal",
        r"$L_j$ (pA)",
        xlims_list[1],
        rec_colors,
    )
    plot_recordable(
        axs[4, 1], events_mm_out, "V_m", r"$v_k$ (mV)", xlims_list[1], out_colors
    )
    plot_recordable(
        axs[5, 1], events_mm_out, "target_signal", r"$y^*_k$", xlims_list[1], out_colors
    )
    plot_recordable(
        axs[6, 1], events_mm_out, "readout_signal", r"$y_k$", xlims_list[1], out_colors
    )
    plot_recordable(
        axs[7, 1],
        events_mm_out,
        "error_signal",
        r"$y_k-y^*_k$",
        xlims_list[1],
        out_colors,
    )

    # Set labels and titles
    axs[0, 0].set_title("Pre-training window", fontsize=12, fontweight="bold")
    axs[0, 1].set_title("Post-training window", fontsize=12, fontweight="bold")
    for i in range(8):
        axs[i, 0].label_outer()
        axs[i, 0].tick_params(axis="both", which="major", labelsize=8)
        axs[i, 1].tick_params(axis="both", which="major", labelsize=8)
    axs[-1, 0].set_xlabel(r"$t$ (ms)")
    axs[-1, 1].set_xlabel(r"$t$ (ms)")
    axs[-1, 0].set_xlim(*xlims_list[0])
    axs[-1, 1].set_xlim(*xlims_list[1])
    fig.align_ylabels()
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.suptitle(
        "Spikes and Dynamics (Pre- and Post-Training)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.savefig(f"{out_prefix}", dpi=300)
    plt.close(fig)


def plot_weight_time_courses(
    events_wr,
    weights_pre_train,
    nrns_rec,
    nrns_out,
    n_record_w,
    colors,
    duration,
    out_path,
):
    """
    Plot the time course of selected synaptic weights during training.
    Args:
        events_wr: Weight recorder events
        weights_pre_train: Dict with keys 'source', 'target', 'weight'
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
                # If the weight exists in pre_train, plot it
                indices = np.where(idc_syn_pre)[0]
                if indices.size > 0:
                    initial_weight = weights_pre_train[label]["weight"][indices[0]]
                else:
                    initial_weight = np.nan
                times = [0.0] + events["times"][idc_syn].tolist()
                weights = [initial_weight] + events["weights"][idc_syn].tolist()
                ax.step(times, weights, c=colors.get("blue", "#1f77b4"))
        ax.set_ylabel(ylabel)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    # rec_rec weights: sender and target both in nrns_rec
    plot_weight_time_course(
        axs[0],
        events_wr,
        nrns_rec[:n_record_w],
        nrns_rec[:n_record_w],
        "rec_rec",
        r"$W_\text{rec}$ (pA)",
    )
    # rec_out weights: sender in nrns_rec, target in nrns_out
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
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_weight_matrices(weights_pre_train, weights_post_train, colors, out_path):
    """
    Plot the initial and final weight matrices for recurrent and output connections.
    This function efficiently reconstructs dense weight matrices from sparse connection data
    provided by `get_weights` and maintains the original 2x2 plot layout and all other
    plot configurations, including the colorbar's appearance and position.
    """

    def reconstruct_weight_matrix(conns_data):
        if not conns_data or not conns_data["source"].size:
            if "len_target" in conns_data and "len_source" in conns_data:
                 return np.zeros((conns_data["len_target"], conns_data["len_source"]))
            return np.array([[]])

        num_post = conns_data["len_target"]
        num_pre = conns_data["len_source"]

        if num_post == 0 or num_pre == 0:
            return np.array([[]])

        weight_matrix = np.zeros((num_post, num_pre))

        target_indices = conns_data["target"].astype(int)
        source_indices = conns_data["source"].astype(int)
        
        weight_matrix[target_indices, source_indices] = conns_data["weight"]
        return weight_matrix

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "cmap", ((0.0, colors["red"]), (0.5, colors["white"]), (1.0, colors["blue"]))
    )

    fig, axs = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(10, 8))
    all_w_extrema = []
    for k in weights_pre_train.keys():
        w_pre = weights_pre_train[k]["weight"]
        w_post = weights_post_train[k]["weight"]
        all_w_extrema.append(
            [np.min(w_pre), np.max(w_pre), np.min(w_post), np.max(w_post)]
        )
    vmin = np.min(all_w_extrema)
    vmax = np.max(all_w_extrema)
    norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    args = {"cmap": cmap, "norm": norm}

    for i, weights in zip([0, 1], [weights_pre_train, weights_post_train]):
        weights["rec_rec"]["weight_matrix"] = reconstruct_weight_matrix(
            weights["rec_rec"]
        )
        weights["rec_out"]["weight_matrix"] = reconstruct_weight_matrix(
            weights["rec_out"]
        )
        axs[0, i].pcolormesh(weights["rec_rec"]["weight_matrix"], **args)
        cmesh = axs[1, i].pcolormesh(weights["rec_out"]["weight_matrix"], **args)
        axs[1, i].set_xlabel("recurrent\nneurons")
    axs[0, 0].set_ylabel("recurrent\nneurons")
    axs[1, 0].set_ylabel("readout\nneurons")
    fig.align_ylabels(axs[:, 0])
    axs[0, 0].text(0.5, 1.1, "pre-training", transform=axs[0, 0].transAxes, ha="center")
    axs[0, 1].text(
        0.5, 1.1, "post-training", transform=axs[0, 1].transAxes, ha="center"
    )
    axs[1, 0].yaxis.get_major_locator().set_params(integer=True)
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.20, 0.02, 0.6])
    cbar = plt.colorbar(cmesh, cax=cbar_ax, label="weight (pA)")
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def tutorial_plot_trajectories_and_targets(trajectory_files, target_files, params, save_path=None):
    """
    Plot joint angle trajectories and corresponding target spike signals for tutorial visualization.

    This function is specific to the motor-controller SNN tutorial.
    It displays the input trajectories and expected output spike rates ("pos" and "neg" channels) used by the model.

    Parameters
    ----------
    trajectory_files : list of str
        List of file paths to joint angle trajectory data.
    target_files : list of str
        List of file paths to target spike raster data.
    params : dict
        Dictionary of simulation/encoding parameters (may be used for plot annotation).
    save_path : str, optional
        If provided, saves the figure to this path.

    Returns
    -------
    None
        Shows the plot and saves it if save_path is specified.

    Example
    -------
    >>> tutorial_plot_trajectories_and_targets(trajectory_files, target_files, params)
    """

    n_trials = len(trajectory_files)
    duration_ms = 650
    num_bins = 650

    fig, axs = plt.subplots(2, n_trials, figsize=(5*n_trials, 6), sharex='col', 
                           gridspec_kw={'height_ratios': [1, 1]})

    all_trajectories = [np.loadtxt(traj_path) for traj_path in trajectory_files]
    global_min = min(traj.min() for traj in all_trajectories)
    global_max = max(traj.max() for traj in all_trajectories)

    for i in range(n_trials):
        trajectory = all_trajectories[i]
        time_traj = np.linspace(0, duration_ms, len(trajectory))
        axs[0, i].plot(time_traj, trajectory, color='tab:blue')
        axs[0, i].set_title(f"Trajectory {i+1}")
        axs[0, i].set_ylim(global_min, global_max)
        axs[0, i].grid(True, linestyle='--', alpha=0.3)
        axs[0, i].set_xlabel("Time (ms)")
        if i == 0:
            axs[0, i].set_ylabel("Joint Angle (rad)")
        else:
            axs[0, i].set_yticklabels([])

        tgt_path = target_files[i]
        with open(tgt_path, "r") as f:
            first_line = f.readline()
        delimiter = "," if "," in first_line else None
        target_spikes = np.loadtxt(tgt_path, delimiter=delimiter)
        if target_spikes.ndim == 1:
            target_spikes = target_spikes.reshape((1, -1))
        pos_spikes = target_spikes[target_spikes[:,0] <= 50, 1]
        neg_spikes = target_spikes[target_spikes[:,0] > 50, 1]
        pos_hist, bin_edges = np.histogram(pos_spikes, bins=num_bins, range=(0, duration_ms))
        neg_hist, _ = np.histogram(neg_spikes, bins=num_bins, range=(0, duration_ms))
        axs[1, i].plot(bin_edges[:-1], pos_hist, color='tab:blue', label='pos')
        axs[1, i].plot(bin_edges[:-1], neg_hist, color='tab:red', label='neg')
        axs[1, i].set_title(f"Target Signal {i+1}")
        axs[1, i].set_xlabel("Time (ms)")
        if i == 0:
            axs[1, i].set_ylabel("Target Spike Rate")
        else:
            axs[1, i].set_yticklabels([])
        axs[1, i].grid(True, linestyle='--', alpha=0.3)
        axs[1, i].legend()

    plt.suptitle("Input Trajectories and Corresponding Target Spike Signals (pos/neg)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        fig.savefig(save_path)
    plt.show()


def tutorial_plot_spike_raster(spikes, nrns_rec, xlims_list, save_path=None):
    """
    Plot spike raster for recurrent neuron activity before and after training in tutorial.

    This function is tailored for the SNN motor-controller tutorial, comparing population activity pre- and post-learning.

    Parameters
    ----------
    spikes : dict
        Dictionary containing spike 'senders' (neuron IDs) and 'times' (ms).
    nrns_rec : array-like
        List or array of recurrent neuron IDs (NEST GIDs).
    xlims_list : list of tuple
        List with two (start, stop) tuples, for pre- and post-training time windows.
    save_path : str, optional
        If provided, saves the figure to this path.

    Returns
    -------
    None
        Shows the plot and saves it if save_path is specified.

    Example
    -------
    >>> tutorial_plot_spike_raster(results['spikes'], results['nrns_rec'], xlims_list)
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for i, xlims in enumerate(xlims_list):
        idc_times = (spikes['times'] > xlims[0]) & (spikes['times'] < xlims[1])
        idc_sender = np.isin(spikes['senders'][idc_times], nrns_rec)
        senders_subset = spikes['senders'][idc_times][idc_sender]
        times_subset = spikes['times'][idc_times][idc_sender]
        min_gid = nrns_rec[0].global_id
        max_gid = nrns_rec[-1].global_id
        margin = abs(max_gid - min_gid) * 0.1 + 1
        axs[i].scatter(times_subset, senders_subset, s=2, color='black', alpha=0.7)
        axs[i].set_ylim(min_gid - margin, max_gid + margin)
        axs[i].set_xlim(xlims)
        axs[i].set_xlabel('Time (ms)')
        axs[i].set_title('Pre-training window' if i == 0 else 'Post-training window')
    axs[0].set_ylabel('Neuron ID')
    for ax in axs:
        ax.grid(True, linestyle='--', alpha=0.3)
    plt.suptitle('Spike Raster Plot: Pre- and Post-Training')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.show()


def tutorial_plot_output_vs_target(output_mm, xlims_list, save_path=None):
    """
    Plot network output and error versus the target signal for motor command channels ("pos"/"neg") in tutorial.

    This figure is specific to the motor-controller SNN tutorial, used to assess learning progress and output accuracy.

    Parameters
    ----------
    output_mm : dict
        Dictionary with keys 'senders', 'readout_signal', 'target_signal', 'times'.
    xlims_list : list of tuple
        List with two (start, stop) tuples for pre- and post-training windows.
    save_path : str, optional
        If provided, saves the figure to this path.

    Returns
    -------
    None
        Shows the plot and saves if save_path is specified.

    Example
    -------
    >>> tutorial_plot_output_vs_target(results['output_multimeter'], xlims_list)
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    senders = output_mm['senders']
    readout = output_mm['readout_signal']
    target = output_mm['target_signal']
    times = output_mm['times']

    unique_gids = np.unique(senders)
    labels = {gid: 'pos' if i == 0 else 'neg' for i, gid in enumerate(unique_gids)}
    colors = {gid: 'tab:blue' if i == 0 else 'tab:red' for i, gid in enumerate(unique_gids)}

    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex='col', gridspec_kw={'height_ratios': [2, 1]})

    for col, xlims in enumerate(xlims_list):
        for gid in unique_gids:
            mask = ((senders == gid) & (times > xlims[0]) & (times < xlims[1]))
            label = labels[gid]
            color = colors[gid]
            axs[0, col].plot(times[mask], readout[mask], color=color, label=f'{label} output')
            axs[1, col].plot(times[mask], readout[mask] - target[mask], color=color, label=f'{label} error')
        axs[0, col].set_title('Pre-training window' if col == 0 else 'Post-training window')
        axs[1, col].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axs[1, col].set_xlabel('Time (ms)')
        ax_inset = inset_axes(axs[0, col], width="35%", height="30%", loc='upper left', borderpad=2)
        for gid in unique_gids:
            mask = ((senders == gid) & (times > xlims[0]) & (times < xlims[1]))
            label = labels[gid]
            color = colors[gid]
            ax_inset.plot(times[mask], target[mask], color=color, alpha=0.6, linewidth=2, label=f'{label} target')
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_title("Target", fontsize=10)

    axs[0, 0].set_ylabel('Output')
    axs[1, 0].set_ylabel('Error')
    for row in range(2):
        axs[row, 0].legend(loc='upper right')
        axs[row, 1].legend(loc='upper right')
    for ax_row in axs:
        for ax in ax_row:
            ax.grid(True, linestyle='--', alpha=0.3)
    plt.suptitle('Network Output and Error vs Target (shown as inset)', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    if save_path:
        fig.savefig(save_path)
    plt.show()


def tutorial_plot_loss_curve(loss, save_path=None):
    """
    Plot training loss curve for the motor-controller SNN tutorial.

    Shows the mean squared error (MSE) loss across training iterations, indicating learning progress.

    Parameters
    ----------
    loss : array-like
        Sequence of loss values per training iteration.
    save_path : str, optional
        If provided, saves the figure to this path.

    Returns
    -------
    None
        Shows the plot and saves if save_path is specified.

    Example
    -------
    >>> tutorial_plot_loss_curve(results['loss'])
    """
    plt.figure(figsize=(8,4))
    plt.plot(np.arange(1, len(loss)+1), loss)
    plt.xlabel('Training iteration')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss Curve')
    plt.grid(True, linestyle='--', alpha=0.3)
    if save_path:
        plt.savefig(save_path)
    plt.show()