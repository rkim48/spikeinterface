import numpy as np
import matplotlib.pyplot as plt

from spikeinterface import full as si

from batch_process.util.curate_util import *

import warnings

# ignore OMP_NUM_THREADS memory leaks warning
warnings.filterwarnings("ignore")


def mean_norm_peak(dense_template, sparse_ch_indices, min_idx=30, range_offset=10):
    max_peaks = np.max(
        np.abs(dense_template[min_idx -
               range_offset: min_idx + range_offset + 1, :]),
        axis=0,
    )
    all_ch_indices = np.arange(dense_template.shape[1])
    # Exclude sparse template channels
    non_sparse_ch_indices = np.setdiff1d(all_ch_indices, sparse_ch_indices)
    max_peaks_non_sparse = max_peaks[non_sparse_ch_indices]
    template_peak = np.min(dense_template[min_idx, :])
    normalized_peaks_non_sparse = max_peaks_non_sparse / np.abs(template_peak)
    top_peaks = np.sort(normalized_peaks_non_sparse)[-5:]
    mnr = np.mean(top_peaks)

    return mnr


def exclude_artifact_subcluster(we, unit_id, subcluster_indices):
    # set 120 um sparsity radius to ignore more channels nearby primary channel
    # for mean_norm_peak calculation
    sparsity = si.compute_sparsity(we, method="radius", radius_um=120)
    unit_id_to_ch_ind = sparsity.unit_id_to_channel_indices
    sparse_ch_indices = unit_id_to_ch_ind[unit_id]
    we.sparsity = None

    wvfs = we.get_waveforms(unit_id)
    dense_template = np.mean(wvfs[subcluster_indices, :, :], axis=0)

    mnr = mean_norm_peak(dense_template, sparse_ch_indices)
    # set 60 um sparsity radius
    we.sparsity = si.compute_sparsity(we, method="radius", radius_um=60)
    # if greater than 0.25, exclude
    return mnr > 0.25


def calculate_snr(primary_ch_wvfs, min_idx, subcluster_indices=None, threshold=2.0):
    if subcluster_indices is not None:
        primary_ch_wvfs = primary_ch_wvfs[subcluster_indices, :]

    mean_primary_wvfs = np.mean(primary_ch_wvfs, axis=0)
    # plt.figure()
    # plt.plot(mean_primary_wvfs)
    std_at_valley = np.std(primary_ch_wvfs[:, min_idx], axis=0)
    mean_at_valley = abs(mean_primary_wvfs[min_idx])
    snr_valley = mean_at_valley / std_at_valley

    snr_good = snr_valley >= threshold
    return snr_valley, snr_good


def check_abs_max_idx(primary_template, min_idx):
    abs_max_idx = np.argmax(np.abs(primary_template))
    if not (min_idx - 1 <= abs_max_idx <= min_idx + 1):
        return False
    return True


def too_large_trough_to_peak(primary_template, trough_idx=30, max_duration=30):
    # Calculate the duration from trough to peak
    peak_idx = np.argmax(
        primary_template[trough_idx: trough_idx + 40]) + trough_idx
    trough_to_peak_duration = peak_idx - trough_idx
    return trough_to_peak_duration > max_duration


def bad_repolarization_time(primary_template, min_samples=2, max_samples=20):
    # TOO MANY FALSE NEGATIVES
    derivative = np.diff(primary_template)
    # Find the late positive peak (local maximum after the trough)
    late_positive_peak_idx = np.argmax(primary_template[31:]) + 31

    # Ensure late_positive_peak_idx is within bounds
    if late_positive_peak_idx >= len(derivative):
        return True, np.nan

    # Find the inflection point after the late positive peak
    falling_branch = derivative[late_positive_peak_idx:]

    # Check if falling_branch is empty
    if falling_branch.size == 0:
        return True, np.nan

    inflection_point_idx = np.argmin(falling_branch) + late_positive_peak_idx
    repolarization_time = inflection_point_idx - late_positive_peak_idx

    return not (min_samples <= repolarization_time <= max_samples), repolarization_time


def exceeds_negative_slope(primary_template, threshold=-5):
    slope = (primary_template[16] - primary_template[0]) / 16
    return slope < threshold


def remove_bad_waveforms(primary_ch_subc_wvfs, indices, threshold_percentage=95):
    # Calculate distance from the mean waveform to determine which waveforms are outliers
    primary_ch_seg_wvfs = primary_ch_subc_wvfs[indices, 20:42]
    mean_waveform = np.mean(primary_ch_seg_wvfs, axis=0)
    distances = np.linalg.norm(primary_ch_seg_wvfs - mean_waveform, axis=1)
    threshold_distance = np.percentile(distances, threshold_percentage)

    # Correctly map indices to mark which are good and bad
    is_good = distances < threshold_distance
    good_indices = indices[is_good]
    return good_indices


def isi_violation_percentage(spike_train, refractory_period=2e-3, threshold=2):
    isis = np.diff(spike_train) / 30000
    n_violations = np.sum(isis < refractory_period)
    total_spikes = len(spike_train)
    violation_percentage = (n_violations / total_spikes) * 100
    isi_good = violation_percentage < threshold
    return violation_percentage, isi_good


def accept_subcluster(we, unit_id, subcluster_indices, min_idx=30, max_iterations=5):
    original_indices = np.arange(len(subcluster_indices))

    template_ch_dict = get_template_ch(we)
    wvf_shape = we.get_template(unit_id).shape
    if wvf_shape[1] <= 3:
        primary_ch_idx = template_ch_dict[unit_id]["primary_ch_idx_dense"]
    else:
        primary_ch_idx = template_ch_dict[unit_id]["primary_ch_idx_sparse"]

    primary_ch_subc_wvfs = we.get_waveforms(unit_id=unit_id)[
        subcluster_indices, :, primary_ch_idx
    ]

    primary_ch_subc_template = np.mean(primary_ch_subc_wvfs, axis=0)

    spike_train = we.sorting.get_unit_spike_train(unit_id=unit_id)[
        subcluster_indices]

    if exclude_artifact_subcluster(we, unit_id, subcluster_indices):
        print("\tSubcluster classified as artifact.")
        return original_indices, [], "artifact"

    if not check_abs_max_idx(primary_ch_subc_template, min_idx):
        print("\tAbsolute max index out of range.")
        return original_indices, [], "max_idx_out_of_range"

    if too_large_trough_to_peak(primary_ch_subc_template):
        print("\tTrough to peak too large.")
        return original_indices, [], "large_trough_to_peak"

    # if bad_repolarization_time(primary_ch_subc_template):
    #     print('\tBad repolarization time.')
    #     return original_indices, [], 'bad_repolarization_time'

    if exceeds_negative_slope(primary_ch_subc_template):
        print("\tWaveform pre-trough exceeds negative slope threshold.")
        return original_indices, [], "negative_slope"

    current_indices = original_indices
    for iteration in range(max_iterations):
        snr_valley, snr_good = calculate_snr(
            primary_ch_subc_wvfs, min_idx, current_indices
        )
        violation_percentage, isi_good = isi_violation_percentage(
            spike_train[current_indices]
        )
        print(f"\tISI violation %: {violation_percentage:.2f}")

        if snr_good and isi_good:
            break

        if not snr_good or not isi_good:
            good_indices = remove_bad_waveforms(
                primary_ch_subc_wvfs, current_indices)
            current_indices = good_indices

        print(
            f"\tIteration {iteration}: SNR {
                snr_valley:.2f}, ISI Violation Rate {violation_percentage:.2f}%"
        )

    if not snr_good:
        print("\tSubcluster SNR too low.")
        return original_indices, [], "low_snr"

    if not isi_good:
        print("\tSubcluster ISI violation rate too high.")
        return original_indices, [], "isi_violation"

    if len(current_indices) <= 50:
        print("\tToo few spikes.")
        return original_indices, [], "too_few_spikes"

    print("\tSubcluster classified as accepted.")
    return current_indices, np.setdiff1d(original_indices, current_indices), "accept"


def post_hoc_check(we, unit_id, subcluster_indices, min_idx=30):
    original_indices = np.arange(len(subcluster_indices))

    template_ch_dict = get_template_ch(we)
    wvf_shape = we.get_template(unit_id).shape
    if wvf_shape[1] <= 3:
        primary_ch_idx = template_ch_dict[unit_id]["primary_ch_idx_dense"]
    else:
        primary_ch_idx = template_ch_dict[unit_id]["primary_ch_idx_sparse"]

    primary_ch_subc_wvfs = we.get_waveforms(unit_id=unit_id)[
        subcluster_indices, :, primary_ch_idx
    ]
    # plt.figure()
    # plt.plot(primary_ch_subc_wvfs.T)

    primary_ch_subc_template = np.mean(primary_ch_subc_wvfs, axis=0)

    spike_train = we.sorting.get_unit_spike_train(unit_id=unit_id)[
        subcluster_indices]

    snr_valley, snr_good = calculate_snr(
        primary_ch_subc_wvfs, min_idx, original_indices
    )

    if exclude_artifact_subcluster(we, unit_id, subcluster_indices):
        print("\tSubcluster classified as artifact.")
        return "artifact"

    if not abs_max_loc_at_valley(primary_ch_subc_template, min_idx):
        print("\tSubcluster template valley not abs max.")
        return "not abs max"

    if not snr_good:
        print("\tSubcluster SNR too low.")
        return "low SNR"

    violation_percentage, isi_good = isi_violation_percentage(
        spike_train[original_indices]
    )

    if not isi_good:
        print("\tSubcluster ISI violation rate too high.")
        print(f"\tISI violation %: {violation_percentage:.2f}")
        return "isi_bad"

    print("\tSubcluster classified as accepted.")
    return "accept"


# %% Viz


def plot_umap_subcluster(unit_id, x, y, final_labels, subcluster_ids):
    plt.figure()
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    label_colors = {
        label: colors[i % len(colors)] for i, label in enumerate(subcluster_ids)
    }
    cluster_colors = [label_colors[label] for label in final_labels]
    scatter = plt.scatter(x, y, c=cluster_colors, s=1, alpha=1)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Cluster {label}",
            markerfacecolor=color,
            markersize=10,
        )
        for label, color in label_colors.items()
    ]
    plt.legend(handles=handles, title="Clusters")
    plt.title(f"UMAP: unit {unit_id}")
    plt.show()
    plt.pause(0.001)


def plot_clustered_waveforms(
    we,
    unit_id,
    split_indices_list,
    template_labels,
    mean_norm_peaks=None,
    plot_mean_std=True,
    N=2,
    fig_width=6,
    fig_height=4,
):
    # Ignore label 0 noise events
    colors = ["k", "C0", "C1", "C2", "C3", "C4", "C5"]
    unique_labels, counts = np.unique(split_indices_list, return_counts=True)

    primary_cluster_index = np.argmax(counts)
    primary_cluster_label = unique_labels[primary_cluster_index]
    wvfs = we.get_waveforms(unit_id=unit_id)

    template_ch_dict = get_template_ch(we)

    if wvfs.shape[2] > 3:
        key = "primary_ch_idx_dense"
    else:
        key = "primary_ch_idx_sparse"

    primary_ch_idx = template_ch_dict[unit_id][key]

    num_cols = 3
    num_rows = (len(unique_labels) + num_cols - 1) // num_cols

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(fig_width, fig_height))
    axes = axes.flatten()

    for i, unique_label in enumerate(unique_labels):
        ax = axes[i]
        event_indices = np.where(split_indices_list == unique_label)[0]

        all_wvfs = wvfs[event_indices, :, primary_ch_idx]

        if plot_mean_std:
            mean_wvfs = np.mean(all_wvfs, axis=0)
            std_wvfs = np.std(all_wvfs, axis=0)
            time_points = np.arange(mean_wvfs.shape[0]) / 30
            ax.plot(
                time_points,
                mean_wvfs,
                label=f"Label {unique_label}",
                color=colors[i % len(colors)],
                linewidth=0.8,
            )

            ax.fill_between(
                time_points,
                mean_wvfs - std_wvfs,
                mean_wvfs + std_wvfs,
                color=colors[i % len(colors)],
                alpha=0.2,
            )
        else:
            for j in range(0, all_wvfs.shape[0], N):
                ax.plot(
                    all_wvfs[j, :],
                    color=colors[i % len(colors)],
                    alpha=0.2,
                    linewidth=0.5,
                )

        if unique_label == 0:
            title = "Label 0 (discarded waveforms)"
        else:
            title = f"Label {unique_label} ({template_labels[i-1]})"

        ax.set_title(f"{title}\n{len(event_indices)} waveforms")
        ax.set_ylim([-500, 300])

        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels([0, 1, 2, 3])

        if i == 0:
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude (uV)")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
    plt.pause(0.001)
