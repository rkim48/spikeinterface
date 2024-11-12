import batch_process.postprocessing.stim_response_util as stim_response_util
import numpy as np
import matplotlib.pyplot as plt


def get_baseline_interval_start_times(stim_timestamps, timing_params):

    trains = stim_response_util.group_stim_pulses_into_trains(stim_timestamps, timing_params)

    fs = timing_params["fs"]
    stim_duration_ms = timing_params["stim_duration_ms"]
    pulse_win_ms = timing_params["pulse_win_ms"]
    baseline_interval_start_times = []

    for train in trains:
        first_pulse_ts = train[0]
        # Get baseline window before first pulse of stim train
        # Baseline window size is same as stim train size
        baseline_start = int(first_pulse_ts - stim_duration_ms * fs / 1000)

        # Calculate the number of intervals in the baseline window
        num_intervals = int(stim_duration_ms / pulse_win_ms)
        for i in range(num_intervals):
            interval_start = int(baseline_start + i * pulse_win_ms * fs / 1000)

            baseline_interval_start_times.append(interval_start)
    return baseline_interval_start_times


def get_spike_prob_distribution(raster_array, timing_params, subsample_fraction=1):

    bin_width_ms = timing_params["bin_width_ms"]
    pulse_win_ms = timing_params["pulse_win_ms"]
    pre_stim_blank_ms = timing_params["pre_stim_blank_ms"]
    post_stim_blank_ms = timing_params["post_stim_blank_ms"]
    bins = np.arange(0, pulse_win_ms + bin_width_ms, bin_width_ms)

    total_trials = len(raster_array)
    num_subsampled_trials = int(total_trials * subsample_fraction)

    if subsample_fraction < 1.0:
        subsampled_trials = np.random.choice(total_trials, num_subsampled_trials, replace=False)
        subsampled_raster_array = [raster_array[i] for i in subsampled_trials]
    else:
        subsampled_raster_array = raster_array

    # Flatten the subsampled raster array to get spike times
    rel_spike_times = [ts for trial in subsampled_raster_array for ts in trial]

    binned_spike_counts, bin_edges = np.histogram(rel_spike_times, bins=bins)

    all_prob_spike = binned_spike_counts / num_subsampled_trials
    all_bin_centers = bins[:-1] + bin_width_ms / 2

    prob_spike = all_prob_spike.copy()
    prob_spike[(all_bin_centers < post_stim_blank_ms) | (all_bin_centers > (pulse_win_ms - pre_stim_blank_ms))] = 0

    # Insert two data points at post_stim_blank_ms: one with fr=0 and one with the actual fr
    insert_index = np.searchsorted(all_bin_centers, post_stim_blank_ms)
    bin_centers = np.insert(all_bin_centers, insert_index, [post_stim_blank_ms, post_stim_blank_ms])
    prob_spike = np.insert(prob_spike, insert_index, [0, prob_spike[insert_index]])

    return bin_centers, prob_spike, binned_spike_counts


def get_latency(bin_centers, prob_spike):
    latency = bin_centers[np.argmax(prob_spike)]
    return latency


def get_bootstrapped_metrics(raster_array, timing_params, subsample_fraction, num_iterations=1000, return_summary=True):
    bootstrapped_spike_probs = []
    bootstrapped_latencies = []
    for _ in range(num_iterations):
        bin_centers, prob_spike, _ = get_spike_prob_distribution(
            raster_array, timing_params, subsample_fraction=subsample_fraction
        )
        latency = get_latency(bin_centers, prob_spike)

        bootstrapped_spike_probs.append(prob_spike)
        bootstrapped_latencies.append(latency)

    bootstrapped_spike_probs = np.array(bootstrapped_spike_probs)
    bootstrapped_latencies = np.array(bootstrapped_latencies)

    if return_summary:
        mean_prob_spike = np.mean(bootstrapped_spike_probs, axis=0)
        std_prob_spike = np.std(bootstrapped_spike_probs, axis=0)
        mean_latency = np.mean(bootstrapped_latencies)
        std_latency = np.std(bootstrapped_latencies)
        return bin_centers, mean_prob_spike, std_prob_spike, mean_latency, std_latency
    else:
        return bin_centers, bootstrapped_spike_probs, bootstrapped_latencies


def calculate_pulse_locked_index(bin_centers, spike_prob_distribution, timing_params):
    pulse_win_ms = timing_params["pulse_win_ms"]
    pre_stim_blank_ms = timing_params["pre_stim_blank_ms"]
    post_stim_blank_ms = timing_params["post_stim_blank_ms"]

    # Determine valid indices that are outside the blanking periods
    valid_indices = (bin_centers >= post_stim_blank_ms) & (bin_centers <= (pulse_win_ms - pre_stim_blank_ms))

    # Filter out the spike probabilities in the valid region
    valid_spike_prob_distribution = spike_prob_distribution[valid_indices]
    valid_bin_centers = bin_centers[valid_indices]

    # Identify the index of the maximum spike probability within the valid region
    max_index = np.argmax(valid_spike_prob_distribution)

    # Safely compute the mean of the top 3 probabilities around the max_index within the valid region
    if max_index == 0:
        mean_top3 = np.mean(valid_spike_prob_distribution[max_index : max_index + 3])
    elif max_index == len(valid_spike_prob_distribution) - 1:
        mean_top3 = np.mean(valid_spike_prob_distribution[max_index - 2 : max_index + 1])
    else:
        mean_top3 = np.mean(valid_spike_prob_distribution[max_index - 1 : max_index + 2])

    # Calculate the median probability of the valid indices
    median_prob = np.median(valid_spike_prob_distribution)

    # Calculate the Pulse-Locked Index (PLI) as a ratio of max to median
    pulse_locked_index = mean_top3 - median_prob

    return pulse_locked_index


def get_null_distribution(
    raster_array,
    timing_params,
    num_iterations=1000,
    subsample_fraction=0.2,
    plot_flag=False,
):

    total_trials = len(raster_array)
    num_subsampled_trials = int(total_trials * subsample_fraction)

    if subsample_fraction < 1.0:
        subsampled_trials = np.random.choice(total_trials, num_subsampled_trials, replace=False)
        subsampled_raster_array = [raster_array[i] for i in subsampled_trials]
    else:
        subsampled_raster_array = raster_array

    # Flatten the subsampled raster array to get spike times
    num_spikes = sum(len(trial) for trial in subsampled_raster_array)

    phase_lock_indices = []

    for _ in range(num_iterations):
        # Generate random spike times from a uniform distribution within the valid time window
        random_spike_times = np.random.uniform(
            timing_params["post_stim_blank_ms"],
            timing_params["pulse_win_ms"] - timing_params["pre_stim_blank_ms"],
            num_spikes,
        )

        # Create a histogram of these random spike times
        bins = np.arange(
            0, timing_params["pulse_win_ms"] + timing_params["bin_width_ms"], timing_params["bin_width_ms"]
        )
        binned_spike_counts, _ = np.histogram(random_spike_times, bins=bins)

        # Convert the histogram counts into a spike probability distribution
        prob_spike = binned_spike_counts / len(subsampled_trials)
        bin_centers = bins[:-1] + timing_params["bin_width_ms"] / 2

        # Calculate the PLI for the uniform distribution
        phase_lock_index = calculate_pulse_locked_index(bin_centers, prob_spike, timing_params)
        phase_lock_indices.append(phase_lock_index)

    # Optionally plot the distribution of PLIs
    if plot_flag:
        plt.hist(phase_lock_indices, bins=30, alpha=0.7)
        plt.xlabel("Pulse-Locked Index (PLI)")
        plt.ylabel("Frequency")
        plt.title("Null Distribution of Pulse-Locked Index (PLI)")
        plt.show()

    return phase_lock_indices


def is_pulse_locked(stim_pli, phase_lock_indices):
    return stim_pli > np.percentile(phase_lock_indices, 99)


# if __name__ == "__main__":

#     rel_stim_spike_times, stim_trial_indices, _ = stim_response_util.get_relative_spike_data(
#         spike_times, stim_times, timing_params
#     )
#     stim_raster_array = stim_response_util.get_pulse_interval_raster_array(rel_stim_spike_times, stim_trial_indices)

#     baseline_interval_start_times = get_baseline_interval_start_times(stim_times, timing_params)
#     rel_baseline_spike_times, baseline_trial_indices, _ = stim_response_util.get_relative_spike_data(
#         spike_times, baseline_interval_start_times, timing_params
#     )
#     baseline_raster_array = stim_response_util.get_pulse_interval_raster_array(
#         rel_baseline_spike_times, baseline_trial_indices
#     )
