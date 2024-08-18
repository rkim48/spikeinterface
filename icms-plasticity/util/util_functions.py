import seaborn as sns
import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from spikeinterface import full as si


def get_non_stim_periods(all_stim_timestamps):
    # returns tuple of non-stim intervals
    non_stim_periods = []
    last_end = None
    for train in all_stim_timestamps:
        if not train:
            continue
        if last_end is not None:
            start_current_train = train[0]
            non_stim_periods.append((last_end, start_current_train))
        last_end = train[-1]
    return non_stim_periods


def weighted_choice(choices, weights):
    return random.choices(choices, weights=weights, k=1)[0]


def sample_windows(non_stim_periods, num_windows, window_size_ms=140, fs=30000):
    # set window size to be 140 (two consecutive 70 ms windows to compare)
    sampled_windows = []
    window_size = window_size_ms * fs / 1000  # Convert ms to seconds if needed
    # Calculate weights based on the length of each non-stimulated period
    weights = [end - start for start, end in non_stim_periods]
    while len(sampled_windows) < num_windows:
        # Randomly select a non-stimulated period
        start_period, end_period = weighted_choice(non_stim_periods, weights)
        # Check if the chosen period can accommodate the window size
        if (end_period - start_period) >= window_size:
            start_window = random.uniform(
                start_period, end_period - window_size)
            end_window = start_window + window_size
            sampled_windows.append((start_window, end_window))
    return sampled_windows


def create_null_distribution(unit_spike_times, non_stim_periods, pulse_win=10, post_stim_win=2.5, pre_stim_win=0.5, num_windows=100, fs=30000):
    total_window_size_ms = 1400  # Total window size of 1.4 seconds
    single_window_size_ms = 700  # Each window is 0.7 seconds
    total_window_size_samples = int(total_window_size_ms * fs / 1000)
    pulse_period_samples = int(pulse_win * fs / 1000)
    post_blank_samples = int(post_stim_win * fs / 1000)
    pre_blank_samples = int(pre_stim_win * fs / 1000)

    sampled_windows = sample_windows(
        non_stim_periods, num_windows, total_window_size_ms)
    null_distribution = []
    window_A_spikes_all = []
    window_B_spikes_all = []
    # Convert unit_spike_times to a numpy array for vectorized operations
    unit_spike_times = np.array(unit_spike_times)

    for start_window, _ in sampled_windows:
        start_window = int(start_window)
        mid_point = start_window + int(single_window_size_ms * fs / 1000)

        for pulse_start in range(start_window, start_window + total_window_size_samples, pulse_period_samples):
            valid_t0 = pulse_start + post_blank_samples
            pulse_end = pulse_start + pulse_period_samples
            valid_t1 = min(pulse_end - pre_blank_samples,
                           start_window + total_window_size_samples)

            # Collect spikes outside the blanking intervals
            valid_spikes = unit_spike_times[(
                unit_spike_times > valid_t0) & (unit_spike_times < valid_t1)]
            window_spikes_relative = (valid_spikes - start_window) * 1000 / fs

            # Assign spikes to Window A or B using vectorized approach
            window_A_spikes = window_spikes_relative[window_spikes_relative <
                                                     single_window_size_ms]
            window_B_spikes = window_spikes_relative[window_spikes_relative >=
                                                     single_window_size_ms]

            window_A_spikes_all.append(window_A_spikes)
            window_B_spikes_all.append(window_B_spikes)

            firing_rate_A = len(window_A_spikes) / \
                (single_window_size_ms / 1000.0)
            firing_rate_B = len(window_B_spikes) / \
                (single_window_size_ms / 1000.0)
            null_distribution.append(firing_rate_B - firing_rate_A)

    return sampled_windows, window_A_spikes_all, window_B_spikes_all, null_distribution


def plot_consecutive_window_raster(sampled_windows, window_A_spikes_all, window_B_spikes_all, pulse_win=10, post_stim_win=2.5, pre_stim_win=0.5, fs=30000):
    plt.figure(figsize=(15, 8))
    total_window_ms = 1400  # Total window size of 1.4 seconds in milliseconds
    half_window_ms = 700   # Half of the window is 0.7 seconds in milliseconds
    num_trials = len(sampled_windows)

    # Combine all spikes for Window A and Window B into single arrays
    all_spikes_A = np.concatenate([np.array(spikes)
                                  for spikes in window_A_spikes_all])
    all_trials_A = np.concatenate(
        [np.full(len(spikes), idx) for idx, spikes in enumerate(window_A_spikes_all)])
    all_spikes_B = np.concatenate(
        [np.array(spikes) for spikes in window_B_spikes_all])
    all_trials_B = np.concatenate(
        [np.full(len(spikes), idx) for idx, spikes in enumerate(window_B_spikes_all)])

    # Plot all spikes for Window A and Window B
    plt.scatter(all_spikes_A, all_trials_A, color='k', s=6, marker='|')
    plt.scatter(all_spikes_B, all_trials_B, color='k', s=6, marker='|')

    # Add red shading for blanking periods
    # for pulse_start_ms in np.arange(0, total_window_ms, pulse_win):
    #     # Post-pulse blanking period (0 to 2.5 ms after each pulse)
    #     rect_post = patches.Rectangle(
    #         (pulse_start_ms, -0.5), post_stim_win, num_trials, color='red', alpha=0.3)
    #     plt.gca().add_patch(rect_post)

    #     # Pre-next pulse blanking period (9.5 to 10 ms before the next pulse)
    #     # Skip shading for the pulse before 700 ms and 1400 ms
    #     if pulse_start_ms in [690, 1390]:
    #         continue

    #     pre_blank_start = pulse_start_ms + pulse_win - pre_stim_win
    #     if pre_blank_start < total_window_ms:
    #         rect_pre = patches.Rectangle(
    #             (pre_blank_start, -0.5), pre_stim_win, num_trials, color='red', alpha=0.3)
    #         plt.gca().add_patch(rect_pre)
    plt.axvline(half_window_ms, color='r')
    plt.xlabel('Time (ms)')
    plt.ylabel('Trial')
    plt.title('Consecutive Window Raster Plot with Blanking')
    plt.xlim(0, total_window_ms)
    plt.show()


def group_stim_pulses_into_trains(stim_times, fs=30000):
    assert(stim_times == sorted(stim_times))
    diff = np.diff(stim_times)
    threshold = fs/100 + 10  # stim period (100 Hz => 10 ms)
    train_boundaries = np.where(diff > threshold)[0]
    trains = []
    start_idx = 0
    for boundary in train_boundaries:
        trains.append(stim_times[start_idx:boundary + 1])
        start_idx = boundary + 1
    trains.append(stim_times[start_idx:])

    return trains


def get_stim_spike_train(stim_times, unit_spike_times, pulse_win=10, pre_post_win=700, fs=30000, post_stim_win=2.5, pre_stim_win=0.5):

    trains = group_stim_pulses_into_trains(stim_times)
    pre_post_win_samples = int(pre_post_win * fs / 1000)
    pulse_win_samples = int(pulse_win * fs / 1000)
    post_blank_samples = int(post_stim_win * fs / 1000)
    pre_blank_samples = int(pre_stim_win * fs / 1000)

    stim_trial_data = []
    firing_rate_changes = []
    stim_spike_index_data = []

    for train in trains:
        # 0.7 seconds before the first pulse
        pre_stim_start = int(train[0] - pre_post_win_samples)
        pre_stim_end = int(train[0])

        # Create an array of start and end times for blanking intervals
        pre_stim_intervals = np.arange(
            pre_stim_start, pre_stim_end, pulse_win_samples)
        valid_pre_stim_starts = pre_stim_intervals + post_blank_samples
        valid_pre_stim_ends = pre_stim_intervals + pulse_win_samples - pre_blank_samples

        # Use broadcasting to compare unit_spike_times with all intervals
        valid_pre_stim_spikes = (unit_spike_times[:, None] > valid_pre_stim_starts) & (
            unit_spike_times[:, None] < valid_pre_stim_ends)
        pre_stim_spikes = unit_spike_times[np.any(
            valid_pre_stim_spikes, axis=1)]

        stim_spike_bool = (unit_spike_times >= train[0]) & (
            unit_spike_times < train[-1] + pulse_win_samples)
        stim_spikes = unit_spike_times[stim_spike_bool]
        stim_spike_indices = np.where(stim_spike_bool)[0]

        # Calculate firing rates
        firing_rate_pre_stim = len(pre_stim_spikes) / (pre_post_win / 1000)
        firing_rate_stim = len(stim_spikes) / (len(train) * pulse_win / 1000.0)

        firing_rate_change = firing_rate_stim - firing_rate_pre_stim
        firing_rate_changes.append(firing_rate_change)

        train_spike_times = np.concatenate(
            [pre_stim_spikes, stim_spikes]) - pre_stim_start
        stim_trial_data.append(train_spike_times / fs * 1000)
        stim_spike_index_data.extend(stim_spike_indices)

    return stim_trial_data, firing_rate_changes, stim_spike_index_data


def plot_stim_and_prestim_raster(stim_trial_data, pre_post_win=700, pulse_win=10, post_stim_win=2.5, pre_stim_win=0.5, fs=30000):
    # plt.figure(figsize=(15, 8))
    # num_trials = len(stim_trial_data)
    total_window_ms = pre_post_win * 2  # Total window size: pre-stim + stim

    # all_spikes = np.concatenate([np.array(spikes) / fs * 1000
    #                             for spikes in stim_trial_data])
    # all_trials = np.concatenate([np.full(len(spikes), idx)
    #                             for idx, spikes in enumerate(stim_trial_data)])
    # plt.scatter(all_spikes, all_trials, color='k', s=6, marker='|')
    plt.eventplot(stim_trial_data, orientation='horizontal')
    # Add red shading for blanking periods
    # for pulse_start_ms in np.arange(0, total_window_ms, pulse_win):
    #     # Post-pulse blanking period
    #     rect_post = patches.Rectangle(
    #         (pulse_start_ms, -0.5), post_stim_win, num_trials, color='red', alpha=0.3)
    #     plt.gca().add_patch(rect_post)

    #     # Pre-next pulse blanking period
    #     if (pulse_start_ms + pulse_win) % pre_post_win != 0:  # Avoiding the end of each window
    #         pre_blank_start = pulse_start_ms + pulse_win - pre_stim_win
    #         rect_pre = patches.Rectangle(
    #             (pre_blank_start, -0.5), pre_stim_win, num_trials, color='red', alpha=0.3)
    #         plt.gca().add_patch(rect_pre)

    plt.axvline(pre_post_win, color='r')

    plt.xlabel('Time (ms)')
    plt.ylabel('Trial')
    plt.title('Pre-Stimulus and Stimulation Train Raster Plot with Blanking')
    plt.xlim(0, total_window_ms)
    plt.show()


def get_unit_distance_from_stim_ch(we, stim_param, unit_id):
    # sort channel locations
    unsorted = we.get_channel_locations()
    sorted_ch_loc = unsorted[(-unsorted[:, 1]).argsort()]
    stim_channel_depth_idx = stim_param[1]
    stim_channel_location = sorted_ch_loc[stim_channel_depth_idx]
    unit_locations = si.compute_unit_locations(we)
    unit_ids = we.unit_ids  # exclude first column of zeros
    idx = np.where(unit_ids == unit_id)
    unit_location = unit_locations[idx][0][1:]
    distance = np.linalg.norm(stim_channel_location - unit_location)
    return distance


def get_modulation_score(unit_spike_times, stim_times, all_stim_timestamps):
    non_stim_periods = get_non_stim_periods(all_stim_timestamps)
    sampled_windows, window_A_spikes_all, window_B_spikes_all, null_distribution = create_null_distribution(
        unit_spike_times, non_stim_periods)

    stim_trial_data, firing_rate_changes, _ = get_stim_spike_train(
        stim_times, unit_spike_times)
    mean_delta_fr = np.mean(firing_rate_changes)

    null_mean = np.mean(null_distribution)
    null_std = np.std(null_distribution)
    stim_delta_mean = np.mean(firing_rate_changes)
    # compare firing rate changes (multiple values) with scalar null dist mean
    t_statistic, p_value = stats.ttest_1samp(firing_rate_changes, null_mean)
    z_score = (stim_delta_mean - null_mean) / null_std
    return t_statistic, p_value, z_score


def plot_prestim_stim_poststim_rasters(stim_times, unit_spike_times, plot_color='C0', no_axes=False, pulse_win=10, pre_post_win=700, fs=30000):
    trains = group_stim_pulses_into_trains(stim_times)
    pulse_win_samples = int(pulse_win * fs / 1000)
    pre_post_win_samples = int(pre_post_win * fs / 1000)
    trial_data = []
    for train in trains:
        pre_stim_start = int(train[0] - pre_post_win_samples)
        post_stim_end = int(
            train[-1] + pulse_win_samples + pre_post_win_samples)
        trial_spike_times = unit_spike_times[(unit_spike_times >= pre_stim_start) & (
            unit_spike_times < post_stim_end)]
        # Convert to milliseconds
        trial_spike_times_ms = (trial_spike_times - pre_stim_start) / fs * 1000
        trial_data.append(trial_spike_times_ms)

    plt.eventplot(trial_data, orientation='horizontal',
                  colors=plot_color, linewidths=2)
    if no_axes:
        plt.axis('off')
        scale_bar_y = len(trial_data) + 0.5
        plt.plot([700, 1400],
                 [scale_bar_y, scale_bar_y], color='k', lw=2)
        plt.xlim([0, 2100])
    plt.show()
    return trial_data


def plot_stim_train_waveforms(we, unit_id, primary_ch_idx, stim_spike_index_data, plot_color='C0', no_axes=False, fs=30000, ax=None):
    scale_factor = we.recording.get_channel_gains()[0]
    stim_wvfs = we.get_waveforms(unit_id=unit_id) * scale_factor
    mean_waveform = np.mean(
        stim_wvfs[stim_spike_index_data, :, primary_ch_idx], axis=0)
    std_waveform = np.std(
        stim_wvfs[stim_spike_index_data, :, primary_ch_idx], axis=0)

    # Define the time range (0.5 ms to 2 ms)
    time_range_start = 0.5  # in ms
    time_range_end = 2  # in ms

    # Samples corresponding to the time range
    start_sample = int(time_range_start * fs / 1000)
    end_sample = int(time_range_end * fs / 1000)

    # Slice the mean waveform and std to the specified time range
    mean_waveform = mean_waveform[start_sample:end_sample]
    std_waveform = std_waveform[start_sample:end_sample]

    time_vector = np.linspace(
        time_range_start, time_range_end, end_sample - start_sample)
    if ax is None:
        ax = plt.gca()

    ax.plot(time_vector, mean_waveform, color=plot_color, linewidth=2)
    ax.fill_between(time_vector, mean_waveform - std_waveform,
                    mean_waveform + std_waveform, color=plot_color, alpha=0.3)

    time_bar_length = 0.5  # in ms
    voltage_bar_length = 100  # in µV
    time_bar_start_x = 0.6  # Start slightly after 0.5 ms for clarity
    voltage_bar_start_y = 1.3 * min(mean_waveform)  # Offset from the bottom

    # Draw scale bars
    ax.plot([time_bar_start_x, time_bar_start_x + time_bar_length],
            [voltage_bar_start_y, voltage_bar_start_y], color='k', lw=2)
    ax.plot([time_bar_start_x, time_bar_start_x], [voltage_bar_start_y,
                                                   voltage_bar_start_y + voltage_bar_length], color='k', lw=2)

    # Annotate scale bars
    # plt.text(time_bar_start_x + time_bar_length / 2,
    #          voltage_bar_start_y * 1.15, '0.5ms', ha='center')
    # plt.text(time_bar_start_x * 0.7, voltage_bar_start_y +
    #          voltage_bar_length / 2, '100µV', va='center', rotation='vertical')
    if no_axes:
        ax.axis('off')


def compute_smoothed_firing_rate(trial_data, bin_size=10, sigma=30):
    """
    Computes smoothed firing rate from spike times using a Gaussian kernel.

    Parameters:
    - trial_data : list of lists of spike times in ms where each nested list is a trial
    - bin_size : size of time bins in ms
    - sigma : standard deviation of the Gaussian kernel in ms

    Returns:
    - smoothed_rate : smoothed firing rate in Hz (spikes/s)
    - times : corresponding time points
    """
    # Flatten the list of lists of spike times
    all_spikes = [spike for trial in trial_data for spike in trial]

    if not all_spikes:  # Check if all_spikes is empty
        return [], []

    num_trials = len(trial_data)  # Number of trials

    # Adjust the spike time range to account for the Gaussian kernel
    kernel_extent = 3 * sigma
    hist_range = (min(all_spikes) - kernel_extent,
                  max(all_spikes) + kernel_extent)

    # Create a histogram of the spikes
    hist, bins = np.histogram(all_spikes, bins=np.arange(
        hist_range[0], hist_range[1] + bin_size, bin_size))
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Normalize histogram to get average spike count per trial per bin
    hist = hist / num_trials

    # Convert average spike count per bin to firing rate in Hz (spikes/s)
    hist = hist / (bin_size / 1000)  # Convert bin size to seconds and divide

    # Create a Gaussian kernel
    kernel_range = np.arange(-3 * sigma, 3 * sigma + bin_size, bin_size)
    kernel = np.exp(-kernel_range**2 / (2 * sigma**2))
    kernel /= sum(kernel)

    # Convolve the spike histogram with the Gaussian kernel
    smoothed_rate_full = np.convolve(hist, kernel, mode='full')

    # Extract the 'valid' portion of the convolution output
    valid_start_idx = len(kernel) // 2
    valid_end_idx = valid_start_idx + len(hist)
    smoothed_rate = smoothed_rate_full[valid_start_idx:valid_end_idx]

    return smoothed_rate, bin_centers


def plot_smoothed_firing_rate(trial_data, bin_size, sigma, plot_color='C0', ax=None):
    # Assuming compute_smoothed_firing_rate is a predefined function
    smoothed_rate, times = compute_smoothed_firing_rate(
        trial_data, bin_size=bin_size, sigma=sigma)
    if ax is None:
        ax = plt.gca()
    ax.plot(times, smoothed_rate, color=plot_color, linewidth=2)
    ax.set_xlim([0, 2100])
    ax.set_ylim([0, 100])


def plot_raster_and_stim_wvf(we, unit_id, stim_times, unit_spike_times, stim_spike_index_data, primary_ch_idx, plot_color='C0'):
    # Create a figure
    fig = plt.figure(figsize=(6, 4))
    # Define the GridSpec layout
    gs = gridspec.GridSpec(3, 4, figure=fig)
    ax1 = fig.add_subplot(gs[:, 0:3])
    plot_prestim_stim_poststim_rasters(
        stim_times, unit_spike_times, plot_color=plot_color, no_axes=True)
    ax2 = fig.add_subplot(gs[0:1, 3:4])
    plot_stim_train_waveforms(
        we, unit_id, primary_ch_idx, stim_spike_index_data, plot_color=plot_color, no_axes=True)
    plt.tight_layout()


def plot_raster_fr_stim_wvf(we, unit_id, stim_times, unit_spike_times, stim_spike_index_data, primary_ch_idx, plot_color='C0'):
    # Raster with fr plot below
    fig = plt.figure(figsize=(6, 4))

    gs = gridspec.GridSpec(4, 4, figure=fig)

    ax1 = fig.add_subplot(gs[0:2, 0:3])
    trial_data = plot_prestim_stim_poststim_rasters(
        stim_times, unit_spike_times, plot_color=plot_color, no_axes=True)
    ax2 = fig.add_subplot(gs[2:, 0:3])
    plot_smoothed_firing_rate(trial_data, bin_size=10,
                              sigma=20, plot_color=plot_color)
    ax3 = fig.add_subplot(gs[0:1, 3:4])
    plot_stim_train_waveforms(
        we, unit_id, primary_ch_idx, stim_spike_index_data, plot_color=plot_color, no_axes=True)
    plt.tight_layout()


def plot_joint_raster_fr_stim_wvf(we, unit_id, response, select_params, title):
    fig = plt.figure(figsize=(5, 7))

    num_currents = len(select_params)

    gs = gridspec.GridSpec(num_currents+1, 4, figure=fig)
    ax2 = fig.add_subplot(gs[num_currents, 0:3])
    unit_spike_times = response.all_spike_times
    primary_ch_idx = response.primary_ch_idx

    for i, stim_param in enumerate(select_params):
        palette = sns.color_palette("deep")
        plot_color = palette[i]
        response_data = response.get_response_data(stim_param)
        stim_times = response_data['stim_times']
        stim_trial_data, firing_rate_changes, stim_spike_index_data = get_stim_spike_train(
            stim_times, unit_spike_times)

        if type(response_data) == str:
            continue
        # plot raster
        ax1 = fig.add_subplot(gs[i, 0:3])
        trial_data = plot_prestim_stim_poststim_rasters(
            stim_times, unit_spike_times, plot_color=plot_color, no_axes=True)
        if i == 0:
            plt.title(title)
        # plot firing rate
        plot_smoothed_firing_rate(trial_data, bin_size=10,
                                  sigma=20, plot_color=plot_color, ax=ax2)
        # plot wvf
        ax3 = fig.add_subplot(gs[i, 3])
        plot_stim_train_waveforms(
            we, unit_id, primary_ch_idx, stim_spike_index_data, plot_color=plot_color, no_axes=True)
    plt.tight_layout()


def is_modulated(response, select_params):
    unit_spike_times = response.all_spike_times
    for i, stim_param in enumerate(select_params):
        response_data = response.get_response_data(stim_param)
        stim_times = response_data['stim_times']
        _, p_value, _ = get_modulation_score(
            unit_spike_times, stim_times, all_stim_timestamps)
        if p_value < 0.001:
            return True
    return False


def get_spike_times_and_indices(unit_spike_times, stim_times, fs=30000, post_stim_win=300):
    spike_times = []
    trial_indices = []
    for i, ts in enumerate(stim_times):
        spikes_in_window = (unit_spike_times[np.logical_and(
            unit_spike_times >= ts, unit_spike_times <= ts + post_stim_win)] - ts) * 1000.0 / fs
        spike_times.extend(spikes_in_window)
        # Repeat the trial index for each spike
        trial_indices.extend([i] * len(spikes_in_window))

    return spike_times, trial_indices


def get_spike_times_by_trial(unit_spike_times, stim_times, fs=30000, post_stim_win=300):
    trial_data = []
    for i, ts in enumerate(stim_times):
        spikes_in_window = (unit_spike_times[np.logical_and(
            unit_spike_times >= ts, unit_spike_times <= ts + post_stim_win)] - ts) * 1000.0 / fs
        # Convert to list and append to the list for each trial
        trial_data.append(spikes_in_window.tolist())
    return trial_data


def plot_stim_pulse_raster(unit_spike_times, stim_times, fs=30000, plot_color='C0'):
    spike_times, trial_indices = get_spike_times_and_indices(
        unit_spike_times, stim_times)
    plt.scatter(spike_times, trial_indices,
                color=plot_color, s=6, marker='|')  # 6
    plt.axvspan(0, 2.5, color='lightgray', alpha=0.5)
    plt.xlim([0, 10])


def plot_binned_spike_counts(unit_spike_times, stim_times, fs=30000, bin_width=0.5, plot_color='C0'):
    spike_times, trial_indices = get_spike_times_and_indices(
        unit_spike_times, stim_times)
    unique_trials = np.unique(trial_indices)
    bins = np.arange(0, 10.5, bin_width)
    binned_spike_counts, bin_edges = np.histogram(spike_times, bins=bins)
    # normalize by number of trials with spikes!
    prob_spike = binned_spike_counts / len(stim_times)
    bin_centers = bins[:-1] + bin_width/2
    bin_centers_blank = bin_centers[5:]
    prob_spike_blank = prob_spike[5:]
    plt.plot(bin_centers_blank, prob_spike_blank,
             color=plot_color, linewidth=2)
    plt.axvspan(0, 2.5, color='lightgray', alpha=0.5)
    plt.xlim([0, 10])
    return bin_centers_blank, prob_spike_blank


def calculate_phase_lock_index(unit_spike_times, stim_times, fs=30000, bin_width=0.5):
    spike_times, trial_indices = get_spike_times_and_indices(
        unit_spike_times, stim_times)
    unique_trials = np.unique(trial_indices)
    bins = np.arange(0, 10.5, bin_width)
    binned_spike_counts, bin_edges = np.histogram(spike_times, bins=bins)
    prob_spike = binned_spike_counts / len(stim_times)
    bin_centers = bins[:-1] + bin_width/2
    bin_centers_blank = bin_centers[5:]
    prob_spike_blank = prob_spike[5:]

    max_idx = np.argmax(prob_spike_blank)
    if max_idx == 0:
        max_and_adjoining = [0, 1]
    elif max_idx == len(prob_spike) - 1:
        max_and_adjoining = [len(prob_spike) - 2, len(prob_spike) - 1]
    else:
        max_and_adjoining = [max_idx - 1, max_idx, max_idx + 1]

    avg_spike_prob = np.mean(prob_spike_blank[max_and_adjoining])
    median_spike_prob = np.median(prob_spike_blank)
    phase_lock_index = avg_spike_prob - median_spike_prob
    return phase_lock_index


def custom_binning(spikes, bins):
    binned_counts = [0] * (len(bins) - 1)
    for spike in spikes:
        for i in range(len(bins) - 1):
            if bins[i] <= spike < bins[i + 1]:
                binned_counts[i] += 1
                break
    return binned_counts


def shuffle_bins_within_trials(spike_times, trial_indices, bin_width, blanking_bins=5):
    unique_trials = np.unique(trial_indices)
    shuffled_spike_times = []
    bins = np.arange(0, 10.5, bin_width)
    rng = np.random.default_rng()
    for trial in unique_trials:
        # Extract spike times for the current trial
        trial_spike_times = spike_times[trial_indices == trial]

        # Bin the spike times for the current trial
        binned = custom_binning(trial_spike_times, bins)

        # Shuffle the bins after the blanking period
        binned_after_blanking = binned[blanking_bins:]
        rng.shuffle(binned_after_blanking)
        binned[blanking_bins:] = binned_after_blanking

        # Reconstruct spike times from shuffled binned data
        trial_shuffled_spike_times = np.repeat(
            bins[:-1], binned) + bin_width / 2
        shuffled_spike_times.extend(trial_shuffled_spike_times)

    return np.array(shuffled_spike_times)


def calculate_phase_lock_index_from_binned_data(binned_shuffled_counts, sample_size, bin_width=0.5):
    prob_spike = binned_shuffled_counts / sample_size
    prob_spike_blank = prob_spike[5:]
    max_value = np.max(prob_spike_blank)
    max_indices = np.where(prob_spike_blank == max_value)[0]
    # Randomly select one index if there are multiple
    if len(max_indices) > 1:
        max_idx = random.choice(max_indices)
    else:
        max_idx = max_indices[0]

    if max_idx == 0:
        max_and_adjoining = [0, 1]
    elif max_idx == len(prob_spike_blank) - 1:
        max_and_adjoining = [len(prob_spike_blank) - 2,
                             len(prob_spike_blank) - 1]
    else:
        max_and_adjoining = [max_idx - 1, max_idx, max_idx + 1]

    avg_spike_prob = np.mean(prob_spike_blank[max_and_adjoining])
    median_spike_prob = np.median(prob_spike_blank)
    phase_lock_index = avg_spike_prob - median_spike_prob
    return phase_lock_index


def get_pli_null_distribution(unit_spike_times, stim_times, fs=30000, bin_width=0.5, num_iterations=5000):
    null_distribution = []
    spike_times, trial_indices = get_spike_times_and_indices(
        unit_spike_times, stim_times, fs)

    # we need to sample 20% from all trials!
    # Total number of trials, spiking and non-spiking
    total_trials = len(stim_times)
    sample_size = int(np.floor(0.2 * total_trials))  # 20% of total trials

    unique_trials = np.unique(trial_indices)  # trials with spikes
    bins = np.arange(0, 10.5, bin_width)
    if sample_size < 50:
        return None

    for _ in range(num_iterations):
        subsampled_spike_times = []
        subsampled_spike_trial_indices = []
        selected_trials = np.random.choice(
            np.arange(total_trials), sample_size, replace=False)

        for trial in selected_trials:
            # Check if the trial is one with spikes
            if trial in trial_indices:
                # Extract spike times for this trial
                trial_index = int(trial)
                spikes = spike_times[trial_indices == trial_index]
                spikes = np.atleast_1d(spikes)
                subsampled_spike_times.extend(spikes)
                subsampled_spike_trial_indices.extend(
                    [trial_index] * len(spikes))

        shuffled_spike_times = shuffle_bins_within_trials(
            np.array(subsampled_spike_times), np.array(subsampled_spike_trial_indices), bin_width)
        binned_shuffled_counts = np.array(
            custom_binning(shuffled_spike_times, bins))
        phase_lock_index = calculate_phase_lock_index_from_binned_data(
            binned_shuffled_counts, sample_size, bin_width)
        null_distribution.append(phase_lock_index)

    return null_distribution


def calculate_bootstrapped_stats(unit_spike_times, stim_times, fs=30000, bin_width=0.5, num_iterations=5000):
    spike_times, trial_indices = get_spike_times_and_indices(
        unit_spike_times, stim_times, fs)
    total_trials = len(stim_times)
    sample_size = int(np.floor(0.2 * total_trials))  # 20% of total trials

    unique_trials = np.unique(trial_indices)  # trials with spikes
    bins = np.arange(0, 10.5, bin_width)
    # if sample_size < 50:
    #     return None

    bins = np.arange(0, 10 + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width/2
    all_spike_probs = []
    latency_distributions = []

    for _ in range(num_iterations):
        subsampled_spike_times = []
        subsampled_spike_trial_indices = []
        selected_trials = np.random.choice(
            np.arange(total_trials), sample_size, replace=False)

        for trial in selected_trials:
            if trial in trial_indices:
                trial_index = int(trial)
                # print(trial_index)
                spikes = spike_times[trial_indices == trial_index]
                print(spikes)
                spikes = np.atleast_1d(spikes)
                subsampled_spike_times.extend(spikes)
                subsampled_spike_trial_indices.extend(
                    [trial_index] * len(spikes))

        binned_spike_counts = custom_binning(subsampled_spike_times, bins)
        spike_prob = np.array(binned_spike_counts) / sample_size
        all_spike_probs.append(spike_prob)

        max_spike_bin_idx = np.argmax(spike_prob)
        latency = (bins[max_spike_bin_idx] + bins[max_spike_bin_idx + 1]) / 2
        latency_distributions.append(latency)

    all_spike_probs = np.array(all_spike_probs)

    # Calculate mean and standard error for each bin across all iterations
    # Calculate mean and variance (jitter) of latencies
    bin_centers = bin_centers[5:]
    all_spike_probs = all_spike_probs[:, 5:]
    mean_spike_prob = np.mean(all_spike_probs, axis=0)
    se_spike_prob = np.std(all_spike_probs, axis=0) / np.sqrt(num_iterations)
    mean_latency = np.mean(latency_distributions)
    jitter = np.var(latency_distributions)

    return mean_spike_prob, se_spike_prob, mean_latency, jitter, bin_centers


def calculate_bootstrapped_stats(unit_spike_times, stim_times, fs=30000, bin_width=0.5, num_iterations=5000):
    spike_times, trial_indices = get_spike_times_and_indices(
        unit_spike_times, stim_times, fs)
    unique_trials = np.unique(trial_indices)
    sample_size = int(len(unique_trials) * 0.2)  # 20% of trials

    bins = np.arange(0, 10 + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width/2
    all_spike_probs = []
    latency_distributions = []

    for _ in range(num_iterations):
        selected_trials = np.random.choice(
            unique_trials, sample_size, replace=False)
        selected_spike_times = [spike for spike, trial in zip(
            spike_times, trial_indices) if trial in selected_trials]

        binned_spike_counts = custom_binning(selected_spike_times, bins)
        spike_prob = np.array(binned_spike_counts) / sample_size
        all_spike_probs.append(spike_prob)

        max_spike_bin_idx = np.argmax(spike_prob)
        latency = (bins[max_spike_bin_idx] + bins[max_spike_bin_idx + 1]) / 2
        latency_distributions.append(latency)

    all_spike_probs = np.array(all_spike_probs)

    # Calculate mean and standard error for each bin across all iterations
    # Calculate mean and variance (jitter) of latencies
    bin_centers = bin_centers[5:]
    all_spike_probs = all_spike_probs[:, 5:]
    mean_spike_prob = np.mean(all_spike_probs, axis=0)
    se_spike_prob = np.std(all_spike_probs, axis=0) / np.sqrt(num_iterations)
    mean_latency = np.mean(latency_distributions)
    jitter = np.var(latency_distributions)

    return mean_spike_prob, se_spike_prob, mean_latency, jitter, bin_centers


def calculate_bootstrapped_stats_for_baseline(unit_spike_times, stim_times, fs=30000, bin_width=0.5, num_iterations=5000):
    '''
    IN PROGRESS
    '''
    spike_times, trial_indices = get_spike_times_and_indices(
        unit_spike_times, stim_times, fs)
    unique_trials = np.unique(trial_indices)
    sample_size = int(len(unique_trials) * 0.2)  # 20% of trials

    bins = np.arange(0, 10 + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width/2
    all_spike_probs = []
    latency_distributions = []

    # sample


def plot_bootstrapped_spike_distribution(mean_spike_prob, se_spike_prob, bin_centers, plot_color='C0', no_axes=False):
    plt.plot(bin_centers, mean_spike_prob, color=plot_color, linewidth=2)
    plt.fill_between(bin_centers, mean_spike_prob - se_spike_prob,
                     mean_spike_prob + se_spike_prob, color=plot_color, alpha=0.3)
    plt.axvspan(0, 2.5, color='lightgray', alpha=0.5)
    plt.xlim([0, 10])
    plt.ylim(bottom=0)

    if no_axes:
        plt.axis('off')
        # Determine the range for x and y axes
        x_range = max(bin_centers) - min(bin_centers)
        y_range = max(mean_spike_prob + se_spike_prob) - \
            min(mean_spike_prob - se_spike_prob)

        # Calculate aspect ratio and adjust line widths
        aspect_ratio = y_range / x_range
        lw_x = 3  # Default line width for x
        lw_y = max(3, 3 * aspect_ratio)  # Ensure a minimum line width

        # Draw scale bars
        time_bar_length = 1  # in ms
        prob_bar_length = 0.1
        time_bar_start_x = 0
        prob_bar_start_y = 0

        plt.plot([time_bar_start_x, time_bar_start_x + time_bar_length],
                 [prob_bar_start_y, prob_bar_start_y], color='k', lw=lw_x)
        plt.plot([time_bar_start_x, time_bar_start_x], [
                 prob_bar_start_y, prob_bar_start_y + prob_bar_length], color='k', lw=lw_y)
