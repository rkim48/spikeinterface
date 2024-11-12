import numpy as np
import math
import pandas as pd
import os
import batch_process.util.file_util as file_util
import dataloader as dataloader_module
from collections import namedtuple
from scipy.stats import ttest_rel, ttest_1samp

StimData = namedtuple("StimData", ["timestamps", "currents", "depths"])


def get_stim_data(data_folder):
    """
    Retrieves the stimulus timestamps, currents, and depths for a given session,
    filtering out bad trials and using only the included blocks.

    Args:
        data_folder (str): Path to the data folder for the session.

    Returns:
        StimData: A namedtuple containing:
            - timestamps (list[list[float]]): Filtered stimulus timestamps.
            - currents (list[int]): Filtered current values.
            - depths (list[int]): Filtered depth values.
    """
    # Get animal ID and build the path to the session blocks CSV file
    animalID = file_util.get_animal_id(data_folder)
    directory = os.path.join(
        "batch_process", "postprocessing", "behavioral_data")
    file_name = f"{animalID}_blocks.csv"
    csv_file_path = os.path.join(directory, file_name)

    # Check if the CSV file exists
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(
            f"The CSV file does not exist at the path: {csv_file_path}")

    # Load the session blocks DataFrame
    session_blocks_df = pd.read_csv(csv_file_path)

    # Check if the DataFrame is empty
    if session_blocks_df.empty:
        raise ValueError("The session_blocks_df DataFrame is empty.")

    # Normalize the paths in the 'Session' column and the data folder path
    session_blocks_df["Session"] = session_blocks_df["Session"].apply(
        os.path.normpath)
    normalized_data_folder = os.path.normpath(data_folder)

    # Filter based on the normalized session paths
    include_array = session_blocks_df.loc[session_blocks_df["Session"]
                                          == normalized_data_folder, "Include"].tolist()
    if np.sum(np.array(include_array)) == 0:
        print("All blocks for this session are bad.")
        return None

    # Load data using the dataloader
    dataloader = dataloader_module.load_data(data_folder, make_folder=False)
    fs = dataloader.fs
    pre_stim_win = int(0.003 * fs)  # 3ms pre-stimulus window (hardcoded)
    post_stim_win = int(0.007 * fs)  # 7ms post-stimulus window (hardcoded)
    all_stim_timestamps = dataloader.all_stim_timestamps
    modified_stim_timestamps = dataloader.all_modified_stim_timestamps
    currents = dataloader.all_currents
    depths = dataloader.all_depths

    # Check for NaN values in modified stimulus timestamps
    contains_nan = any(math.isnan(item)
                       for sublist in modified_stim_timestamps for item in sublist)
    if contains_nan:
        recorded_trial_idx = get_recorded_trial_idx(
            modified_stim_timestamps, currents, depths)
        print("NaN values detected. Filtering trials based on valid timestamps.")
    else:
        recorded_trial_idx = list(range(400))
        modified_stim_timestamps = all_stim_timestamps

    # Filter trial-based data using included blocks from the DataFrame
    included_trials = get_included_trials(include_array, recorded_trial_idx)

    # Extract filtered timestamps, currents, and depths
    all_stim_timestamps = [modified_stim_timestamps[i]
                           for i in included_trials]
    depths = [depths[i] for i in included_trials]
    currents = [currents[i] for i in included_trials]

    # Ensure the lengths of the lists match
    assert (
        len(all_stim_timestamps) == len(depths) == len(currents)
    ), "Mismatch in lengths of timestamps, currents, and depths."

    # Return the filtered data as a namedtuple
    return StimData(all_stim_timestamps, currents, depths)


def get_recorded_trial_idx(modified_stim_timestamps, currents, depths):
    """
    Identifies the indices of trials with valid (non-NaN) timestamps.

    Args:
        modified_stim_timestamps (list[list[float]]): Modified stimulus timestamps.
        currents (list[int]): Current values.
        depths (list[int]): Depth values.

    Returns:
        list[int]: Indices of valid trials.
    """
    return [i for i, sublist in enumerate(modified_stim_timestamps) if all(not np.isnan(item) for item in sublist)]


def get_included_trials(include_array, recorded_trial_idx, trials_per_block=100):
    """
    Filters trials based on inclusion criteria from session blocks.

    Args:
        include_array (list[int]): List indicating which blocks to include (1 = include, 0 = exclude).
        recorded_trial_idx (list[int]): Indices of recorded trials.
        trials_per_block (int): Number of trials per block.

    Returns:
        list[int]: Indices of trials to be included.
    """
    included_trials = [
        idx for idx in recorded_trial_idx if include_array[idx // trials_per_block] == 1]
    return included_trials


def get_stim_ts_indices(all_stim_timestamps, currents, depths, current, depth):
    """
    Retrieves the stimulus timestamps for a specific current and depth combination.

    Args:
        all_stim_timestamps (list[list[float]]): All stimulus timestamps.
        currents (list[int]): List of current values.
        depths (list[int]): List of depth values.
        current (int): Target current value.
        depth (int): Target depth value.

    Returns:
        list[float]: List of stimulus timestamps for the given current and depth.
    """
    filtered_stim_ts = [ts for i, ts in enumerate(
        all_stim_timestamps) if currents[i] == current and depths[i] == depth]

    if current == 0:
        assert all(
            not sublist for sublist in filtered_stim_ts
        ), "There should be no stimulus timestamps when current is 0."

    stim_ts = [ts for sublist in filtered_stim_ts for ts in sublist]
    return stim_ts


def get_waveforms_during_stim(analyzer, stim_times):
    pass


def get_stim_and_nonstim_waveforms(analyzer, stim_times, stim_channel, stim_current):
    pass

# %% Common


def compute_smoothed_firing_rate(trial_data, bin_min, bin_max, bin_size=10, sigma=30):
    """
    Computes smoothed firing rate from spike times using a Gaussian kernel.

    Parameters:
    - trial_data : list of lists of spike times in ms where each nested list is a trial.
    - bin_min : float, start time of bins in ms.
    - bin_max : float, end time of bins in ms.
    - bin_size : float, size of time bins in ms.
    - sigma : float, standard deviation of the Gaussian kernel in ms.

    Returns:
    - smoothed_rate : np.ndarray of smoothed firing rate in Hz (spikes/s).
    - times : np.ndarray of corresponding time points in ms.
    """
    # Flatten the list of lists of spike times
    all_spikes = [spike for trial in trial_data for spike in trial]

    if not all_spikes:  # Check if all_spikes is empty
        return np.array([]), np.array([])

    num_trials = len(trial_data)  # Number of trials

    # Define the histogram bins based on bin_min and bin_max
    bins = np.arange(bin_min, bin_max + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size / 2

    # Create a histogram of the spikes
    hist, _ = np.histogram(all_spikes, bins=bins)

    # Normalize histogram to get average spike count per trial per bin
    hist = hist / num_trials

    # Convert average spike count per bin to firing rate in Hz (spikes/s)
    hist = hist / (bin_size / 1000)  # Convert bin size to seconds and divide

    # Create a Gaussian kernel
    # The kernel needs to be in units of bins
    kernel_width = int(6 * sigma / bin_size)  # Cover ±3 sigma
    if kernel_width % 2 == 0:
        kernel_width += 1  # Make it odd to have a center bin
    kernel_range = np.arange(-kernel_width // 2,
                             kernel_width // 2 + 1) * bin_size
    kernel = np.exp(-(kernel_range**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    # Convolve the spike histogram with the Gaussian kernel
    smoothed_rate = np.convolve(hist, kernel, mode='same')

    return smoothed_rate, bin_centers


# %% Pulse response util


def get_relative_spike_data(spike_times, stim_times, timing_params):
    """
    Calculate the relative spike times around each stimulus time, along with their corresponding indices.

    Args:
        spike_times (np.ndarray): Array of spike times.
        stim_times (np.ndarray): Array of stimulus times.

    Returns:
        tuple:
            - rel_spike_times (list): Spike times relative to each stimulus, in milliseconds.
            - pulse_indices (list): Indices indicating which stimulus each spike time corresponds to.
            - original_indices (list): Original indices of the spikes in the spike_times array.
    """
    pulse_pre_samples = timing_params["pulse_pre_samples"]
    pulse_post_samples = timing_params["pulse_post_samples"]
    fs = timing_params["fs"]
    pre_stim_blank_samples = timing_params["pre_stim_blank_ms"] * fs / 1000.0
    post_stim_blank_samples = timing_params["post_stim_blank_ms"] * fs / 1000.0

    rel_spike_times, pulse_indices, original_indices = [], [], []
    for i, ts in enumerate(stim_times):
        # Identify spikes that fall within the interval around the stimulus time
        interval_mask = np.logical_and(
            spike_times >= ts + post_stim_blank_samples, spike_times <= ts +
            pulse_post_samples - pre_stim_blank_samples
        )

        spikes_in_interval = (spike_times[interval_mask] - ts) * 1000.0 / fs
        # Get the original indices of the spikes in the window
        indices_in_interval = np.where(interval_mask)[0]

        rel_spike_times.extend(spikes_in_interval)
        # Associate each spike with the current stimulus index
        pulse_indices.extend([i] * len(spikes_in_interval))
        original_indices.extend(indices_in_interval)

    return rel_spike_times, pulse_indices, original_indices


def get_pulse_interval_raster_array(rel_spike_times, trial_indices):
    """
    Generates a raster array from relative spike times and their corresponding trial indices.

    Args:
        rel_spike_times (list[float]): List of spike times relative to a stimulus.
        trial_indices (list[int]): List of trial indices corresponding to each spike time.

    Returns:
        np.ndarray: A raster array where each row contains spike times for a specific trial.
                    If no trials are present, an empty array is returned.
    """
    # Handle the case where there are no trials or spike times
    if len(trial_indices) == 0 or len(rel_spike_times) == 0:
        return np.array([], dtype=object)

    # Determine the number of trials (rows) needed for the array
    num_trials = max(trial_indices) + 1  # Assuming trial indices start from 0

    # Initialize a list of lists to hold spike times for each trial
    spike_times_by_trial = [[] for _ in range(num_trials)]

    # Populate the list of lists with spike times
    for spike_time, trial_index in zip(rel_spike_times, trial_indices):
        spike_times_by_trial[trial_index].append(spike_time)

    # Convert the list of lists into a NumPy array with dtype=object
    raster_array = np.array(spike_times_by_trial, dtype=object)

    return raster_array


# %% Train response util


def group_stim_pulses_into_trains(stim_times, timing_params):
    """
    Groups stimulus pulses into trains based on inter-pulse intervals.

    Args:
        stim_times (list or np.ndarray): Sorted list of stimulus times.
        timing_params (dict): Dictionary containing timing parameters, including:
            - fs (int): Sampling frequency in Hz.
            - window_size_ms (float): Expected time window for a train in ms (not used in current implementation).

    Returns:
        list of lists: A list where each sublist contains the stimulus times for a single train.
    """
    # Ensure stim_times is sorted
    assert np.all(stim_times == np.sort(stim_times)
                  ), "stim_times must be sorted."

    fs = timing_params["fs"]
    # Threshold to detect the end of a train (based on 100 Hz stimulation)
    threshold = fs / 100 + 10

    # Calculate the difference between consecutive stimulus times
    diff = np.diff(stim_times)

    # Identify the boundaries between trains
    train_boundaries = np.where(diff > threshold)[0]

    # Group stimulus times into trains
    trains = []
    start_idx = 0
    for boundary in train_boundaries:
        trains.append(stim_times[start_idx: boundary + 1])
        start_idx = boundary + 1

    # Handle the final train
    if start_idx < len(stim_times):
        trains.append(stim_times[start_idx:])

    return trains


def extract_spike_times(spike_times, start_time, end_time, reference_time, fs):
    """
    Extracts spike times within a given time window and normalizes to a reference time.

    Args:
        spike_times (np.ndarray): Array of spike times.
        start_time (int): Start of the time window in samples.
        end_time (int): End of the time window in samples.
        reference_time (int): Time to normalize against (e.g., start of the pre-stimulus period).
        fs (float): Sampling frequency in Hz.

    Returns:
        np.ndarray: Normalized spike times in milliseconds relative to the reference time.
        np.ndarray: Indices of spikes within the time window.
    """
    mask = np.logical_and(spike_times >= start_time, spike_times < end_time)
    indices = np.where(mask)[0]
    normalized_times = (spike_times[indices] - reference_time) / fs * 1000
    return normalized_times, indices


def analyze_stimulus_train_response(stim_times, unit_spike_times, timing_params):
    """
    Analyzes the response of a unit to a train of stimulus pulses.

    Args:
        stim_times (list of lists): List of stimulus times, where each sublist is a train.
        unit_spike_times (np.ndarray): Array of spike times.
        timing_params (dict): Dictionary containing timing parameters.

    Returns:
        tuple: Dictionary containing spike times, spike indices, t-test results, and Z-score info.
    """
    if len(stim_times) == 0:
        return None, None, None, None

    fs = timing_params["fs"]
    stim_duration_samples = int(timing_params["stim_duration_ms"] * fs / 1000)

    pre_stim_spike_times = []
    train_spike_times = []
    post_stim_spike_times = []
    pre_stim_spike_indices = []
    stim_spike_indices = []
    post_stim_spike_indices = []

    all_pre_times = []
    all_stim_times = []
    all_post_times = []

    for train in stim_times:
        pre_stim_start = int(train[0] - stim_duration_samples)
        pre_stim_end = int(train[0])
        post_stim_end = int(train[-1] + stim_duration_samples)

        reference_time = pre_stim_end  # Normalize times

        # Extract spikes for each phase
        pre_times, pre_indices = extract_spike_times(
            unit_spike_times, pre_stim_start, pre_stim_end, reference_time, fs)
        stim_phase_times, stim_indices = extract_spike_times(
            unit_spike_times, pre_stim_end, train[-1], reference_time, fs
        )
        post_times, post_indices = extract_spike_times(
            unit_spike_times, train[-1], post_stim_end, reference_time, fs)

        all_pre_times.extend(pre_times)
        all_stim_times.extend(stim_phase_times)
        all_post_times.extend(post_times)

        # Store extracted times and indices
        pre_stim_spike_times.append(pre_times)
        train_spike_times.append(stim_phase_times)
        post_stim_spike_times.append(post_times)
        pre_stim_spike_indices.append(pre_indices)
        stim_spike_indices.append(stim_indices)
        post_stim_spike_indices.append(post_indices)

    # Calculate overall firing rates
    pre_stim_mean_fr = len(all_pre_times) / \
        (stim_duration_samples * len(stim_times) / fs)
    stim_mean_fr = len(all_stim_times) / \
        (stim_duration_samples * len(stim_times) / fs)

    # Calculate delta firing rate
    delta_fr = stim_mean_fr - pre_stim_mean_fr

    stim_spikes_count = [len(spikes) for spikes in train_spike_times]
    pre_stim_spikes_count = [len(spikes) for spikes in pre_stim_spike_times]

    # Create dictionaries for spike times and indices
    spike_times_dict = {
        'pre_stim_spike_times': pre_stim_spike_times,
        'train_spike_times': train_spike_times,
        'post_stim_spike_times': post_stim_spike_times
    }

    spike_indices_dict = {
        'pre_stim_spike_indices': pre_stim_spike_indices,
        'stim_spike_indices': stim_spike_indices,
        'post_stim_spike_indices': post_stim_spike_indices
    }

    # Perform t-tests
    t_test_dict = {
        'paired_t_val': np.nan,
        'paired_p_val': np.nan,
        'ones_samp_t_val': np.nan,
        'one_samp_p_val': np.nan
    }

    if len(stim_spikes_count) > 1:
        t_test_dict['paired_t_val'], t_test_dict['paired_p_val'] = ttest_rel(
            stim_spikes_count, pre_stim_spikes_count)
        t_test_dict['ones_samp_t_val'], t_test_dict['one_samp_p_val'] = ttest_1samp(
            np.array(stim_spikes_count) - np.array(pre_stim_spikes_count), popmean=0)

    # Z-score calculation
    pre_stim_sigma_fr = np.std(
        [len(pre_times) / (stim_duration_samples / fs) for pre_times in pre_stim_spike_times])
    z_score = (stim_mean_fr - pre_stim_mean_fr) / \
        pre_stim_sigma_fr if pre_stim_sigma_fr > 0 else np.nan

    z_score_dict = {
        'pre_stim_mean_fr': pre_stim_mean_fr,
        'stim_mean_fr': stim_mean_fr,
        'pre_stim_sigma_fr': pre_stim_sigma_fr,
        'z_score': z_score
    }

    # Return structured data
    return spike_times_dict, spike_indices_dict, t_test_dict, z_score_dict


def get_train_win_raster_arr(
    spike_times_dict,
    stim_trains,
    timing_params,
):
    """
    Generates a raster array of spike times for each trial, concatenating pre-stimulus, stimulus, and post-stimulus periods.

    Args:
        pre_stim_spike_times (list of np.ndarray): List of arrays containing pre-stimulus spike times for each trial.
        train_spike_times (list of np.ndarray): List of arrays containing stimulus spike times for each trial.
        post_stim_spike_times (list of np.ndarray): List of arrays containing post-stimulus spike times for each trial.
        stim_trains (list of lists): List of stimulus times for each trial.
        timing_params (dict): Dictionary containing timing parameters, including the sampling frequency (`fs`).

    Returns:
        np.ndarray: A raster array where each row corresponds to a trial and contains concatenated spike times.
    """
    # Ensure that the lengths of inputs are consistent
    num_trials = len(stim_trains)

    pre_stim_spike_times = spike_times_dict['pre_stim_spike_times']
    train_spike_times = spike_times_dict['train_spike_times']
    post_stim_spike_times = spike_times_dict['post_stim_spike_times']

    assert len(pre_stim_spike_times) == num_trials
    assert len(train_spike_times) == num_trials
    assert len(post_stim_spike_times) == num_trials

    raster_array = []
    for trial_idx in range(num_trials):
        # Concatenate pre-stim, stim, and post-stim spike times
        trial_spike_times = np.concatenate(
            [
                pre_stim_spike_times[trial_idx],
                train_spike_times[trial_idx],
                post_stim_spike_times[trial_idx],
            ]
        )
        # Append to the raster array
        raster_array.append(trial_spike_times)

    # Convert the list of arrays into a NumPy array
    raster_array = np.array(raster_array, dtype=object)
    return raster_array
