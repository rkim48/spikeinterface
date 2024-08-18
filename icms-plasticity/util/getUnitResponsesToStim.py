from spikeinterface import full as si
import numpy as np


def compute_smoothed_firing_rate(spikes, indices, bin_size=1, sigma=5):
    """
    Computes smoothed firing rate from spike times using a Gaussian kernel.

    Parameters:
    - spikes : list of spike times
    - indices : list of corresponding indices for each spike
    - bin_size : size of time bins in ms
    - sigma : standard deviation of the Gaussian kernel in ms

    Returns:
    - smoothed_rate : smoothed firing rate
    - times : corresponding time points
    """
    if not spikes:  # Check if spikes is empty
        return [], []

    # Adjust the spike time range to account for the Gaussian kernel
    kernel_extent = 3 * sigma
    hist_range = (min(spikes) - kernel_extent, max(spikes) + kernel_extent)

    # Create a histogram of the spikes
    hist, bins = np.histogram(spikes, bins=np.arange(
        hist_range[0], hist_range[1] + bin_size, bin_size))
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Create a Gaussian kernel
    kernel_range = np.arange(-3 * sigma, 3 * sigma + bin_size, bin_size)
    kernel = np.exp(-kernel_range**2 / (2 * sigma**2))
    kernel = kernel / sum(kernel)

    # Convolve the spike histogram with the Gaussian kernel
    smoothed_rate_full = np.convolve(hist, kernel, mode='full')

    # Extract the 'valid' portion of the convolution output
    valid_start_idx = len(kernel) // 2
    valid_end_idx = valid_start_idx + len(hist)
    smoothed_rate = smoothed_rate_full[valid_start_idx:valid_end_idx]

    return smoothed_rate, bin_centers


def get_stim_ts_indices(all_stim_timestamps, currents, depths, current, depth):
    filtered_stim_ts = [ts for i, ts in enumerate(
        all_stim_timestamps) if currents[i] == current and depths[i] == depth]
    if current == 0:
        assert all(not sublist for sublist in filtered_stim_ts)
    stim_ts = [ts for sublist in filtered_stim_ts for ts in sublist]
    return stim_ts


def get_relative_spike_data(spike_times, stim_ts, fs, pre_stim_win, post_stim_win):
    rel_spike_times, trial_indices, original_indices = [], [], []
    for i, ts in enumerate(stim_ts):
        # Find the indices of spikes within the window
        window_mask = np.logical_and(spike_times >= ts - pre_stim_win,
                                     spike_times <= ts + post_stim_win)
        spikes_in_window = (spike_times[window_mask] - ts) * 1000.0 / fs
        # Get the original indices of the spikes in the window
        indices_in_window = np.where(window_mask)[0]

        rel_spike_times.extend(spikes_in_window)
        trial_indices.extend([i] * len(spikes_in_window))
        # Store the original indices
        original_indices.extend(indices_in_window)

    return rel_spike_times, trial_indices, original_indices


class ResponseExtractor:
    def __init__(self, waveform_extractor):
        self.responses = {}
        self.we = waveform_extractor  # Shared waveform_extractor
        self.unit_ids = []

    def __repr__(self):
        return f"{self.__class__.__name__}: {len(self.responses)} units\nUnit IDs: {list(self.unit_ids)}"

    def add_response(self, response):
        self.responses[response.unit_id] = response
        if response.unit_id not in self.unit_ids:
            self.unit_ids.append(response.unit_id)

    def get_response(self, unit_id):
        return self.responses.get(unit_id, None)

    def get_all_responses(self):
        return list(self.responses.values())


class Response:
    def __init__(self, unit_id, waveform_extractor):
        extremum_ch_dict = si.get_template_extremum_channel(
            waveform_extractor, outputs="index")
        self.unit_id = unit_id
        self.waveform_extractor = waveform_extractor
        self.primary_ch_idx = extremum_ch_dict[unit_id]
        self.all_spike_times = []
        self.stim_params = []  # list of stim current, stim channel tuples
        self.response_data = {}  # stim_param:response dictionary

    def __repr__(self):
        stim_params_summary = ', '.join(map(str, self.response_data.keys()))
        return f"{self.__class__.__name__}: Unit {self.unit_id}\nStim params (current, depth): {stim_params_summary}"

    def add_waveforms(self, stim_params):
        # Ensure that stim_params is a key in response_data
        if stim_params not in self.response_data:
            self.response_data[stim_params] = {}
        # Check if 'waveforms' is not already fetched for these stim_params
        if 'waveforms' not in self.response_data[stim_params]:
            # Fetch and store the waveforms
            self.response_data[stim_params]['waveforms'] = self.waveform_extractor.get_waveforms(
                unit_id=self.unit_id)

    def get_waveforms(self, stim_params, only_primary_ch=False):
        self.add_waveforms(stim_params)
        waveforms = self.response_data[stim_params].get('waveforms')
        original_indices = self.response_data[stim_params].get(
            'original_indices')
        # Use the original indices to index into spike_times
        if waveforms is not None and original_indices is not None:
            selected_waveforms = waveforms[original_indices]
            if only_primary_ch:
                return selected_waveforms[:, :, self.primary_ch_idx]
            else:
                return selected_waveforms
        else:
            # Handle the case where there are no waveforms or indices
            return None

    def get_stim_params(self):
        self.stim_params = list(self.response_data.keys())
        return self.stim_params

    def get_response_data(self, stim_params):
        # Return the response data for the given stimulation parameters
        return self.response_data.get(stim_params, "No data for given stim params")
