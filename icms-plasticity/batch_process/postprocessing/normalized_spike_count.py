import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from collections import defaultdict
from scipy.stats import norm
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import KMeans
import batch_process.util.file_util as file_util
from util.file_util import file_dialog
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import seaborn as sns
from batch_process.util.ppt_image_inserter import PPTImageInserter
import math


def is_OK_response(stim_response):
    tr = stim_response.train_response
    pr = stim_response.pulse_response

    baseline_fr_good = tr.pre_stim_mean_fr > 0.5

    # Check if there are enough spikes
    num_spikes = len(pr.rel_spike_timestamps)
    enough_spikes = num_spikes > 50

    # Check if the stimulation current is within the desired range
    stim_current = stim_response.stim_current
    stim_current_in_range = (stim_current > 2) and (stim_current < 7)

    return baseline_fr_good and enough_spikes and stim_current_in_range


def process_stim_conditions_by_week(animal_id, good_data_folders, max_week=5):
    stim_channels = get_stim_channels(animal_id)
    # Define stimulation conditions for each channel and current range
    stim_conditions = [(ch, curr)
                       for ch in stim_channels for curr in range(3, 7)]

    # Initialize a dictionary to store data grouped by week
    stim_condition_dict = {stim_condition: {
        'spike_counts': [], 'week_indices': []} for stim_condition in stim_conditions}

    days_relative = file_util.convert_dates_to_relative_days(good_data_folders)

    # Loop over each data folder, assuming folders are ordered by session date
    for folder_index, data_folder in enumerate(good_data_folders):
        pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"

        with open(pkl_path, "rb") as file:
            session_responses = pickle.load(file)

        unit_ids = session_responses.unit_ids
        bins = np.arange(0, 110, 10)  # Define bin edges, 10 pulses

        # Convert session day to relative week number
        week_index = days_relative[folder_index] // 7

        if week_index >= max_week:
            continue

        # Loop over each stimulation condition
        for stim_condition in stim_conditions:
            binned_spike_counts_array = np.zeros(
                (len(unit_ids), len(bins) - 1))
            num_trials = None

            for unit_index, unit_id in enumerate(unit_ids):
                ur = session_responses.get_unit_response(unit_id)
                if ur is None:
                    continue
                scr = ur.get_stim_response(
                    stim_condition[0], stim_condition[1])

                if scr is None:
                    continue

                if not is_OK_response(scr):
                    continue

                tr = scr.train_response
                train_raster = tr.raster_array

                # Accumulate spike counts in histogram bins if there are trials
                if len(train_raster) > 0:
                    num_trials = len(train_raster)
                    for row in train_raster:
                        counts, _ = np.histogram(row, bins=bins)
                        binned_spike_counts_array[unit_index, :] += counts

            if num_trials is None or num_trials == 0:
                continue

            # Calculate total spikes across all units and normalize
            total_spikes = np.sum(binned_spike_counts_array, axis=0)
            normalized_spike_counts = total_spikes / \
                (num_trials * len(unit_ids))

            # Append only if `normalized_spike_counts` has non-zero data
            if np.any(normalized_spike_counts):
                # Append data, grouping by week index instead of session index
                stim_condition_dict[stim_condition]['spike_counts'].append(
                    normalized_spike_counts)
                stim_condition_dict[stim_condition]['week_indices'].append(
                    week_index)

    # Group spike counts by week for each stim condition
    # weekly_stim_condition_dict = {}
    # for stim_condition, data in stim_condition_dict.items():
    #     unique_weeks = np.unique(data['week_indices'])
    #     weekly_stim_condition_dict[stim_condition] = {
    #         'spike_counts': [], 'week_indices': unique_weeks}

    #     for week in unique_weeks:
    #         # Get all spike counts for the current week and calculate average across sessions
    #         week_spike_counts = [sc for i, sc in enumerate(
    #             data['spike_counts']) if data['week_indices'][i] == week]
    #         avg_spike_counts = np.mean(week_spike_counts, axis=0)
    #         weekly_stim_condition_dict[stim_condition]['spike_counts'].append(
    #             avg_spike_counts)

    return stim_condition_dict


def get_stim_channels(animal_id, max_week=5):
    base_path = Path("C:/data/")
    # Get stim channels used in at least 5 sessions
    df = load_session_data(animal_id, base_path)
    for ch in df['stim_channel'].unique():
        ch_df = df[df['stim_channel'] == ch]
        if len(ch_df['days_relative'].unique()) < max_week:
            df = df[df['stim_channel'] != ch]

    df['weeks_relative'] = (df['days_relative'] // 7).astype(int)
    df = df[df['weeks_relative'] < max_week]

    days = df['days_relative'].unique()
    thresholds_over_time = {}
    sessions_over_time = []

    for day in days:
        df_session = df[df['days_relative'] == day]
        thresholds = df_session['detection_threshold'].unique()
        # Get session date for this day
        session_date = df_session['session'].iloc[0]
        sessions_over_time.append(session_date)

        for threshold in thresholds:
            a = df_session[df_session['detection_threshold'] == threshold]
            stim_channels = a['stim_channel'].unique()

            for stim_channel in stim_channels:
                if stim_channel not in thresholds_over_time:
                    thresholds_over_time[stim_channel] = []
                # Append (session_date, threshold) tuple to each channel's list
                thresholds_over_time[stim_channel].append(
                    (session_date, threshold))

    stim_channels = df['stim_channel'].unique()
    return stim_channels


def plot_normalized_spike_counts_by_week(stim_condition_dict, max_week, animal_id):
    plt.rcParams.update({'font.size': 8})

    # Get channels and currents based on `stim_condition_dict`
    stim_channels = get_stim_channels(animal_id)
    stim_conditions = [(ch, curr)
                       for ch in stim_channels for curr in range(3, 7)]

    channels = sorted(list(set([stim[0] for stim in stim_conditions])))
    currents = sorted(list(set([stim[1] for stim in stim_conditions])))

    plt.figure(figsize=(4, 2.5))
    cmap = plt.cm.get_cmap('tab10', len(channels) + 5)

    pulse_index = 0  # Adjust as needed

    # Loop through each channel and current
    for channel_index, channel in enumerate(channels):
        base_color = cmap(channel_index)

        for current_index, current in enumerate(currents):
            stim_condition = (channel, current)

            # Skip if the condition doesn't exist in the dictionary
            if stim_condition not in stim_condition_dict:
                continue

            # Get spike counts and week indices for this condition
            stim_condition_response = stim_condition_dict[stim_condition]['spike_counts']
            week_indices = stim_condition_dict[stim_condition]['week_indices']

            # Collect x (week) and y (spike count) values for the line plot
            x_vals = []
            y_vals = []

            # Loop through each week to collect points
            for week in range(max_week + 1):
                week_sessions = [i for i, w in enumerate(
                    week_indices) if w == week]

                # Plot individual points for each spike count within the week
                for i in week_sessions:
                    spike_count = stim_condition_response[i][pulse_index]
                    plt.scatter(week, spike_count, color=base_color, s=10 + (current_index * 5),
                                label=f'Ch {channel}, {current} µA' if i == 0 and week == 0 else "")

                    # Collect points for line plot
                    x_vals.append(week)
                    y_vals.append(spike_count)

            # Plot line connecting points
            plt.plot(x_vals, y_vals, color=base_color,
                     linewidth=1 + current_index * 0.5)

            # # Only plot non-NaN values
            # valid_weeks = np.where(~np.isnan(week_spike_counts))[0]

            # # Line thickness and marker size based on current level
            # line_thickness = 1 + (current_index * 1)
            # marker_size = 15 + (current_index * 15)

            # plt.scatter(
            #     valid_weeks, week_spike_counts[valid_weeks], color=base_color, s=marker_size)
            # plt.plot(valid_weeks, week_spike_counts[valid_weeks],
            #          color=base_color, linewidth=line_thickness,
            #          label=f'Ch {channel}, {current} µA')

    # Create legends for channels and currents
    channel_legend = [mlines.Line2D([], [], color=cmap(i), label=f'Channel {ch}', linewidth=3)
                      for i, ch in enumerate(channels)]
    current_legend = [mlines.Line2D([], [], color='gray', label=f'Current {curr} µA', linewidth=1 + (i * 0.5))
                      for i, curr in enumerate(currents)]

    plt.legend(handles=channel_legend + current_legend,
               fontsize='x-small', loc='best')

    # Set integer ticks for weeks
    plt.xticks(ticks=np.arange(0, max_week + 1, 1))

    # Labels, title, and layout adjustments
    plt.xlabel('Week Index', fontsize=7)
    plt.ylabel('Normalized Spike Count', fontsize=7)
    plt.title(f'Normalized Spike Counts vs. Weeks for {
              animal_id} (Pulse {pulse_index})', fontsize=7)
    plt.tight_layout()
    plt.show()


def sort_folders(data_folders, max_week=5):
    sorted_data_folders = np.array(file_util.sort_data_folders(data_folders))

    days_relative = np.array(
        file_util.convert_dates_to_relative_days(sorted_data_folders))
    weeks_relative = (days_relative // 7).astype(int)

    # Filter folders to include only those within the first 5 weeks
    good_data_folders = sorted_data_folders[np.where(
        weeks_relative < max_week)[0]]

    return good_data_folders


# %%


def aggregate_data_across_animals_by_current(animal_data_dicts, max_week=6):
    # Initialize a dictionary to aggregate spike counts by current and week
    aggregated_data = {current: {'spike_counts': [
        [] for _ in range(max_week + 1)]} for current in range(3, 7)}

    for animal_data in animal_data_dicts:
        for stim_condition, data in animal_data.items():
            _, current = stim_condition  # Extract current, ignore channel

            # Aggregate spike counts across weeks
            for week_index, spike_counts in zip(data['week_indices'], data['spike_counts']):
                if week_index <= max_week:
                    aggregated_data[current]['spike_counts'][week_index].extend(
                        spike_counts)

    return aggregated_data


def plot_aggregated_data_by_current(aggregated_data, max_week=6, plot_type="median"):
    # Define currents to be plotted
    currents = sorted(list(aggregated_data.keys()))
    plt.figure(figsize=(6, 4))
    cmap = plt.cm.get_cmap('tab10', len(currents) + 5)

    # Define offset for scatter points within each week (e.g., for 4 currents)
    num_currents = len(currents)
    offset_spacing = 0.1  # Adjust this for spacing between currents within a week

    # Loop through each current to create a line and scatter points
    for current_index, current in enumerate(currents):
        x_offset = (current_index - num_currents / 2) * \
            offset_spacing + offset_spacing/2  # Center around each week
        color = cmap(current_index)
        current_median = []  # Holds aggregated values for plotting the line
        week_indices = []  # Holds weeks with valid data

        # Loop through each week
        for week in range(max_week + 1):
            # Retrieve spike counts for the current and week
            spike_counts = aggregated_data[current]['spike_counts'][week]

            # Filter out any non-numeric or empty entries
            spike_counts = [count for count in spike_counts if isinstance(
                count, (int, float)) or (isinstance(count, np.ndarray) and count.size > 0)]

            # Check if there is data for the current week
            if spike_counts:
                # Calculate median or mean for the current week
                if plot_type == "median":
                    agg_value = np.median(spike_counts)
                else:
                    agg_value = np.mean(spike_counts)

                # Append data for plotting
                current_median.append(agg_value)
                week_indices.append(week)  # Only add week if data is present

                # Plot individual scatter points for each data point with offset
                for count in spike_counts:
                    plt.scatter([week + x_offset], [count],
                                color=color, s=10, alpha=0.6)
            else:
                print(f"No valid data for week {week}, current {current} µA")

        # Plot line for median/mean values across weeks with valid data only
        if week_indices and current_median:
            plt.plot(week_indices, current_median, color=color, linewidth=2,
                     label=f'Current {current} µA')

    # Legend for currents
    plt.legend(fontsize='small', loc='upper right')

    # Set integer ticks for weeks, with tick labels centered under each grouped set of points
    plt.xticks(ticks=np.arange(0, max_week, 1))

    # Labels, title, and layout adjustments
    plt.xlabel('Week Index')
    plt.ylabel('Normalized Spike Count')
    plt.title(f'{plot_type.capitalize()
                 } Spike Counts by Current vs. Weeks Across All Animals')
    plt.tight_layout()
    plt.show()


# %%
# Load data and process animal IDs
with open('all_animal_folder_list.pkl', 'rb') as f:
    all_animal_folder_list_0 = pickle.load(f)

# Sort animal IDs based on the numeric portion
animal_ids = np.unique([file_util.get_animal_id(files[0])
                       for files in all_animal_folder_list_0])
numeric_values = np.array([int(s[4:]) for s in animal_ids])
sorted_indices = np.argsort(numeric_values)
sorted_animal_ids = animal_ids[sorted_indices]

# %% Plot each animal spike counts per week

for animal_index, animal_id in enumerate(sorted_animal_ids):
    animal_folder_list = all_animal_folder_list_0[animal_index]
    good_data_folders = sort_folders(animal_folder_list)

    weekly_data = process_stim_conditions_by_week(animal_id, good_data_folders)

    plot_normalized_spike_counts_by_week(
        weekly_data, max_week=5, animal_id=animal_id)
# %%
# Example of usage:
animals = []
for animal_index, animal_id in enumerate(sorted_animal_ids):
    animal_folder_list = all_animal_folder_list_0[sorted_indices[animal_index]]
    good_data_folders = sort_folders(animal_folder_list)
    animals.append((animal_id, good_data_folders))

animal_data_dicts = [process_stim_conditions_by_week(
    animal_id, good_data_folders) for animal_id, good_data_folders in animals]
aggregated_data = aggregate_data_across_animals_by_current(
    animal_data_dicts, max_week=6)
# %%

plot_aggregated_data_by_current(
    aggregated_data, max_week=6, plot_type="median")
