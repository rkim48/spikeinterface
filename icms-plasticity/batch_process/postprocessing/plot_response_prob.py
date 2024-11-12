# load session
# for every stim condition
# for every neuron
# look at train response
# plot probability as function of pulse #
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
# %%


def is_OK_response(stim_response):
    tr = stim_response.train_response

    baseline_fr_good = tr.pre_stim_mean_fr > 0.5

    # Check if there are enough spikes
    num_spikes = len(pr.rel_spike_timestamps)
    enough_spikes = num_spikes > 50

    # Check if the stimulation current is within the desired range
    stim_current = stim_response.stim_current
    stim_current_in_range = (stim_current > 2) and (stim_current < 7)

    return baseline_fr_good and enough_spikes and stim_current_in_range


def is_good_train_response(stim_response):
    tr = stim_response.train_response
    pr = stim_response.pulse_response

    # Calculate p-value from z-score
    z_score = tr.z_score_dict['z_score']
    p_value = 2 * (1 - norm.cdf(abs(z_score)))  # Two-tailed test
    is_modulated = p_value < 0.05

    # Check z-score range and baseline firing rate
    z_score_in_range = (tr.z_score > 0) and (tr.z_score < 30)
    baseline_fr_good = tr.pre_stim_mean_fr > 0.5

    # Check if there are enough spikes
    num_spikes = len(pr.rel_spike_timestamps)
    enough_spikes = num_spikes > 200

    # Check if the stimulation current is within the desired range
    stim_current = stim_response.stim_current
    stim_current_in_range = (stim_current > 2) and (stim_current < 7)

    # Return True if all conditions are met
    return is_modulated and z_score_in_range and baseline_fr_good and enough_spikes and stim_current_in_range


def is_good_pulse_response(stim_response):
    tr = stim_response.train_response
    pr = stim_response.pulse_response

    is_pulse_locked = pr.is_pulse_locked
    # Check z-score range and baseline firing rate
    baseline_fr_good = tr.pre_stim_mean_fr > 0.5

    # Check if there are enough spikes
    num_spikes = len(pr.rel_spike_timestamps)
    enough_spikes = num_spikes > 200

    # Check if the stimulation current is within the desired range
    stim_current = stim_response.stim_current
    stim_current_in_range = (stim_current > 2) and (stim_current < 7)

    # Return True if all conditions are met
    return is_pulse_locked and baseline_fr_good and enough_spikes and stim_current_in_range


def get_week_relative(data_folder, all_animal_folder_list_0):
    import pandas as pd
    date_str = file_util.get_date_str(data_folder)
    animal_id = file_util.get_animal_id(data_folder)

    # Find the corresponding animal's folder list in all_animal_folder_list_0
    for animal_list in all_animal_folder_list_0:
        if animal_id in animal_list[0]:  # Assuming animal ID is in the path
            animal_folders = animal_list
            break
    else:
        raise ValueError(f"No matching animal folder found for {animal_id}")

    # Extract session dates from the folder paths
    session_dates = []
    for folder in animal_folders:
        # Extract the session name (date) from the folder path
        # Extract the date part from the folder path
        session_name = folder.split('\\')[-1]
        session_date = pd.to_datetime(session_name, format="%d-%b-%Y")
        session_dates.append(session_date)

    # Find the minimum date (earliest session)
    min_date = min(session_dates)

    # Compute the relative weeks
    session_week_relative = {}
    for folder, session_date in zip(animal_folders, session_dates):
        # Calculate relative weeks
        weeks_relative = (session_date - min_date).days // 7
        session_week_relative[folder] = weeks_relative

    # For the input data folder, return its relative week
    data_folder_date = pd.to_datetime(
        data_folder.split('\\')[-1], format="%d-%b-%Y")
    relative_week = (data_folder_date - min_date).days // 7

    return relative_week


# %%
data_folder = "C:\\data\\ICMS92\Behavior\\14-Sep-2023"
pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"

with open(pkl_path, "rb") as file:
    session_responses = pickle.load(file)

unit_ids = session_responses.unit_ids
stim_conditions = session_responses.get_unit_response(
    unit_id=1).show_stim_conditions()

colors = sns.color_palette("husl", len(unit_ids) + 2)

for stim_condition in stim_conditions:

    plt.figure()

    for unit_index, unit_id in enumerate(unit_ids):
        ur = session_responses.get_unit_response(unit_id)
        scr = ur.get_stim_response(stim_condition[0], stim_condition[1])
        tr = scr.train_response
        pr = scr.pulse_response

        if not is_good_train_response(scr):
            continue

        train_raster = tr.raster_array

        bins = np.arange(0, 50, 10)

        # Bins will have 1 less edge than the number of bins
        cumulative_counts = np.zeros(len(bins) - 1)

        # Loop through each row and sum the counts into the cumulative array
        for row in train_raster:
            counts, _ = np.histogram(row, bins=bins)
            cumulative_counts += (counts > 0).astype(int)

        bin_centers = (bins[:-1] + bins[1:]) / 2

        prob = cumulative_counts / len(train_raster)

        # Adjust width to slightly smaller than bin size
        plt.plot(bin_centers, prob,
                 color=colors[unit_index], linewidth=1, label=f'Unit {unit_id}')
        # Circles at each point
        plt.scatter(bin_centers, prob, color=colors[unit_index], s=10)

        plt.xlabel('Time (ms)')
        plt.ylabel('Probability')
        plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

    plt.ylim(0, 1)
    plt.title(f'Channel {stim_condition[0]} at {
              stim_condition[1]} µA\n{len(train_raster)} trials')
    plt.legend(loc='upper right', fontsize='small')

# %%
starting_dir = "C:\\data"
data_folders = file_dialog(starting_dir=starting_dir)
sorted_data_folders = file_util.sort_data_folders(data_folders)

# %%
stim_conditions = [(9, 3), (9, 4), (9, 5), (11, 3), (11, 4),
                   (11, 5), (12, 3), (12, 4), (12, 5)]
for stim_condition in stim_conditions:
    plt.figure(figsize=(8, 4))

    for folder_index, data_folder in enumerate(sorted_data_folders):
        pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"

        with open(pkl_path, "rb") as file:
            session_responses = pickle.load(file)

        unit_ids = session_responses.unit_ids

        session_dp1 = []
        session_dp2 = []

        plt.figure()

        for unit_index, unit_id in enumerate(unit_ids):
            ur = session_responses.get_unit_response(unit_id)
            scr = ur.get_stim_response(stim_condition[0], stim_condition[1])

            if scr is None:
                continue

            tr = scr.train_response
            pr = scr.pulse_response

            if not is_good_train_response(scr):
                continue

            train_raster = tr.raster_array

            bins = np.arange(0, 50, 10)
            bin_centers = np.arange(1, len(bins))  # Adjusted bin centers
            cumulative_counts = np.zeros(len(bins) - 1)

            # Loop through each row and sum the counts into the cumulative array
            for row in train_raster:
                counts, _ = np.histogram(row, bins=bins)
                cumulative_counts += (counts > 0).astype(int)

            prob = cumulative_counts / len(train_raster)

            dp1 = prob[1] - prob[0]
            dp2 = prob[2] - prob[1]

            plt.plot(bin_centers, prob,
                     color=colors[unit_index], linewidth=1, label=f'Unit {unit_id}')
            # Circles at each point
            plt.scatter(bin_centers, prob, color=colors[unit_index], s=10)

        if len(session_dp1) == 0 and len(session_dp2) == 0:
            continue

        x1 = len(session_dp1) * [folder_index - 0.1]
        x2 = len(session_dp2) * [folder_index + 0.1]

        # Add scatter plot points
        plt.scatter(x1, session_dp1, c='C0')
        plt.scatter(x2, session_dp2, c='C1')

    # Adding scatter plot points for labels only once (outside the loop)
    plt.scatter([], [], c='C0', label='ΔP1')
    plt.scatter([], [], c='C1', label='ΔP2')

    # Set legend and plot details
    plt.xticks(ticks=np.arange(len(sorted_data_folders)))
    plt.xlabel('Session Index')
    plt.ylabel('Change in Firing Probability')
    plt.title(f'ICMS92 ΔP for Ch.{stim_condition[0]} at {
              stim_condition[1]} uA')
    plt.legend(loc='best')  # Ensure the legend is displayed properly
    plt.show()
    plt.savefig(f"delta_p_{stim_condition}.png")

# %%
stim_conditions = [(11, 4)]

for stim_condition in stim_conditions:
    for folder_index, data_folder in enumerate(sorted_data_folders):
        pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"

        with open(pkl_path, "rb") as file:
            session_responses = pickle.load(file)

        unit_ids = session_responses.unit_ids

        # Create a new figure for each stim condition and session
        plt.figure(figsize=(8, 6))

        colors = sns.color_palette("husl", len(unit_ids) + 2)

        for unit_index, unit_id in enumerate(unit_ids):
            ur = session_responses.get_unit_response(unit_id)
            scr = ur.get_stim_response(stim_condition[0], stim_condition[1])

            if scr is None:
                continue

            tr = scr.train_response
            pr = scr.pulse_response

            if not is_good_train_response(scr):
                continue

            train_raster = tr.raster_array

            # Define bins for calculating probability
            bins = np.arange(0, 60, 10)
            cumulative_counts = np.zeros(len(bins) - 1)

            # Loop through each row and sum the counts into the cumulative array
            for row in train_raster:
                counts, _ = np.histogram(row, bins=bins)
                cumulative_counts += (counts > 0).astype(int)

            prob = cumulative_counts / len(train_raster)

            # Plot the probability for each unit
            bin_centers = np.arange(1, 6)
            plt.plot(bin_centers, prob,
                     color=colors[unit_index], linewidth=1, label=f'Unit {unit_id}')
            # Circles at each point for visualization
            plt.scatter(bin_centers, prob, color=colors[unit_index], s=10)

        plt.xlabel('Pulse number')
        plt.ylabel('Probability')
        plt.ylim([0, 1])
        plt.xticks(ticks=np.arange(1, 6))  # Set x-ticks to integers only
        plt.title(f'Probability of Firing for Ch.{stim_condition[0]} at {
                  stim_condition[1]} uA (Session {folder_index})')
        plt.tight_layout()

        # Save figure for each stimulation condition and session
        plt.savefig(f"probability_stim_{
                    stim_condition}_session_{folder_index}.png")
        plt.close()  # Close the figure to avoid overlapping

# %% Plot population spike rate for each bin
# Aggregate total spikes for all N neurons for first M bins
starting_dir = "C:\\data"
data_folders = file_dialog(starting_dir=starting_dir)
# %% 10/27 normalized spike count
# 10/29 average currents?
sorted_data_folders = np.array(file_util.sort_data_folders(data_folders))

days_relative = np.array(
    file_util.convert_dates_to_relative_days(sorted_data_folders))
weeks_relative = (days_relative // 7).astype(int)

good_data_folders = sorted_data_folders[np.where(weeks_relative < 5)[
    0]]  # CUTOFF AT 5


data_folder = good_data_folders[3]  # RANDOM INDEX
animal_id = file_util.get_animal_id(data_folder)
pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"
with open(pkl_path, "rb") as file:
    session_responses = pickle.load(file)
ur = session_responses.get_unit_response(0)
stim_conditions = ur.show_stim_conditions()

stim_condition_spike_count_dict = {}
stim_condition_session_index_dict = {}

for stim_condition in stim_conditions:
    # Create a new figure for each stim condition
    # plt.figure(figsize=(8, 6))
    normalized_spike_counts_arr = []
    session_arr = []
    for folder_index, data_folder in enumerate(good_data_folders):
        print(f'folder index: {folder_index}')
        pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"

        with open(pkl_path, "rb") as file:
            session_responses = pickle.load(file)

        unit_ids = session_responses.unit_ids

        bins = np.arange(0, 110, 10)
        # Spike counts array is N neurons x M bins array
        binned_spike_counts_array = np.zeros((len(unit_ids), len(bins) - 1))
        unit_trial_count_list = []

        for unit_index, unit_id in enumerate(unit_ids):
            ur = session_responses.get_unit_response(unit_id)
            scr = ur.get_stim_response(stim_condition[0], stim_condition[1])

            if scr is None:
                continue

            tr = scr.train_response
            pr = scr.pulse_response

            if not is_good_train_response(scr):
                continue

            train_raster = tr.raster_array

            # Loop through each row and sum the counts into the cumulative array
            for row in train_raster:
                counts, _ = np.histogram(row, bins=bins)
                binned_spike_counts_array[unit_index, :] += counts

            unit_trial_count_list.append(len(train_raster))

        if len(unit_trial_count_list) == 0:
            continue

        # Check if all units have the same number of trials
        if len(unit_trial_count_list) == 1:
            all_units_have_same_num_trials = True
        else:
            all_units_have_same_num_trials = np.all(
                np.array(unit_trial_count_list) == unit_trial_count_list[0])

        if not all_units_have_same_num_trials:
            print(f"Skipping session {folder_index} due to mismatched trial counts: {
                  unit_trial_count_list}")
            continue

        # Sum spikes and normalize by number of neurons and trials
        total_spikes = np.sum(binned_spike_counts_array, axis=0)
        normalized_spike_counts = total_spikes / \
            (unit_trial_count_list[0] * len(unit_ids))
        normalized_spike_counts_arr.append(normalized_spike_counts)
        session_arr.append(folder_index)

        # Plot the normalized spike counts
        # bin_centers = np.arange(1, len(bins))
        # plt.plot(bin_centers, normalized_spike_counts,
        #          color=f'C{folder_index}', linewidth=1, label=f'Session {folder_index}')

    stim_condition_spike_count_dict[stim_condition] = normalized_spike_counts_arr
    stim_condition_session_index_dict[stim_condition] = session_arr

    # plt.xlabel('Pulse number')
    # plt.ylabel('Normalized spike count')
    # plt.xticks(ticks=np.arange(1, len(bins)))  # Set x-ticks based on bins
    # plt.title(f'{animal_id} Normalized Spike Count for Ch.{
    #           stim_condition[0]} at {stim_condition[1]} uA')
    # plt.legend()
    # plt.tight_layout()

    # Save figure for each stimulation condition and session
    # plt.savefig(f"{animal_id}_normalized_spike_count_ch{stim_condition[0]}_{
    #             stim_condition[1]}uA_session_{folder_index}.png")
    # plt.close()  # Close the figure to avoid overlapping

stim_condition_spike_count_dict_old = stim_condition_spike_count_dict
stim_condition_session_index_dict_old = stim_condition_session_index_dict
# %%


def process_stim_conditions_by_week(good_data_folders, stim_channels, max_week=5):
    # Define stimulation conditions for each channel and current range
    stim_conditions = [(ch, curr)
                       for ch in stim_channels for curr in range(3, 7)]

    # Initialize a dictionary to store data grouped by week
    stim_condition_dict = {stim_condition: {
        'spike_counts': [], 'week_indices': []} for stim_condition in stim_conditions}

    # Loop over each data folder, assuming folders are ordered by session date
    for folder_index, data_folder in enumerate(good_data_folders):
        pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"

        with open(pkl_path, "rb") as file:
            session_responses = pickle.load(file)

        unit_ids = session_responses.unit_ids
        bins = np.arange(0, 110, 10)  # Define bin edges, 10 pulses

        # Convert session day to relative week number
        days_relative = file_util.convert_dates_to_relative_days([data_folder])
        week_index = days_relative[0] // 7

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

                if scr is None or not is_OK_response(scr):
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
    weekly_stim_condition_dict = {}
    for stim_condition, data in stim_condition_dict.items():
        unique_weeks = np.unique(data['week_indices'])
        weekly_stim_condition_dict[stim_condition] = {
            'spike_counts': [], 'week_indices': unique_weeks}

        for week in unique_weeks:
            # Get all spike counts for the current week and calculate average across sessions
            week_spike_counts = [sc for i, sc in enumerate(
                data['spike_counts']) if data['week_indices'][i] == week]
            avg_spike_counts = np.mean(week_spike_counts, axis=0)
            weekly_stim_condition_dict[stim_condition]['spike_counts'].append(
                avg_spike_counts)

    return weekly_stim_condition_dict


def get_stim_channels(animal_id):
    base_path = Path("C:/data/")
    # Get stim channels used in at least 5 sessions
    df = load_session_data(animal_id, base_path)
    for ch in df['stim_channel'].unique():
        ch_df = df[df['stim_channel'] == ch]
        if len(ch_df['days_relative'].unique()) < 5:
            df = df[df['stim_channel'] != ch]

    df['weeks_relative'] = (df['days_relative'] // 7).astype(int)
    df = df[df['weeks_relative'] < 5]

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

    # Extract unique channels and currents
    stim_channels = get_stim_channels(animal_id)
    stim_conditions = [(ch, curr)
                       for ch in stim_channels for curr in range(3, 7)]

    channels = sorted(list(set([stim[0] for stim in stim_conditions])))
    currents = sorted(list(set([stim[1] for stim in stim_conditions])))

    plt.figure(figsize=(4, 2.5))
    cmap = plt.cm.get_cmap('tab10', len(channels) + 5)

    pulse_index = 1  # Adjust as needed
    weekly_session_indices = []

    # Loop through each channel and current
    for channel_index, channel in enumerate(channels):
        base_color = cmap(channel_index)

        for current_index, current in enumerate(currents):
            stim_condition = (channel, current)

            # Skip if the condition doesn't exist in the dictionary
            if stim_condition not in stim_condition_dict:
                continue

            # Get spike counts and session indices for this condition
            stim_condition_response = stim_condition_dict[stim_condition]['spike_counts']
            session_indices = stim_condition_dict[stim_condition]['session_indices']

            # Convert session indices to weeks and aggregate spike counts by week
            week_spike_counts = np.full(max_week + 1, np.nan)
            for week in range(max_week + 1):
                # Find indices for sessions in the same week
                week_sessions = [i for i, s in enumerate(
                    session_indices) if s // 7 == week]

                # If there are sessions in this week, average their spike counts
                if week_sessions:
                    weekly_counts = [stim_condition_response[i]
                                     [pulse_index] for i in week_sessions]
                    week_spike_counts[week] = np.mean(weekly_counts)

            # Only plot non-NaN values
            valid_weeks = np.where(~np.isnan(week_spike_counts))[0]
            weekly_session_indices.append(valid_weeks)

            line_thickness = 1 + (current_index * 1)
            marker_size = 15 + (current_index * 15)

            plt.scatter(
                valid_weeks, week_spike_counts[valid_weeks], color=base_color, s=marker_size)
            plt.plot(valid_weeks, week_spike_counts[valid_weeks],
                     color=base_color, linewidth=line_thickness,
                     label=f'Ch {channel}, {current} µA')

    # Legend for channels and currents
    channel_legend = [mlines.Line2D([], [], color=cmap(i), label=f'Channel {ch}', linewidth=3)
                      for i, ch in enumerate(channels)]
    current_legend = [mlines.Line2D([], [], color='gray', label=f'Current {curr} µA', linewidth=1 + (i * 0.5))
                      for i, curr in enumerate(currents)]

    plt.legend(handles=channel_legend + current_legend,
               fontsize='x-small', loc='lower left')

    # Labels, title, and layout adjustments
    plt.xlabel('Week Index', fontsize=7)
    plt.ylabel('Normalized Spike Count', fontsize=7)
    plt.title(f'Normalized Spike Counts vs. Weeks for {
              animal_id} (Pulse {pulse_index})', fontsize=7)
    plt.tight_layout()
    plt.show()

#%%
def process_and_aggregate_across_animals(animal_data_dict, stim_channels, max_week=5):
    # Define stimulation conditions for each current
    stim_conditions = [curr for curr in range(3, 7)]

    # Initialize a dictionary to store combined data for all animals by week and current
    aggregated_data = {current: {'spike_counts': [[] for _ in range(max_week + 1)]} for current in stim_conditions}

    # Loop through each animal and its associated session folders
    for animal_id, session_folders in animal_data_dict.items():
        animal_weekly_data = process_stim_conditions_by_week(session_folders, stim_channels, max_week)

        # Aggregate each animal's weekly data into the combined structure
        for current in stim_conditions:
            for week_index, spike_counts in enumerate(animal_weekly_data[current]['spike_counts']):
                if spike_counts:
                    aggregated_data[current]['spike_counts'][week_index].extend(spike_counts)

    return aggregated_data

def plot_aggregated_data_by_current(aggregated_data, max_week=6, plot_type="median"):
    # Define currents to be plotted
    currents = sorted(aggregated_data.keys())
    plt.figure(figsize=(6, 4))
    cmap = plt.cm.get_cmap('tab10', len(currents) + 5)

    for current_index, current in enumerate(currents):
        x_offset = current_index * 0.375 - 0.25
        color = cmap(current_index)
        current_median = []  # Holds median values for line plot
        week_indices = []  # Holds valid week indices

        for week in range(max_week + 1):
            # Retrieve spike counts for the current and week
            spike_counts = aggregated_data[current]['spike_counts'][week]

            # Check if there is data for the current week
            if spike_counts:
                # Calculate median or mean for the current week across animals
                if plot_type == "median":
                    agg_value = np.median(spike_counts)
                else:
                    agg_value = np.mean(spike_counts)

                # Append data for plotting
                current_median.append(agg_value)
                week_indices.append(week)  # Only add week if data is present

                # Plot individual scatter points for each data point
                for count in spike_counts:
                    plt.scatter(week + x_offset, count, color=color, s=10, alpha=0.6)

        # Plot line for median/mean values across weeks with valid data only
        if week_indices:
            plt.plot(week_indices, current_median, color=color, linewidth=2, label=f'Current {current} µA')

    # Set legend, labels, and title
    plt.legend(fontsize='small', loc='upper right')
    plt.xticks(ticks=np.arange(0, max_week + 1, 1))
    plt.xlabel('Week Index')
    plt.ylabel('Normalized Spike Count')
    plt.title(f'{plot_type.capitalize()} Spike Counts by Current vs. Weeks Across All Animals')
    plt.tight_layout()
    plt.show()



# %% Read all animal folder list

def sort_folders(data_folders):
    sorted_data_folders = np.array(file_util.sort_data_folders(data_folders))

    days_relative = np.array(
        file_util.convert_dates_to_relative_days(sorted_data_folders))
    weeks_relative = (days_relative // 7).astype(int)

    # Filter folders to include only those within the first 5 weeks
    good_data_folders = sorted_data_folders[np.where(weeks_relative < 5)[0]]

    return good_data_folders


# Load data and process animal IDs
with open('all_animal_folder_list.pkl', 'rb') as f:
    all_animal_folder_list_0 = pickle.load(f)

# Sort animal IDs based on the numeric portion
animal_ids = np.unique([file_util.get_animal_id(files[0])
                       for files in all_animal_folder_list_0])
numeric_values = np.array([int(s[4:]) for s in animal_ids])
sorted_indices = np.argsort(numeric_values)
sorted_animal_ids = animal_ids[sorted_indices]

# Process each sorted animal ID
for animal_index, animal_id in enumerate(sorted_animal_ids):
    print(animal_id)
    animal_folder_list = all_animal_folder_list_0[sorted_indices[animal_index]]
    good_data_folders = sort_folders(animal_folder_list)
    stim_channels = get_stim_channels(animal_id)

    weekly_data = process_and_aggregate_across_animals(
        good_data_folders, stim_channels, max_week=5)
    plot_aggregated_count_by_week(
        weekly_data, max_week=5, animal_id=animal_id)


# %% Plot population spike rate for each bin
# Aggregate total spikes for all N neurons for first M bins
starting_dir = "C:\\data"
data_folders = file_dialog(starting_dir=starting_dir)

# %%
sorted_data_folders = np.array(file_util.sort_data_folders(data_folders))

days_relative = np.array(
    file_util.convert_dates_to_relative_days(sorted_data_folders))
weeks_relative = (days_relative // 7).astype(int)

good_data_folders = sorted_data_folders[np.where(weeks_relative < 5)[
    0]]  # CUTOFF AT 5

data_folder = good_data_folders[3]  # RANDOM INDEX
animal_id = file_util.get_animal_id(data_folder)
pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"
with open(pkl_path, "rb") as file:
    session_responses = pickle.load(file)
ur = session_responses.get_unit_response(0)

# Get stim channels used in at least 5 sessions
df = load_session_data(animal_id, base_path)
for ch in df['stim_channel'].unique():
    ch_df = df[df['stim_channel'] == ch]
    if len(ch_df['days_relative'].unique()) < 5:
        df = df[df['stim_channel'] != ch]

df['weeks_relative'] = (df['days_relative'] // 7).astype(int)
df = df[df['weeks_relative'] < 5]

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
stim_conditions = [(ch, curr) for ch in stim_channels for curr in range(3, 7)]
stim_condition_dict = {stim_condition: {'spike_counts': [],
                                        'session_indices': []} for stim_condition in stim_conditions}

for folder_index, data_folder in enumerate(good_data_folders):

    print(f'Folder index: {folder_index}')
    pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"

    with open(pkl_path, "rb") as file:
        session_responses = pickle.load(file)

    unit_ids = session_responses.unit_ids
    bins = np.arange(0, 110, 10)  # Define bin edges, 10 pulses

    # Loop over each stimulation condition
    for stim_condition in stim_conditions:

        # Initialize per-condition arrays
        binned_spike_counts_array = np.zeros((len(unit_ids), len(bins) - 1))
        num_trials = None

        for unit_index, unit_id in enumerate(unit_ids):
            ur = session_responses.get_unit_response(unit_id)
            if ur is None:
                continue
            scr = ur.get_stim_response(stim_condition[0], stim_condition[1])

            if scr is None or not is_OK_response(scr):
                continue

            tr = scr.train_response
            train_raster = tr.raster_array

            # Accumulate spike counts in the histogram bins
            if len(train_raster) > 0:
                # Set num_trials once, assuming consistency across units
                num_trials = len(train_raster)
                for row in train_raster:  # Sum up spikes per row in raster array for unit
                    counts, _ = np.histogram(row, bins=bins)
                    binned_spike_counts_array[unit_index, :] += counts

        if num_trials is None or num_trials == 0:
            continue

        # Calculate total spikes across all units and normalize
        total_spikes = np.sum(binned_spike_counts_array, axis=0)
        normalized_spike_counts = total_spikes / (num_trials * len(unit_ids))

        # Append only if `normalized_spike_counts` has non-zero data
        if np.any(normalized_spike_counts):
            stim_condition_dict[stim_condition]['spike_counts'].append(
                normalized_spike_counts)
            stim_condition_dict[stim_condition]['session_indices'].append(
                folder_index)


# %% 10/27
# TODO difference in bins
plt.rcParams.update({'font.size': 8})

# Extract unique channels and currents
channels = sorted(list(set([stim[0] for stim in stim_conditions])))
currents = sorted(list(set([stim[1] for stim in stim_conditions])))

# Determine the maximum session index across all conditions
max_session_index = max(
    max(stim_condition_dict[cond]['session_indices']) for cond in stim_conditions if stim_condition_dict[cond]['session_indices']
)

plt.figure(figsize=(4, 2.5))
# Create a colormap for channels (distinct colors for each channel)
cmap = plt.cm.get_cmap('tab10', len(channels) + 5)

# Set the pulse index (adjust as needed)
pulse_index = 0
valid_session_indices = []
# Loop through each channel and current
for channel_index, channel in enumerate(channels):
    # Get the base color for the channel (same hue for all currents)
    base_color = cmap(channel_index)

    # Loop through currents and vary the thickness and marker size
    for current_index, current in enumerate(currents):
        stim_condition = (channel, current)

        # Skip if the condition doesn't exist in the dictionary
        if stim_condition not in stim_condition_dict:
            continue

        # Get the spike count data and session indices for this condition
        stim_condition_response = stim_condition_dict[stim_condition]['spike_counts']
        session_indices = stim_condition_dict[stim_condition]['session_indices']

        # Create an array of NaNs for all possible sessions and fill available data
        normalized_spike_counts = np.full(max_session_index + 1, np.nan)

        for idx, session_idx in enumerate(session_indices):
            # Only update if the normalized spike count is non-zero
            if stim_condition_response[idx][pulse_index] != 0:
                normalized_spike_counts[session_idx] = stim_condition_response[idx][pulse_index]

        # Line thickness and marker size based on current level
        line_thickness = 1 + (current_index * 1)  # Scale thickness as needed
        marker_size = 15 + (current_index * 15)     # Scale markers as needed

        # Plot only the valid sessions (non-NaN values)
        valid_sessions = np.where(~np.isnan(normalized_spike_counts))[0]
        valid_session_indices.append(valid_sessions)

        plt.scatter(
            valid_sessions, normalized_spike_counts[valid_sessions], color=base_color, s=marker_size)
        plt.plot(valid_sessions, normalized_spike_counts[valid_sessions],
                 color=base_color, linewidth=line_thickness,
                 label=f'Ch {channel}, {current} µA')

# Create proxy legend entries for channels (colors)
channel_legend = [mlines.Line2D([], [], color=cmap(i), label=f'Channel {ch}', linewidth=3)
                  for i, ch in enumerate(channels)]

# Create proxy legend entries for currents (line thickness)
current_legend = [mlines.Line2D([], [], color='gray', label=f'Current {curr} µA', linewidth=1 + (i * 0.5))
                  for i, curr in enumerate(currents)]

# Combine channel and current legends
plt.legend(handles=channel_legend + current_legend,
           fontsize='x-small', loc='lower left')

# Labels, title, and grid
plt.xlabel('Session Index', fontsize=7)
plt.ylabel('Normalized Spike Count', fontsize=7)
plt.title(f'Normalized Spike Counts vs. Sessions for {
          animal_id} (Pulse {pulse_index})', fontsize=7)
plt.tight_layout()
plt.show()


all_valid_sessions = np.concatenate(valid_session_indices)
good_session_indices = np.unique(all_valid_sessions)
good_sessions = good_data_folders[good_session_indices]


# %%

# Extract all unique dates and sort
all_dates = sorted(set(date for ch in thresholds_over_time for date, _ in thresholds_over_time[ch]),
                   key=lambda d: datetime.strptime(d, '%d-%b-%Y'))
# Map each date to index starting from 0
date_to_index = {date: idx for idx, date in enumerate(all_dates)}

plt.figure(figsize=(4, 2.5))

for ch, data in thresholds_over_time.items():
    # Convert dates to indices and extract threshold values
    session_indices = [date_to_index[date] for date, _ in data]
    thresholds = [threshold for _, threshold in data]

    # Plot the data for each channel
    plt.plot(session_indices, thresholds, label=f'Channel {ch}')

# Step 4: Configure x-axis with date labels
plt.xlabel('Session Index')
plt.ylabel('Detection Threshold (uA)')
plt.title(f'Detection Thresholds Over Time {animal_id}')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Identify unique channels and currents in stim_conditions
channels = sorted(list(set([stim[0] for stim in stim_conditions])))
currents_per_channel = {channel: set(
    [stim[1] for stim in stim_conditions if stim[0] == channel]) for channel in channels}

# Find currents common to all channels
common_currents = set.intersection(*currents_per_channel.values())
common_currents = sorted(list(common_currents))

plt.figure(figsize=(6, 4))
# Set color map for different common currents
cmap = plt.cm.get_cmap('tab10', len(common_currents))

# Set pulse index for the data you want to visualize
pulse_index = 0

# Dictionary to store aggregated spike counts and session indices for each common current
aggregated_spike_counts = {current: [] for current in common_currents}
aggregated_session_indices = {current: [] for current in common_currents}

# Aggregate spike counts across all channels for each common current
for stim_condition in stim_conditions:
    channel, current = stim_condition

    # Only process if current is in the list of common currents
    if current not in common_currents or stim_condition not in stim_condition_spike_count_dict:
        continue

    # Get the spike count data and session index for this condition
    stim_condition_response = np.array(
        stim_condition_spike_count_dict[stim_condition])
    stim_condition_session_index = stim_condition_session_index_dict[stim_condition]

    if len(stim_condition_response) == 0:
        continue

    # Append data to the respective current in the dictionary
    aggregated_spike_counts[current].append(
        stim_condition_response[:, pulse_index])
    aggregated_session_indices[current].append(stim_condition_session_index)

# Plot the average spike counts for each common current
for current_index, current in enumerate(common_currents):
    base_color = cmap(current_index)

    # Align responses by session indices and calculate the mean across channels
    if len(aggregated_spike_counts[current]) > 0:
        # Find the unique session indices across all channels for this current
        all_session_indices = sorted(
            set(np.concatenate(aggregated_session_indices[current])))

        # Initialize an array to hold aligned spike counts
        aligned_spike_counts = []

        # Align each channel's spike data by session indices
        for i, session_indices in enumerate(aggregated_session_indices[current]):
            response = aggregated_spike_counts[current][i]
            aligned_response = np.full(len(all_session_indices), np.nan)
            session_mask = np.in1d(all_session_indices, session_indices)
            aligned_response[session_mask] = response
            aligned_spike_counts.append(aligned_response)

        # Stack aligned responses and compute mean across channels
        stacked_responses = np.nanmean(aligned_spike_counts, axis=0)

        # Plot the mean response for the current
        line_thickness = 1 + (current_index * 1)
        marker_size = 20 + (current_index * 30)

        plt.scatter(all_session_indices, stacked_responses,
                    color=base_color, s=marker_size)
        plt.plot(all_session_indices, stacked_responses, color=base_color, linewidth=line_thickness,
                 label=f'{current} µA')

# Create a legend for common currents
current_legend = [mlines.Line2D([], [], color=cmap(i), label=f'Current {curr} µA', linewidth=1 + (i * 2))
                  for i, curr in enumerate(common_currents)]

# Add the legend to the plot
plt.legend(handles=current_legend, fontsize='smaller')

# Labels and title
plt.xlabel('Session Index')
plt.ylabel('Average Normalized Spike Count')
plt.title(f'{animal_id} Averaged Normalized Spike Counts after Pulse {
          pulse_index}', fontsize=10)
plt.tight_layout()

# Save and show the plot
plt.savefig(f"images/{animal_id}_averaged_spike_count_pulse_{pulse_index}.png")
plt.show()


# %% Train response
starting_dir = "C:\\data"
data_folders = file_dialog(starting_dir=starting_dir)
sorted_data_folders = file_util.sort_data_folders(data_folders)

ppt_inserter = PPTImageInserter(grid_dims=(
    2, 2), spacing=(0.05, 0.05), title_font_size=16)

colors = {1: 'gray', 2: 'C0', 3: 'C1', 4: 'C2',
          5: 'C3', 6: 'C4', 7: 'C6', 8: 'C7', 9: 'C8', 10: 'C9', 11: 'C10', 12: 'k', 13: 'k'}
bins = np.arange(-700, 1400, 10)
bin_centers = (bins[:-1] + bins[1:]) / 2


for data_folder in sorted_data_folders:

    date_str = file_util.get_date_str(data_folder)
    animal_id = file_util.get_animal_id(data_folder)

    pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"

    with open(pkl_path, "rb") as file:
        session_responses = pickle.load(file)

    unit_ids = session_responses.unit_ids

    stim_conditions = session_responses.get_unit_response(
        unit_id=1).show_stim_conditions()

    stim_channels = set([sc[0] for sc in stim_conditions]
                        )  # Unique stimulation channels

    ppt_inserter.add_slide(f'{animal_id}: {date_str}')
    for stim_channel in stim_channels:

        plt.figure()

        # Loop over all currents for the same stim channel
        for stim_condition in [sc for sc in stim_conditions if sc[0] == stim_channel]:
            stim_current = stim_condition[1]
            neuron_total_counts = []
            neuron_total_trials = []
            valid_neurons_for_sc = 0

            for unit_index, unit_id in enumerate(unit_ids):
                ur = session_responses.get_unit_response(unit_id)
                scr = ur.get_stim_response(
                    stim_condition[0], stim_current)
                tr = scr.train_response
                pr = scr.pulse_response

                if not is_good_train_response(scr):
                    continue

                valid_neurons_for_sc += 1

                train_raster = tr.raster_array

                # Bins will have 1 less edge than the number of bins
                cumulative_counts = np.zeros(len(bins) - 1)

                # Loop through each row and sum the counts into the cumulative array
                for row in train_raster:
                    counts, _ = np.histogram(row, bins=bins)
                    cumulative_counts += (counts > 0).astype(int)

                neuron_total_counts.append(cumulative_counts)
                neuron_total_trials.append(len(train_raster))

            # Convert the list to an array for total activation probability calculation
            if len(neuron_total_counts) == 0:
                continue

            neuron_total_counts = np.array(neuron_total_counts)

            # Sum the counts across all neurons and normalize by the total number of trials
            total_trials = np.sum(neuron_total_trials)
            total_prob = np.sum(neuron_total_counts, axis=0) / total_trials

            # Plot total activation probability for this current
            plt.plot(bin_centers, total_prob, linewidth=2, color=colors[stim_current],
                     label=f'{stim_current} uA')

            plt.title(f'{date_str}: Total Activation Probability for Stim Ch.{
                      stim_channel}, {valid_neurons_for_sc} neurons')

        # Add labels, grid, and legend
        plt.xlabel('Time (ms)')
        plt.ylabel('Activation Probability (All Neurons)')
        plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)
        plt.ylim(0, 1)
        plt.axvline(0, linestyle='--', color='k')
        plt.axvline(700, linestyle='--', color='k')
        plt.legend(title='Current')
        plt.show()

        plt.savefig("test.png")
        ppt_inserter.add_image("test.png")
        plt.close()

ppt_inserter.save(f"{animal_id}_stim_response_probability.pptx")

# %%
starting_dir = "C:\\data"
data_folders = file_dialog(starting_dir=starting_dir)
sorted_data_folders = file_util.sort_data_folders(data_folders)

ppt_inserter = PPTImageInserter(grid_dims=(
    1, 1), spacing=(0.05, 0.2), title_font_size=16)

for data_folder in sorted_data_folders:

    date_str = file_util.get_date_str(data_folder)
    animal_id = file_util.get_animal_id(data_folder)

    pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"

    with open(pkl_path, "rb") as file:
        session_responses = pickle.load(file)

    unit_ids = session_responses.unit_ids

    stim_conditions = session_responses.get_unit_response(
        unit_id=1).show_stim_conditions()

    stim_channels = set([sc[0] for sc in stim_conditions]
                        )  # Unique stimulation channels

    ppt_inserter.add_slide(f'{animal_id}: {date_str}, {len(unit_ids)} neurons')

    for stim_channel in stim_channels:

        # Find all unique currents for the current stim channel
        unique_currents = sorted(
            set([sc[1] for sc in stim_conditions if sc[0] == stim_channel]))

        n_currents = len(unique_currents)
        # Create grid with enough subplots
        grid_size = math.ceil(np.sqrt(n_currents))

        # Create subplots with the determined grid size
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(12, 8))
        axs = axs.flatten()  # Flatten to easily index into individual subplots

        colors = sns.color_palette("husl", len(unit_ids))

        # Loop over all unique currents for the same stim channel
        for current_index, stim_current in enumerate(unique_currents):

            ax = axs[current_index]  # Select the appropriate subplot
            for unit_index, unit_id in enumerate(unit_ids):
                ur = session_responses.get_unit_response(unit_id)
                scr = ur.get_stim_response(stim_channel, stim_current)
                pr = scr.pulse_response

                if not is_good_pulse_response(scr):
                    continue

                # Plot firing rate for each unit
                ax.plot(pr.fr_times, pr.firing_rate,
                        color=colors[unit_index], label=f'Unit {unit_id}')

                for boundary in boundaries:
                    ax.axvline(x=boundary, linestyle='--', color='black')

            ax.legend(loc='upper right', fontsize='small')
            ax.set_ylim([0, 250])
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Firing rate (Hz)')
            ax.set_title(f'{stim_channel} at {stim_current} uA')

        # Remove unused subplots
        for i in range(n_currents, len(axs)):
            fig.delaxes(axs[i])  # Remove unused axes

        # Add overall title and layout adjustment
        plt.suptitle(f'{date_str}: Pulse Response to Stim Ch.{
                     stim_channel}', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Adjust for the suptitle
        plt.show()

        plt.savefig("test.png")
        ppt_inserter.add_image("test.png")
        plt.close()

ppt_inserter.save(f"{animal_id}_stim_pulse_response_fr.pptx")

# %% Cluster latencies
all_animal_folder_list_0 = []
for i in range(5):
    starting_dir = "C:\\data"
    data_folders = file_dialog(starting_dir=starting_dir)
    sorted_data_folders = file_util.sort_data_folders(data_folders)
    all_animal_folder_list_0.append(sorted_data_folders)

# %%
# with open('all_animal_folder_list.pkl', 'wb') as f:
#     pickle.dump(all_animal_folder_list_0, f)

with open('all_animal_folder_list.pkl', 'rb') as f:
    all_animal_folder_list_0 = pickle.load(f)

all_animal_folder_list = [
    item for sublist in all_animal_folder_list_0 for item in sublist]
file_animal_ids = [file_util.get_animal_id(f) for f in all_animal_folder_list]
animal_ids = np.unique(file_animal_ids)

new_folder_list = []

for animal_id in animal_ids:
    data_folders = [folder for folder, id_ in zip(
        all_animal_folder_list, file_animal_ids) if id_ == animal_id]

    # Sort the data folders for this animal
    sorted_data_folders = np.array(file_util.sort_data_folders(data_folders))

    # Compute relative days and weeks
    days_relative = np.array(
        file_util.convert_dates_to_relative_days(sorted_data_folders))
    weeks_relative = (days_relative // 7).astype(int)

    # Filter out folders where the relative week is greater than or equal to 5
    good_data_folders = sorted_data_folders[np.where(weeks_relative < 5)[0]]

    # Use extend() instead of append() to avoid creating sublists
    new_folder_list.extend(good_data_folders)


# %%
latencies = []
for data_folder in all_animal_folder_list:

    date_str = file_util.get_date_str(data_folder)
    animal_id = file_util.get_animal_id(data_folder)

    pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"

    with open(pkl_path, "rb") as file:
        session_responses = pickle.load(file)

    unit_ids = session_responses.unit_ids

    stim_conditions = session_responses.get_unit_response(
        unit_id=1).show_stim_conditions()

    stim_channels = set([sc[0] for sc in stim_conditions])

    for stim_channel in stim_channels:

        unique_currents = sorted(
            set([sc[1] for sc in stim_conditions if sc[0] == stim_channel]))

        n_currents = len(unique_currents)

        for current_index, stim_current in enumerate(unique_currents):

            for unit_index, unit_id in enumerate(unit_ids):
                ur = session_responses.get_unit_response(unit_id)
                scr = ur.get_stim_response(stim_channel, stim_current)
                pr = scr.pulse_response

                if not is_good_pulse_response(scr):
                    continue

                latency = pr.fr_times[np.argmax(pr.firing_rate)]

                latencies.append(latency)

# %%
data = np.array(latencies).reshape(-1, 1)

# Number of clusters
n_clusters = 3

# Apply KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(data)

# Get cluster centers
centers = np.sort(kmeans.cluster_centers_.flatten())

# Calculate the midpoints between the cluster centers (these are the boundaries)
boundaries = [(centers[i] + centers[i+1]) / 2 for i in range(len(centers) - 1)]

plt.hist(latencies, 20)
for boundary in boundaries:
    plt.axvline(x=boundary, linestyle='--', color='black',
                label=f'Boundary at {boundary:.2f}')


def classify_latencies(latency, boundaries):
    # Number of classes is one more than the number of boundaries
    n = len(boundaries) + 1
    for i in range(len(boundaries)):
        if latency < boundaries[i]:
            return i
    return n - 1

# %%

# Define boundaries for classifying latencies


def compute_proportions(latency_group_data, boundaries):
    """
    Compute proportions of latency groups for each week.

    Args:
        latency_group_data (dict): Nested dictionary {week_relative: {latency_group: count}}.
        boundaries (list): List of boundaries for latency classification.

    Returns:
        dict: Proportions of each latency group for each week {group: [proportion per week]}.
        list: Sorted list of weeks for plotting.
    """
    weeks = sorted(latency_group_data.keys())
    # Number of groups is one more than number of boundaries
    num_groups = len(boundaries) + 1

    # Initialize an empty dictionary to store the proportions of each group for each week
    proportions_by_week = {group: [] for group in range(num_groups)}

    # Calculate proportions for each week
    for week in weeks:
        # Total counts for the week
        total_count = sum(latency_group_data[week].values())
        for group in range(num_groups):
            # Get the count for the group, default to 0 if not present
            count = latency_group_data[week].get(group, 0)
            # Calculate proportion (avoid division by zero by setting proportion to 0 if total_count is 0)
            proportion = count / total_count if total_count > 0 else 0
            proportions_by_week[group].append(proportion)

    return proportions_by_week, weeks


def plot_proportions(proportions_by_week, weeks, title):
    """
    Plot the proportions of latency groups over weeks.

    Args:
        proportions_by_week (dict): Proportions of each latency group {group: [proportion per week]}.
        weeks (list): Sorted list of weeks.
        title (str): Plot title.
    """
    plt.figure(figsize=(10, 6))
    num_groups = len(proportions_by_week)

    for group in range(num_groups):
        plt.plot(weeks, proportions_by_week[group], label=f'Group {group}')

    plt.xlabel('Weeks Relative')
    plt.ylabel('Proportion of Latency Groups')
    plt.xticks(ticks=np.arange(min(weeks), max(weeks) + 1, 1))
    plt.title(title)
    plt.legend(title='Latency Group')
    plt.grid(True)
    plt.show()


def compute_counts(latency_group_data, boundaries):
    """
    Compute counts of latency groups for each week.

    Args:
        latency_group_data (dict): Nested dictionary {week_relative: {latency_group: count}}.
        boundaries (list): List of boundaries for latency classification.

    Returns:
        dict: Counts of each latency group for each week {group: [counts per week]}.
        list: Sorted list of weeks for plotting.
    """
    weeks = sorted(latency_group_data.keys())
    # Number of groups is one more than number of boundaries
    num_groups = len(boundaries) + 1

    # Initialize an empty list to store the counts of each group for each week
    counts_by_week = {group: [] for group in range(num_groups)}

    # Collect counts for each week
    for week in weeks:
        for group in range(num_groups):
            # Get the count for the group in this week, default to 0 if no count is present
            count = latency_group_data[week].get(group, 0)
            counts_by_week[group].append(count)

    return counts_by_week, weeks


def plot_counts(counts_by_week, weeks, title):
    """
    Plot the counts of latency groups over weeks.

    Args:
        counts_by_week (dict): Counts of each latency group {group: [proportions per week]}.
        weeks (list): Sorted list of weeks.
        title (str): Plot title.
    """
    plt.figure(figsize=(10, 6))
    num_groups = len(counts_by_week)

    for group in range(num_groups):
        plt.plot(weeks, counts_by_week[group], label=f'Group {group}')

    plt.xlabel('Weeks Relative')
    plt.ylabel('Counts of Latency Groups')
    plt.xticks(ticks=np.arange(min(weeks), max(weeks) + 1, 1))
    plt.title(title)
    plt.legend(title='Latency Group')
    plt.grid(True)
    plt.show()


# Main loop: Calculate latency groups for both individual animals and aggregated data
latency_group_by_animal = defaultdict(
    lambda: defaultdict(lambda: defaultdict(int)))
latency_group_aggregated = defaultdict(lambda: defaultdict(int))

for data_folder in all_animal_folder_list:

    date_str = file_util.get_date_str(data_folder)
    animal_id = file_util.get_animal_id(data_folder)

    pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"

    with open(pkl_path, "rb") as file:
        session_responses = pickle.load(file)

    unit_ids = session_responses.unit_ids

    stim_conditions = session_responses.get_unit_response(
        unit_id=1).show_stim_conditions()

    stim_channels = set([sc[0] for sc in stim_conditions])

    for stim_channel in stim_channels:

        unique_currents = sorted(
            set([sc[1] for sc in stim_conditions if sc[0] == stim_channel]))

        for stim_current in unique_currents:

            for unit_index, unit_id in enumerate(unit_ids):
                ur = session_responses.get_unit_response(unit_id)
                scr = ur.get_stim_response(stim_channel, stim_current)
                pr = scr.pulse_response

                if not is_good_pulse_response(scr):
                    continue

                # Calculate latency
                latency = pr.fr_times[np.argmax(pr.firing_rate)]

                # Classify the latency into a group
                latency_group = classify_latencies(latency, boundaries)

                # Get the relative week
                week_relative = get_week_relative(
                    data_folder, all_animal_folder_list_0)

                if week_relative > 4:
                    continue

                # Increment the count for this week for both the individual animal and aggregated
                latency_group_by_animal[animal_id][week_relative][latency_group] += 1
                latency_group_aggregated[week_relative][latency_group] += 1

# %%
# Plot for each individual animal
for animal_id, latency_group_by_week in latency_group_by_animal.items():
    proportions_by_week, weeks = compute_proportions(
        latency_group_by_week, boundaries)
    plot_proportions(proportions_by_week, weeks,
                     f'Proportion of Latency Groups Over Time for Animal {animal_id}')

# Plot for the aggregated data
proportions_by_week_aggregated, weeks_aggregated = compute_proportions(
    latency_group_aggregated, boundaries)
plot_proportions(proportions_by_week_aggregated, weeks_aggregated,
                 'Aggregated Proportion of Latency Groups Over Time')

# Plot counts
for animal_id, latency_group_by_week in latency_group_by_animal.items():
    counts_by_week, weeks = compute_counts(
        latency_group_by_week, boundaries)
    plot_counts(counts_by_week, weeks,
                f'Counts of Latency Groups Over Time for Animal {animal_id}')

# Plot for the aggregated data
counts_by_week_aggregated, weeks_aggregated = compute_counts(
    latency_group_aggregated, boundaries)
plot_counts(counts_by_week_aggregated, weeks_aggregated,
            'Aggregated Counts of Latency Groups Over Time')

# %% Plot number of units across weeks for all animals


# %%
# Initialize lists to store counts across animals
all_unit_counts_agg = []
modulated_unit_counts_agg = []
pulse_locked_unit_counts_agg = []

# Loop through each animal and collect the unit counts
for animal_id in animal_ids:
    base_path = Path("C:/data/")  # Replace with your base path
    original_df = load_session_data(animal_id, base_path)
    original_df['weeks_relative'] = (
        original_df['days_relative'] // 7).astype(int)
    df = filter_data(original_df)

    relative_weeks = df['weeks_relative'].unique()

    all_unit_counts = []
    modulated_unit_counts = []
    pulse_locked_unit_counts = []

    for relative_week in relative_weeks:
        original_session_df = original_df[original_df['weeks_relative']
                                          == relative_week]
        session_df = df[df['weeks_relative'] == relative_week]
        pulse_locked_df = session_df[session_df['is_pulse_locked'] == True]

        all_unit_counts.append(len(original_session_df['unit_id'].unique()))
        modulated_unit_counts.append(len(session_df['unit_id'].unique()))
        pulse_locked_unit_counts.append(
            len(pulse_locked_df['unit_id'].unique()))

    # Add the counts from this animal to the aggregated lists
    all_unit_counts_agg.append(all_unit_counts)
    modulated_unit_counts_agg.append(modulated_unit_counts)
    pulse_locked_unit_counts_agg.append(pulse_locked_unit_counts)


all_unit_counts_agg = [np.array(counts, dtype=float)
                       for counts in all_unit_counts_agg]
modulated_unit_counts_agg = [
    np.array(counts, dtype=float) for counts in modulated_unit_counts_agg]
pulse_locked_unit_counts_agg = [
    np.array(counts, dtype=float) for counts in pulse_locked_unit_counts_agg]

# Determine the maximum length of sessions across animals
max_length = max(len(lst) for lst in all_unit_counts_agg)

# Pad arrays to align lengths across animals
all_unit_counts_agg = np.array([np.pad(
    counts, (0, max_length - len(counts)), constant_values=np.nan) for counts in all_unit_counts_agg])
modulated_unit_counts_agg = np.array([np.pad(counts, (0, max_length - len(
    counts)), constant_values=np.nan) for counts in modulated_unit_counts_agg])
pulse_locked_unit_counts_agg = np.array([np.pad(counts, (0, max_length - len(
    counts)), constant_values=np.nan) for counts in pulse_locked_unit_counts_agg])

# Calculate the sum counts across animals, ignoring NaNs
sum_all_unit_counts = np.nansum(all_unit_counts_agg, axis=0)
sum_modulated_unit_counts = np.nansum(modulated_unit_counts_agg, axis=0)
sum_pulse_locked_unit_counts = np.nansum(pulse_locked_unit_counts_agg, axis=0)

# Plot the aggregated results
plt.figure(figsize=(4, 2.5))
plt.plot(sum_all_unit_counts, label='All Units', color='C0')
plt.plot(sum_modulated_unit_counts, label='Modulated Units', color='C1')
plt.plot(sum_pulse_locked_unit_counts, label='Pulse-locked Units', color='C2')
plt.legend()
plt.xlabel("Week")
plt.ylabel("Total Unit Count")
plt.title("Total Unit Counts Across Animals")
plt.show()


# %% Max to average ratio
dfs = [load_session_data(animal_id, base_path) for animal_id in animal_ids]
df = pd.concat(dfs, ignore_index=True)
df = filter_data(df)


train_indices = np.where((df['fr_times'].iloc[0] >= 0)
                         & (df['fr_times'].iloc[0] <= 700))

df['max_fr'] = df['firing_rate'].apply(lambda x: np.max(
    np.array(x)[train_indices]) if isinstance(x, (list, np.ndarray)) else x)
df['avg_fr'] = df['firing_rate'].apply(lambda x: np.mean(
    np.array(x)[train_indices]) if isinstance(x, (list, np.ndarray)) else x)
df['max_to_avg_fr_ratio'] = df['max_fr'] / df['avg_fr']

fig, ax = plt.subplots()
p_values_var1, p_values_var2 = plot_aggregated_weekly_data_with_iqr(
    df, var1='max_to_avg_fr_ratio', ax1=ax, last_week=4)
ax.set_ylabel("Milliseconds")
ax.set_title("Latency to train response peak")

# %%
