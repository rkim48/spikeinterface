from util.file_util import *
from util.load_data import save_block_detection_thresholds, get_dataframe
from pathlib import Path
import pickle
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import re
import numpy as np


def write_thresholds(sorted_data_folders):
    # batch process thresholds
    # write session thresholds to pickle file in date folder
    for data_folder in sorted_data_folders:
        save_block_detection_thresholds(data_folder)
    return data_folders


def read_thresholds(data_folders):
    all_session_thresholds = {}
    for data_folder in data_folders:
        with open(Path(data_folder) / 'block_thresholds.pkl', 'rb') as file:
            block_thresholds = pickle.load(file)
            all_session_thresholds[data_folder] = block_thresholds
    return all_session_thresholds


def get_avg_channel_thresholds(data_folder):
    sums = {}
    with open(Path(data_folder) / 'block_thresholds.pkl', 'rb') as file:
        block_thresholds = pickle.load(file)
    for block in block_thresholds.values():
        for key, value in block.items():
            # Add the value to the sum for this key
            if key in sums:
                sums[key] += value
            else:
                sums[key] = value

    # Calculate the average for each key
    num_blocks = len(block_thresholds)
    averages = {key: value / num_blocks for key, value in sums.items()}
    return averages


def parse_date_from_path(path):
    date_str = path.split('\\')[-1]  # Extract the date part of the string
    return datetime.datetime.strptime(date_str, '%d-%b-%Y')  # Parse the date

# Function to read and process thresholds


def process_thresholds(data):
    # Prepare lists to hold the parsed data
    dates, blocks, channels, thresholds, relative_days = [], [], [], [], []

    # Sort the dates
    sorted_dates = sorted(data, key=lambda x: datetime.datetime.strptime(
        x.split('\\')[-1], '%d-%b-%Y'))

    # Earliest date for relative day calculation
    earliest_date = datetime.datetime.strptime(
        sorted_dates[0].split('\\')[-1], '%d-%b-%Y')

    for date_path in sorted_dates:
        date_str = date_path.split('\\')[-1]
        date_obj = datetime.datetime.strptime(date_str, '%d-%b-%Y')
        relative_day = (date_obj - earliest_date).days

        for block, channel_dict in data[date_path].items():
            for channel, threshold in channel_dict.items():
                dates.append(date_obj)
                blocks.append(block)
                channels.append(channel)
                thresholds.append(threshold)
                relative_days.append(relative_day)

    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Date'
        'Relative Day': relative_days,
        'Block': blocks,
        'Channel': channels,
        'Threshold': thresholds
    })

    return df


def plot_thresholds_over_time(df):

    palette = sns.color_palette("deep")

    # Assuming 'df' is your DataFrame
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=df, x='Date', y='Threshold',
                 hue='Channel', errorbar='sd', palette=palette)

    plt.title('Thresholds Over Time by Channel')
    # plt.xlabel('Days of training')
    plt.xlabel('')
    plt.ylabel('Threshold')
    plt.xticks(rotation=45)
    plt.legend(title='Channel')
    plt.tight_layout()
    plt.show()

    # plt.figure(figsize=(10, 6))
    # sns.boxplot(data=df, x='Channel', y='Threshold')
    # plt.title('Threshold Distribution by Channel')
    # plt.xlabel('Channel')
    # plt.ylabel('Threshold')
    # plt.show()


def get_hits_and_currents_per_ch(df):

    assert len(df) == 400
    trials_per_block = 100  # Each block has 100 trials
    hits_and_currents_per_ch = {}

    for block in range(4):
        start_idx = trials_per_block * block
        end_idx = start_idx + trials_per_block
        sub_df = df.iloc[start_idx:end_idx]

        unique_channels = sorted(
            sub_df[sub_df['channel'] != 0]['channel'].unique())
        channel_hits = {}
        channel_currents = {}

        for channel in unique_channels:
            channel_data = sub_df[sub_df['channel'] == channel]
            channel_currents[channel] = np.sort(
                channel_data['current'].unique()).tolist()
            current_hits = []
            for current in channel_currents[channel]:
                current_data = channel_data[channel_data['current'] == current]
                current_hits.append(current_data['response'].sum())
            channel_hits[channel] = current_hits

        hits_and_currents_per_ch[block] = {
            'hits': channel_hits, 'currents': channel_currents}

    return hits_and_currents_per_ch


def get_hits_and_currents_dict(sorted_data_folders):
    hits_and_currents_dict = {}
    for data_folder in sorted_data_folders:
        df = get_dataframe(data_folder)
        hits_and_currents_per_ch = get_hits_and_currents_per_ch(df)

        hits_and_currents_dict[data_folder] = hits_and_currents_per_ch
    return hits_and_currents_dict


def find_closest_current(hits, currents):
    target_hits = 5
    closest_difference = float('inf')
    closest_current = None
    equidistant_hits = []

    # Exclude the third hit if it's less than the second hit
    if hits[2] < hits[1]:
        hits = hits[:2]
        currents = currents[:2]

    for hit, current in zip(hits, currents):
        difference = abs(hit - target_hits)
        if difference < closest_difference:
            closest_difference = difference
            closest_current = current
            equidistant_hits = [(hit, current)]
        elif difference == closest_difference:
            equidistant_hits.append((hit, current))

    if len(equidistant_hits) > 1:
        # Pick the current associated with the middle hit from the original order
        middle_index = len(hits) // 2
        closest_current = currents[middle_index]

    return closest_current


def test_find_closest_current():
    test_cases = [
        ([4, 7, 10], [1, 2, 3], 1),  # choose low
        ([1, 7, 10], [1, 2, 3], 2),  # choose middle
        ([1, 3, 6], [1, 2, 3], 3),   # choose high
        ([4, 6, 10], [1, 2, 3], 2),  # Equidistant from 5, choose middle
        ([2, 8, 9], [1, 2, 3], 2),   # Equidistant from 5, choose middle
        ([0, 4, 6], [1, 2, 3], 2),   # Equidistant from 5, choose middle
        ([3, 7, 10], [1, 2, 3], 2),  # Equidistant from 5, choose middle
        ([5, 5, 5], [1, 2, 3], 2),   # All hits are 5, choose middle current
        ([10, 10, 10], [1, 2, 3], 2),  # All hits are 10, choose middle current
        ([2, 2, 8], [1, 2, 3], 2),    # All equidistant, choose middle
        # Third val is less than middle, so compare first 2
        ([4, 8, 6], [1, 2, 3], 1),
        ([1, 7, 8], [1, 2, 3], 2),
        ([4, 10, 6], [1, 2, 3], 1),
    ]

    for hits, currents, expected in test_cases:
        result = find_closest_current(hits, currents)
        assert result == expected, f"Failed for hits={hits}, currents={currents}. Expected {expected}, got {result}."
    print("All test cases passed!")


def is_non_decreasing_with_exceptions(hits, allowed_exceptions=1, min_increasing_or_constant_steps=1, max_allowed_drop=3):
    exceptions = sum(y < x for x, y in zip(hits, hits[1:]))
    increasing_or_constant_steps = sum(y >= x for x, y in zip(hits, hits[1:]))
    largest_drop = max((x - y)
                       for x, y in zip(hits, hits[1:])) if exceptions > 0 else 0
    return (exceptions <= allowed_exceptions and
            increasing_or_constant_steps >= min_increasing_or_constant_steps and
            largest_drop <= max_allowed_drop)


def inclusion_criteria(ch_a_hits, ch_b_hits, ch_c_hits):
    channels_with_sufficient_hits = sum(max(hits) >= 4 for hits in [
                                        ch_a_hits, ch_b_hits, ch_c_hits])
    channels_with_acceptable_trend = sum(is_non_decreasing_with_exceptions(
        hits) for hits in [ch_a_hits, ch_b_hits, ch_c_hits])
    return 1 if channels_with_sufficient_hits >= 2 and channels_with_acceptable_trend >= 2 else 0


def test_inclusion_criteria():
    include_test_cases = [
        # All channels with mostly increasing hits
        ([3, 5, 4], [5, 6, 7], [3, 4, 5], 1),
        # Two channels with mostly increasing hits
        ([3, 4, 3], [5, 6, 5], [4, 5, 6], 1),
        # All channels with max hits >= 4 and mostly increasing hits
        ([10, 10, 10], [10, 10, 10], [4, 6, 8], 1),
        # All channels with mostly increasing hits
        ([4, 5, 6], [7, 6, 5], [5, 5, 5], 1),
        # All channels with mostly increasing hits and one with constant hits
        ([4, 5, 6], [7, 6, 5], [5, 5, 4], 1)
    ]

    exclude_test_cases = [
        # Channels with extreme decreases in hits
        ([4, 0, 0], [5, 2, 1], [3, 4, 5], 0),
        # No channels with max hits >= 4
        ([3, 0, 0], [2, 2, 1], [1, 2, 3], 0),
        # Only one channel with max hits >= 4
        ([0, 3, 4], [2, 2, 1], [1, 2, 3], 0),
        # No channels with max hits >= 4
        ([3, 3, 3], [2, 2, 2], [1, 1, 1], 0),
    ]

    print("Include Test Cases:")
    for idx, (ch_a_hits, ch_b_hits, ch_c_hits, expected) in enumerate(include_test_cases, 1):
        result = inclusion_criteria(ch_a_hits, ch_b_hits, ch_c_hits)
        assert result == expected, f"Include Test case {idx} failed. Expected {expected}, got {result}."

    print("\nExclude Test Cases:")
    for idx, (ch_a_hits, ch_b_hits, ch_c_hits, expected) in enumerate(exclude_test_cases, 1):
        result = inclusion_criteria(ch_a_hits, ch_b_hits, ch_c_hits)
        assert result == expected, f"Exclude Test case {idx} failed. Expected {expected}, got {result}."


def write_csv_file(sorted_data_folders, hits_and_currents_dict, csv_file_name):
    all_session_thresholds = read_thresholds(sorted_data_folders)

    # Process data and write to CSV
    with open(csv_file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Session', 'Block',
                         'ChA', 'CurrentsA', 'HitsA', 'ThresholdA', 'Closest CurrentA to 50%',
                         'ChB', 'CurrentsB', 'HitsB', 'ThresholdB', 'Closest CurrentB to 50%',
                         'ChC', 'CurrentsC', 'HitsC', 'ThresholdC', 'Closest CurrentC to 50%',
                         'Include'])

        for session, blocks in all_session_thresholds.items():
            block_num = 0  # Process only the first block
            block_hits_and_currents = hits_and_currents_dict[session][block_num]
            block_thresholds = blocks[block_num]

            # Convert block number to 1-indexed
            block_index = block_num + 1

            # Sort channels by key (channel number)
            sorted_channels = sorted(
                block_thresholds.items(), key=lambda x: x[0])
            channel_numbers = [channel[0] for channel in sorted_channels]
            thresholds = [channel[1] for channel in sorted_channels]

            ch_a_hits = list(
                block_hits_and_currents['hits'].get(channel_numbers[0], 0))
            ch_b_hits = list(
                block_hits_and_currents['hits'].get(channel_numbers[1], 0))
            ch_c_hits = list(
                block_hits_and_currents['hits'].get(channel_numbers[2], 0))

            ch_a_curr = block_hits_and_currents['currents'].get(
                channel_numbers[0], [])
            ch_b_curr = block_hits_and_currents['currents'].get(
                channel_numbers[1], [])
            ch_c_curr = block_hits_and_currents['currents'].get(
                channel_numbers[2], [])

            # Find closest currents to 50% for each channel
            closest_a = find_closest_current(ch_a_hits, ch_a_curr)
            closest_b = find_closest_current(ch_b_hits, ch_b_curr)
            closest_c = find_closest_current(ch_c_hits, ch_c_curr)

            include = inclusion_criteria(ch_a_hits, ch_b_hits, ch_c_hits)

            # Write row to CSV
            writer.writerow([session, block_index] +
                            [channel_numbers[0], ch_a_curr, ch_a_hits, thresholds[0], closest_a,
                             channel_numbers[1], ch_b_curr, ch_b_hits, thresholds[1], closest_b,
                             channel_numbers[2], ch_c_curr, ch_c_hits, thresholds[2], closest_c,
                             include])

    return hits_and_currents_dict


def write_csv_file(sorted_data_folders, hits_and_currents_dict, csv_file_name):
    all_session_thresholds = read_thresholds(sorted_data_folders)
    column_names = ['Session', 'Block', 'ChA', 'CurrentsA', 'HitsA', 'ThresholdA', 'Closest CurrentA to 50%',
                    'ChB', 'CurrentsB', 'HitsB', 'ThresholdB', 'Closest CurrentB to 50%',
                    'ChC', 'CurrentsC', 'HitsC', 'ThresholdC', 'Closest CurrentC to 50%',
                    'Include', 'Avg Thresholds', 'Avg Closest Currents', 'Delta']

    # Process data and write to CSV
    with open(csv_file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)

        for session, blocks in all_session_thresholds.items():
            for block_num, channels in blocks.items():
                # Convert block number to 1-indexed
                block_index = block_num + 1

                # Sort channels by key (channel number)
                sorted_channels = sorted(channels.items(), key=lambda x: x[0])
                channel_numbers = [channel[0] for channel in sorted_channels]
                thresholds = [channel[1] for channel in sorted_channels]

                block_hit_dict = hits_and_currents_dict[session][block_num]['hits']
                ch_a_hits = block_hit_dict.get(channel_numbers[0], 0)
                ch_b_hits = block_hit_dict.get(channel_numbers[1], 0)
                ch_c_hits = block_hit_dict.get(channel_numbers[2], 0)

                block_curr_dict = hits_and_currents_dict[session][block_num]['currents']
                ch_a_curr = block_curr_dict.get(channel_numbers[0], [])
                ch_b_curr = block_curr_dict.get(channel_numbers[1], [])
                ch_c_curr = block_curr_dict.get(channel_numbers[2], [])

                # Find closest currents to 50% for each channel
                closest_a = find_closest_current(ch_a_hits, ch_a_curr)
                closest_b = find_closest_current(ch_b_hits, ch_b_curr)
                closest_c = find_closest_current(ch_c_hits, ch_c_curr)

                include = inclusion_criteria(ch_a_hits, ch_b_hits, ch_c_hits)

                # Write row to CSV
                writer.writerow([session, block_index] +
                                [channel_numbers[0], ch_a_curr, ch_a_hits, thresholds[0], closest_a,
                                 channel_numbers[1], ch_b_curr, ch_b_hits, thresholds[1], closest_b,
                                 channel_numbers[2], ch_c_curr, ch_c_hits, thresholds[2], closest_c,
                                 include, '', '', ''])  # placeholders for avg values

    return hits_and_currents_dict


def write_avg_thresholds_closest_currents_and_delta(csv_file_name):
    session_data = {}
    with open(csv_file_name, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            session = row['Session']
            include = int(row['Include'])
            if include == 1:
                if session not in session_data:
                    session_data[session] = {'threshold_counts': [0, 0, 0], 'threshold_sums': [0.0, 0.0, 0.0],
                                             'current_counts': [0, 0, 0], 'current_sums': [0.0, 0.0, 0.0]}
                for i, ch in enumerate('ABC'):
                    session_data[session]['threshold_counts'][i] += 1
                    session_data[session]['threshold_sums'][i] += float(
                        row[f'Threshold{ch}'])
                    session_data[session]['current_counts'][i] += 1
                    session_data[session]['current_sums'][i] += float(
                        row[f'Closest Current{ch} to 50%'])

    updated_rows = []
    with open(csv_file_name, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            session = row['Session']
            block = int(row['Block'])
            if session in session_data and block == 1:
                avg_thresholds = [int(round(sums / counts, 0)) if counts > 0 else 0 for counts, sums in zip(
                    session_data[session]['threshold_counts'], session_data[session]['threshold_sums'])]
                avg_currents = [int(round(sums / counts, 0)) if counts > 0 else 0 for counts, sums in zip(
                    session_data[session]['current_counts'], session_data[session]['current_sums'])]
                delta = [round(threshold - current, 2) for threshold,
                         current in zip(avg_thresholds, avg_currents)]
                row['Avg Thresholds'] = '[' + \
                    ', '.join([str(avg) for avg in avg_thresholds]) + ']'
                row['Avg Closest Currents'] = '[' + \
                    ', '.join([str(avg) for avg in avg_currents]) + ']'
                row['Delta'] = '[' + ', '.join([str(d) for d in delta]) + ']'
            updated_rows.append(row)

    with open(csv_file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in updated_rows:
            writer.writerow(row)


def fill_in_csv_with_total_hits(hits_dict, csv_file_name):
    df = pd.read_csv(csv_file_name)
    for index, row in df.iterrows():
        session = row['Session']
        # Adjusting block index to 0-based for dictionary access
        block = row['Block'] - 1
        ch_a = row['ChA']
        ch_b = row['ChB']
        ch_c = row['ChC']

        # Update the DataFrame with total hits
        df.at[index, 'ChA Hits'] = hits_dict[session][block].get(ch_a, 0)
        df.at[index, 'ChB Hits'] = hits_dict[session][block].get(ch_b, 0)
        df.at[index, 'ChC Hits'] = hits_dict[session][block].get(ch_c, 0)

    # Write the updated DataFrame back to the CSV
    df.to_csv(csv_file_name, index=False)


def get_total_hits_per_ch(df):
    assert len(df) == 400
    trials_per_block = 100  # Each block has 100 trials
    total_hits_per_ch = {}
    total_max_cur_hits_per_ch = {}
    for block in range(4):
        start_idx = trials_per_block * block
        end_idx = start_idx + trials_per_block
        sub_df = df.iloc[start_idx:end_idx]
        unique_channels = sorted(
            sub_df[sub_df['channel'] != 0]['channel'].unique())

        channel_hits = {}
        channel_max_current_hits = {}
        for channel in unique_channels:

            channel_data = sub_df[sub_df['channel'] == channel]
            channel_hits[channel] = channel_data['response'].sum()

            channel_currents = np.sort(channel_data['current'].unique())
            max_ch_current = channel_currents[-1]
            max_current_data = sub_df[sub_df['current'] == max_ch_current]
            channel_max_current_hits[channel] = max_current_data['response'].sum(
            )

        total_hits_per_ch[block] = channel_hits
        total_max_cur_hits_per_ch[block] = channel_max_current_hits
    return total_hits_per_ch, total_max_cur_hits_per_ch


def get_hits_per_ch(df):
    assert len(df) == 400
    trials_per_block = 100  # Each block has 100 trials
    hits_per_ch = {}
    for block in range(4):
        start_idx = trials_per_block * block
        end_idx = start_idx + trials_per_block
        sub_df = df.iloc[start_idx:end_idx]
        unique_channels = sorted(
            sub_df[sub_df['channel'] != 0]['channel'].unique())

        channel_hits = {}
        for channel in unique_channels:
            channel_data = sub_df[sub_df['channel'] == channel]
            channel_currents = np.sort(channel_data['current'].unique())
            current_hits = []
            for current in channel_currents:

                current_data = sub_df[sub_df['current'] == current]
                current_hits.append(current_data['response'].sum())
            channel_hits[channel] = current_hits

        hits_per_ch[block] = channel_hits
    return hits_per_ch


def get_hits_per_ch(df):
    assert len(df) == 400
    trials_per_block = 100  # Each block has 100 trials
    hits_per_ch = {}
    for block in range(4):
        start_idx = trials_per_block * block
        end_idx = start_idx + trials_per_block
        sub_df = df.iloc[start_idx:end_idx]
        unique_channels = sorted(
            sub_df[sub_df['channel'] != 0]['channel'].unique())

        channel_hits = {}
        for channel in unique_channels:
            channel_data = sub_df[sub_df['channel'] == channel]
            channel_currents = np.sort(channel_data['current'].unique())
            current_hits = []
            for current in channel_currents:
                current_data = channel_data[channel_data['current'] == current]
                current_hits.append(current_data['response'].sum())
            channel_hits[channel] = current_hits

        hits_per_ch[block] = channel_hits
    return hits_per_ch


def get_currents_per_ch(df):
    assert len(df) == 400
    trials_per_block = 100  # Each block has 100 trials
    currents_per_ch = {}

    for block in range(4):
        start_idx = trials_per_block * block
        end_idx = start_idx + trials_per_block
        sub_df = df.iloc[start_idx:end_idx]

        unique_channels = sorted(
            sub_df[sub_df['channel'] != 0]['channel'].unique())
        channel_currents = {}

        for channel in unique_channels:
            channel_data = sub_df[sub_df['channel'] == channel]
            channel_currents[channel] = channel_data['current'].unique(
            ).tolist()

        currents_per_ch[block] = channel_currents

    return currents_per_ch


def get_hits_dict(sorted_data_folders):
    hits_dict = {}
    for data_folder in sorted_data_folders:
        df = get_dataframe(data_folder)
        hits_per_ch = get_hits_per_ch(df)

        hits_dict[data_folder] = hits_per_ch
    return hits_dict


def get_total_hits_dict(sorted_data_folders):
    total_hits_dict = {}
    max_cur_hits_dict = {}
    for data_folder in sorted_data_folders:
        df = get_dataframe(data_folder)
        total_hits_per_ch, total_max_cur_hits_per_ch = get_total_hits_per_ch(
            df)
        total_hits_dict[data_folder] = total_hits_per_ch
        max_cur_hits_dict[data_folder] = total_max_cur_hits_per_ch
    return total_hits_dict, max_cur_hits_dict


def plot_current_level_hits(hits_dict):
    for date, block_dict in enumerate(hits_dict.items()):
        offsets = np.arange(-0.5, 0.5, 4) + i
        for block, ch_dict in block_dict.items():
            for ch in ch_dict.keys():
                arr = ch_dict[ch]
                plt.scatter(offsets + block, arr)


def plot_current_level_hits(hits_dict, animalID, save_path):
    # Define the offset for each block within a session
    plt.figure(figsize=(12, 5))
    image_save_path = os.path.join(save_path, animalID + '_current_level_hits')

    block_offsets = [-0.15, -0.05, 0.05, 0.15]
    # Define labels for low, medium, and high currents
    current_labels = ['Low Current', 'Medium Current', 'High Current']
    # Get colors from the 'deep' Seaborn palette
    colors = sns.color_palette('deep', n_colors=3)

    # Plot low, medium, high current hits as three separate curves, take mean and std across channels for each block
    for session_idx, (date, block_dict) in enumerate(hits_dict.items()):
        for current_idx in range(3):
            # Initialize lists to store x and y values for the current plot
            x_values = []
            y_values = []
            y_errors = []

            for block_idx, (block, channels) in enumerate(block_dict.items()):
                # Extract the values for the current curve (low, medium, or high)
                current_values = [channel_data[current_idx]
                                  for channel_data in channels.values()]

                # Calculate mean and standard deviation across channels for the current block
                mean = np.mean(current_values)
                std = np.std(current_values)

                # Store the values for plotting
                x_values.append(session_idx + block_offsets[block_idx])
                y_values.append(mean)
                y_errors.append(std)

            # Plot the mean with standard deviation error bars for the current curve
            plt.errorbar(x_values, y_values, yerr=y_errors, fmt='-o', color=colors[current_idx],
                         label=current_labels[current_idx] if session_idx == 0 else None)

    dates = [key.split('\\')[-1] for key in hits_dict.keys()]
    # Add labels and legend
    plt.xticks(range(len(hits_dict)), dates, rotation=45)
    plt.ylabel('Hits')
    plt.title(f'{animalID} mean block hits for current levels across sessions')
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()
    plt.savefig(image_save_path)
    plt.close()


def plot_included_block_thresholds_over_time(csv_file_name):
    fig, ax = plt.subplots(figsize=(12, 5))
    palette = sns.color_palette("deep")

    # Read and filter the data
    df = pd.read_csv(csv_file_name)
    df['DateString'] = df['Session'].apply(lambda x: x.split('\\')[-1])
    df['Date'] = df['DateString'].apply(
        lambda x: datetime.strptime(x, '%d-%b-%Y'))
    df_filtered = df[df['Include'] == 1].copy()

    # Create a mapping from sorted unique dates to integers
    sorted_dates = sorted(df_filtered['Date'].unique())
    date_mapping = {date.strftime(
        '%d-%b-%Y'): i for i, date in enumerate(sorted_dates)}

    # Plot individual scatter points with jitter and the average thresholds for each channel
    threshold_columns = ['ThresholdA', 'ThresholdB', 'ThresholdC']
    closest_current_columns = ['Closest CurrentA to 50%',
                               'Closest CurrentB to 50%', 'Closest CurrentC to 50%']
    jitter_amount = 0.1
    for i, (threshold_column, closest_current_column) in enumerate(zip(threshold_columns, closest_current_columns)):
        color = palette[i]
        jittered_x = [date_mapping[date] +
                      np.random.uniform(-jitter_amount, jitter_amount) for date in df_filtered['DateString']]
        ax.scatter(
            jittered_x, df_filtered[threshold_column], color=color)
        df_filtered['DateIndex'] = df_filtered['DateString'].map(date_mapping)
        average_thresholds = df_filtered.groupby(
            'DateIndex')[threshold_column].mean()
        average_thresholds.plot(kind='line', color=color,
                                ax=ax, marker='o', linestyle='-')
        # Plot the closest current to 50% as dotted lines
        ax.plot(df_filtered['DateIndex'], df_filtered[closest_current_column], color=color,
                linestyle='--', linewidth=2, label=f'{closest_current_column}')

    # Customize the plot
    animalID = csv_file_name.split('_')[0]
    ax.set_title(f'{animalID} Thresholds and Closest Current to 50% Over Time')
    ax.set_xlabel('')
    ax.set_ylabel('Value')
    ax.set_xticks(range(len(date_mapping)))
    ax.set_xticklabels(date_mapping.keys(), rotation=45)
    ax.legend(title='Channel', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# plot_included_block_thresholds_over_time('ICMS98_blocks.csv')
# %%

if __name__ == "__main__":
    test_inclusion_criteria()
    with open('all_animals_dict.pkl', 'rb') as file:
        all_animals_dict = pickle.load(file)

    for animalID, sub_dict in all_animals_dict.items():

        sorted_data_folders = sub_dict['sorted_data_folders']
        hits_and_currents_dict = sub_dict['hc_dict']

        csv_file_name = animalID + '_blocks.csv'
        print(f'Generating csv file: {csv_file_name}')
        write_csv_file(sorted_data_folders,
                       hits_and_currents_dict, csv_file_name)
        write_avg_thresholds_closest_currents_and_delta(csv_file_name)


if __name__ == "__main__":
    # data_folders = write_thresholds()
    data_folders = file_dialog()
    sorted_data_folders = sorted(data_folders, key=parse_date_from_path)

    # try:
    #     all_session_thresholds = read_thresholds(sorted_data_folders)
    # except:
    #     write_thresholds(sorted_data_folders)

    # df = process_thresholds(all_session_thresholds)
    # # Convert 'Date' to datetime objects
    # df['Date'] = pd.to_datetime(df['Date'])

    # # Sort the DataFrame by 'Date'
    # df = df.sort_values('Date')

    # # Format 'Date' to the desired string format (e.g., 22-Nov-2023)
    # df['Date'] = df['Date'].dt.strftime('%d-%b-%Y')

    # # Convert 'Date' back to a categorical type with explicit ordering
    # date_order = pd.Categorical(
    #     df['Date'], categories=df['Date'].unique(), ordered=True)
    # df['Date'] = date_order

    # # Convert 'Date' to a categorical type
    # # df['Date'] = pd.Categorical(df['Date'])

    # plot_thresholds_over_time(df)
    match = re.search(r'ICMS(\d+)', sorted_data_folders[0])
    if match:
        animalID = 'ICMS' + match.group(1)
    else:
        print("ICMS number not found")

    all_session_thresholds = read_thresholds(sorted_data_folders)
    hits_dict = get_hits_dict(sorted_data_folders)
    hits_and_currents_dict = get_hits_and_currents_dict(sorted_data_folders)

    # hits_dict = exclude_blocks(sorted_data_folders, csv_file_name)

    csv_file_name = animalID + '_blocks.csv'
    write_csv_file(sorted_data_folders, hits_and_currents_dict, csv_file_name)
# #     # # save_path = 'C:/data/' + animalID + '/Behavior/analysis'
# #     # # save_path = 'C:/data/' + animalID + '/analysis'
# #     # if animalID == 'ICMS101':
# #     #     save_path = 'D:/data/' + animalID + '/analysis'
# #     # plot_current_level_hits(hits_dict, animalID, save_path)

# #     # plot thresholds with exclusion
    # plot_included_block_thresholds_over_time(csv_file_name, animalID)
    all_animals_dict[animalID] = {
        'sorted_data_folders': sorted_data_folders, 'hc_dict': hits_and_currents_dict}
    with open('all_animals_dict.pkl', 'wb') as file:
        pickle.dump(all_animals_dict, file)
