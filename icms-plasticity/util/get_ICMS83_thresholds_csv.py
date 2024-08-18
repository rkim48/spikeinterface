from datetime import datetime
import pandas as pd
import numpy as np
import os
import scipy.io
import re
import psignifit as ps
# %%
# Read ICMS83 behavioral data


def extract_time_info(file_name):
    match = re.match(r'.*D(\d+)_H(\d+)M(\d+)_.*\.mat', file_name)
    if match:
        channel = int(match.group(1))
        timestamp = match.group(2) + match.group(3)  # Combine hour and minute
        return channel, timestamp
    return None, None


def extract_behavioral_data(trial_data):
    # Define the desired frequency, pulse width, and number of trials per current
    desired_freq = 100
    desired_pulse_width = 167

    # Check if all rows have the desired frequency and pulse width
    if np.all(trial_data[:, 4] == desired_freq) and np.all(trial_data[:, 5] == desired_pulse_width):
        # Get unique currents and their counts
        currents, trial_counts = np.unique(
            trial_data[:, 1], return_counts=True)
        if len(currents) < 2 or trial_counts[0] < 30:
            return None

        assert np.all(trial_counts ==
                      trial_counts[0]), "Not all elements are the same"
        # Construct the Nx3 array for psignifit
        psignifit_array = np.empty((0, 3), dtype=int)
        for current in currents:
            hits = np.sum(trial_data[trial_data[:, 1] == current, 2])
            psignifit_array = np.vstack(
                (psignifit_array, [int(current), int(hits), int(trial_counts[0])]))
        return psignifit_array
    else:
        print("Skipping this mat file due to incorrect frequency or pulse width.")
    return None


# %%
parent_folder = 'S:/ICMS83/Behavior'
subfolders = [f for f in os.listdir(
    parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
sorted_subfolders = sorted(
    subfolders, key=lambda x: datetime.strptime(x, '%d-%b-%Y'))
for folder in sorted_subfolders:
    print(folder)

df = pd.DataFrame(columns=['Session', 'Timestamps', 'Channel',
                  'Trials per current', 'Currents', 'Hits', 'Threshold'])

for folder in sorted_subfolders:
    subfolder_path = os.path.join(parent_folder, folder)
    mat_files = [f for f in os.listdir(subfolder_path) if f.endswith('.mat')]
    for mat_file in mat_files:
        mat_file_path = os.path.join(subfolder_path, mat_file)
        mat_data = scipy.io.loadmat(mat_file_path)
        channel, timestamp = extract_time_info(mat_file_path)
        trial_data = np.array(mat_data['s']['trialData'][0][0])
        psignifit_array = extract_behavioral_data(trial_data)
        if psignifit_array is not None:
            trials_per_current = trial_data.shape[0] // len(
                np.unique(trial_data[:, 1]))
            existing_rows = df.loc[(df['Session'] == folder) & (
                df['Channel'] == channel) & (df['Trials per current'] == trials_per_current)]
            if not existing_rows.empty:
                row_index = existing_rows.index[0]
                df.at[row_index, 'Timestamps'] = df.at[row_index,
                                                       'Timestamps'] + [f"H{timestamp[:2]}M{timestamp[2:]}"]
                combined_currents_hits = list(
                    zip(df.at[row_index, 'Currents'], df.at[row_index, 'Hits']))
                for current, hits in zip(psignifit_array[:, 0], psignifit_array[:, 1]):
                    if current not in df.at[row_index, 'Currents']:
                        combined_currents_hits.append((current, hits))
                combined_currents_hits.sort(
                    key=lambda x: x[0])  # Sort by currents
                df.at[row_index, 'Currents'], df.at[row_index, 'Hits'] = zip(
                    *combined_currents_hits)  # Unzip sorted pairs
            else:
                new_row = pd.DataFrame({
                    'Session': [folder],
                    'Timestamps': [[f"H{timestamp[:2]}M{timestamp[2:]}"]],
                    'Channel': [channel],
                    'Trials per current': [trials_per_current],
                    'Currents': [psignifit_array[:, 0].tolist()],
                    'Hits': [psignifit_array[:, 1].tolist()],
                    'Threshold': [np.nan]  # Placeholder for threshold
                })
                df = pd.concat([df, new_row], ignore_index=True)


# Calculate thresholds for each row
options = {'sigmoidName': 'norm', 'expType': 'YesNo', 'plotThresh': False}
for i, row in df.iterrows():
    psignifit_data = np.column_stack((row['Currents'], row['Hits'], [
                                     row['Trials per current']] * len(row['Currents'])))
    result = ps.psignifit(psignifit_data, options)
    threshold = ps.getThreshold(result, 0.5)[0]
    # Assign the calculated threshold back to the DataFrame
    df.at[i, 'Threshold'] = np.round(threshold, 2)

    # Calculate percentages and find the current closest to 50% hit rate
    percentages = np.array(row['Hits']) / row['Trials per current'] * 100
    # Find the index of the percentage closest to 50%
    closest_index = np.argmin(np.abs(percentages - 50))
    closest_current = row['Currents'][closest_index]
    df.at[i, 'Closest Current to 50%'] = closest_current

df.to_csv('test.csv', index=False)


# %%
parent_folder = 'S:/ICMS83/Behavior'
subfolders = [f for f in os.listdir(
    parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
sorted_subfolders = sorted(
    subfolders, key=lambda x: datetime.strptime(x, '%d-%b-%Y'))
for folder in sorted_subfolders:
    print(folder)

options = {'sigmoidName': 'norm',
           'expType': 'YesNo', 'plotThresh': 'false'}
df = pd.DataFrame(columns=['Session', 'Timestamps', 'Channel', 'Trials per current',
                           'Currents', 'Hits', 'Threshold'])

for folder in sorted_subfolders:
    subfolder_path = os.path.join(parent_folder, folder)
    mat_files = [f for f in os.listdir(subfolder_path) if f.endswith('.mat')]
    channel_blocks = {}
    for mat_file in mat_files:
        mat_file_path = os.path.join(subfolder_path, mat_file)
        mat_data = scipy.io.loadmat(mat_file_path)
        print(f"Processing {mat_file_path}...\n")

        channel, hour, minute = extract_time_info(mat_file_path)
        if channel not in channel_blocks:
            channel_blocks[channel] = 1
        else:
            channel_blocks[channel] += 1
        block = channel_blocks[channel]

        trial_data = np.array(mat_data['s']['trialData'][0][0])
        psignifit_array = extract_behavioral_data(trial_data)
        if psignifit_array is None:
            continue
        currents = psignifit_array[:, 0]
        hits = psignifit_array[:, 1]
        total_trials = psignifit_array[:, 2][0]
        # result = ps.psignifit(psignifit_array, options)
        # threshold = ps.getThreshold(result, 0.5)[0]
        threshold = 5

        new_row = pd.DataFrame({
            'Session': [folder],  # Use folder name instead of full path
            'Timestamps': [timestamps],
            'Channel': [channel],
            'Trials per current': [total_trials],
            'Currents': [currents.tolist()],  # Convert currents array to list
            'Hits': [hits.tolist()],  # Convert hits array to list
            'Threshold': [np.round(threshold, 2)],
            'Include': [include]
        })
        df = pd.concat([df, new_row], ignore_index=True)

df.to_csv('test.csv', index=False)

# %% Plot thresholds


def plot_included_block_thresholds_over_time(csv_file_name='ICMS83_thresholds.csv'):
    fig, ax = plt.subplots(figsize=(12, 5))
    palette = sns.color_palette("deep")

    # Read and filter the data
    df = pd.read_csv(csv_file_name)
    df['DateString'] = df['Session'].apply(lambda x: x.split('\\')[-1])
    df['Date'] = df['DateString'].apply(
        lambda x: datetime.strptime(x, '%d-%b-%y'))
    # Convert 'Channel' column to categorical dtype
    df['Channel'] = df['Channel'].astype('category')

    # Create a mapping from sorted unique dates to integers
    sorted_dates = sorted(df['Date'].unique())
    date_mapping = {date.strftime(
        '%d-%b-%y'): i for i, date in enumerate(sorted_dates)}

    # Plot individual scatter points with jitter
    jitter_amount = 0.1
    for channel, group in df.groupby('Channel'):
        # Use categorical encoding to get consistent colors
        color = palette[group['Channel'].cat.codes.unique()[0]]
        jittered_x = [date_mapping[date.strftime(
            '%d-%b-%y')] + np.random.uniform(-jitter_amount, jitter_amount) for date in group['Date']]
        ax.scatter(jittered_x, group['Threshold'], color=color, label=channel)
        ax.scatter(jittered_x, group['Closest Current to 50%'],
                   color=color, marker='x', label=f"{channel} Closest to 50%")

    # Calculate and plot the average thresholds for each channel
    for channel, group in df.groupby('Channel'):
        color = palette[group['Channel'].cat.codes.unique()[0]]
        group['DateIndex'] = group['Date'].apply(
            lambda x: date_mapping[x.strftime('%d-%b-%y')])
        average_thresholds = group.groupby('DateIndex')['Threshold'].mean()
        ax.plot(average_thresholds.index, average_thresholds.values,
                color=color, marker='o', linestyle='-')
        average_closest_current = group.groupby(
            'DateIndex')['Closest Current to 50%'].mean()
        ax.plot(average_closest_current.index, average_closest_current.values,
                color=color, marker='x', linestyle='--')

    # Customize the plot
    animalID = csv_file_name.split('_')[0]
    ax.set_title(f'{animalID} thresholds over time')
    ax.set_xlabel('')
    ax.set_ylabel('Threshold / Closest Current to 50%')
    ax.set_xticks(range(len(date_mapping)))
    ax.set_xticklabels(date_mapping.keys(), rotation=45)
    ax.legend(title='Channel')
    plt.tight_layout()
    plt.show()


plot_included_block_thresholds_over_time()
