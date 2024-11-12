import quantities as pq
import neo
import elephant.spike_train_synchrony as sts
import pickle
import util.load_data as load_data
from spikeinterface import full as si
from pathlib import Path
import batch_process.util.file_util as file_util
import matplotlib.pyplot as plt
import numpy as np
# %%
with open('all_animal_folder_list.pkl', 'rb') as f:
    all_animal_folder_list_0 = pickle.load(f)

for animal_list in all_animal_folder_list_0:
    animal_data = {}
    animal_id = file_util.get_animal_id(animal_list[0])

    for session_index, session in enumerate(animal_list):

        df = load_data.get_dataframe(session, make_folder=False)
        session_date = Path(session).stem
        save_folder = Path(session) / "batch_sort"
        analyzer = si.load_sorting_analyzer(
            Path(save_folder) / "merge/hmerge_analyzer_curated.zarr")

        unit_spike_trains_dict = {unit_id: analyzer.sorting.get_unit_spike_train(
            unit_id) for unit_id in analyzer.unit_ids}

        stim_trains = []
        for row_index, row in df.iterrows():
            stim_trains.append(row['stim_timestamps'])

        animal_data[session_date] = {}
        animal_data[session_date]['session_index'] = session_index
        animal_data[session_date]['unit_spike_train_dict'] = unit_spike_trains_dict
        animal_data[session_date]['stim_trains'] = stim_trains
        animal_data[session_date]['sampling_rate'] = 30000

    pickle_file_name = f"{animal_id}_spike_data.pkl"
    with open(pickle_file_name, 'wb') as handle:
        pickle.dump(animal_data, handle)

# %%


def get_non_stim_segments(stim_trains, total_duration):

    non_stim_segments = []

    # Filter out empty stim trains
    stim_trains = [train for train in stim_trains if len(train) > 0]

    if len(stim_trains) == 0:
        # No stim trains, return the entire duration as a non-stim segment
        return np.array([(0, total_duration)])

    # Add the initial non-stim segment (from 0 to the first stim train start)
    # First timestamp of the first stim train
    first_stim_start = stim_trains[0][0]
    if first_stim_start > 0:
        non_stim_segments.append((0, first_stim_start))

    # Add non-stim periods between stim trains
    for i in range(len(stim_trains) - 1):
        # Last timestamp of the current stim train
        last_of_current_train = stim_trains[i][-1]
        # First timestamp of the next stim train
        first_of_next_train = stim_trains[i + 1][0]
        non_stim_segments.append((last_of_current_train, first_of_next_train))

    # Add the final non-stim segment, from the end of the last stim train to the total duration
    # Last timestamp of the last stim train
    last_stim_end = stim_trains[-1][-1]
    if last_stim_end < total_duration:
        non_stim_segments.append((last_stim_end, total_duration))

    return np.array(non_stim_segments)


def segment_spike_train(spike_train, segments):

    segmented_spikes = []

    # Use np.searchsorted to efficiently find the spikes within each segment
    for start, end in segments:
        # Find the indices in spike_train where spikes are within the segment (start, end)
        start_idx = np.searchsorted(spike_train, start, side='right')
        end_idx = np.searchsorted(spike_train, end, side='left')
        segmented_spikes.append(spike_train[start_idx:end_idx])

    return segmented_spikes


# %%

pickle_file_name = "Umar/ICMS92_spike_data.pkl"
animal_ids = ["ICMS92", "ICMS93", "ICMS98", "ICMS100", "ICMS101"]


# Define the function to process each animal

def process_animal_data(pickle_file_name, animal_id):
    """
    Process the spike data for a given animal and compute the synchrony for each session.

    Parameters:
    -----------
    pickle_file_name : str
        The name of the pickle file containing the animal's spike data.
    animal_id : str
        The ID of the animal being processed.

    Returns:
    --------
    synchrony_dict : dict
        A dictionary where keys are session names and values are synchrony values.
    """
    synchrony_dict = {}

    with open(pickle_file_name, 'rb') as f:
        data = pickle.load(f)

        for session_index, session in enumerate(data):
            print(f"Processing {session} for {animal_id}")

            unit_spike_trains_dict = data[session]['unit_spike_train_dict']
            fs = data[session]['sampling_rate']
            stim_trains = data[session]['stim_trains']
            stim_trains = [train for train in stim_trains if len(train) > 0]

            if len(stim_trains) == 0:
                print(f"No valid stimulation trains for session {
                      session}. Skipping.")
                synchrony_dict[session] = None
                continue

            keys = unit_spike_trains_dict.keys()
            total_duration_samples = max(
                max(unit_spike_trains_dict[key]) for key in keys)

            # Get non-stimulation segments (in samples)
            non_stim_segments = get_non_stim_segments(
                stim_trains, total_duration_samples)

            # Segment the spike trains based on non-stim periods
            segmented_spike_trains = {}
            for unit_id, spike_train in unit_spike_trains_dict.items():
                segmented_spikes = segment_spike_train(
                    spike_train, non_stim_segments)
                segmented_spike_trains[unit_id] = segmented_spikes

            # Prepare Neo SpikeTrain objects (combine non-stim segments for each unit)
            neo_spike_trains = []
            for unit_id, segments in segmented_spike_trains.items():
                valid_segments = [
                    segment for segment in segments if len(segment) > 0]
                if len(valid_segments) > 0:
                    all_spikes = np.concatenate(valid_segments) / fs * pq.s
                    if len(all_spikes) > 0:
                        t_stop_seconds = (total_duration_samples / fs) * pq.s
                        neo_spike_trains.append(neo.SpikeTrain(
                            all_spikes, t_stop=t_stop_seconds))

            # Compute the synchrony using spike contrast, if there are enough spike trains
            if len(neo_spike_trains) > 1:
                try:
                    synchrony_value = sts.spike_contrast(
                        neo_spike_trains, min_bin=0.01 * pq.s, bin_shrink_factor=0.9)
                    print(f"Synchrony (Spike Contrast) for session {
                          session}: {synchrony_value}")
                    synchrony_dict[session] = synchrony_value
                except Exception as e:
                    print(f"Error calculating synchrony for session {
                          session}: {e}")
                    # Handle exception but avoid returning NaN
                    synchrony_dict[session] = None
            else:
                print(f"Not enough valid spike trains for session {
                      session} to compute synchrony.")
                synchrony_dict[session] = None

    return synchrony_dict


animal_ids = ["ICMS92", "ICMS93", "ICMS98", "ICMS100", "ICMS101"]
animal_data = {}

for animal_id in animal_ids:
    # Construct the pickle file name based on the animal ID
    pickle_file_name = f"Umar/{animal_id}_spike_data.pkl"

    # Process the data for this animal and get the synchrony values
    synchrony_dict = process_animal_data(pickle_file_name, animal_id)

    # Store the synchrony data for this animal
    animal_data[animal_id] = synchrony_dict

# %%
animal_id = 'ICMS101'
pickle_file_name = f"Umar/{animal_id}_spike_data.pkl"
synchrony_dict = process_animal_data(pickle_file_name, animal_id)

# %%


def plot_individual_animal_synchrony(synchrony_dict, animal_id):

    sessions = [session for session in synchrony_dict.keys(
    ) if not np.isnan(synchrony_dict[session])]
    synchrony_values = [synchrony_dict[session] for session in sessions]

    # Plot synchrony values for the current animal
    plt.figure(figsize=(8, 5))
    plt.plot(sessions, synchrony_values, marker='o', linestyle='-',
             color='k', label=f'Synchrony for {animal_id}')

    # Add labels, title, and legend
    plt.xlabel('Session')
    plt.ylabel('Synchrony Value')
    plt.title(f'Synchrony Across Sessions for Animal {animal_id}')
    plt.xticks(rotation=45)  # Rotate session labels if necessary
    plt.grid(True)

    # Show the plot for this animal
    plt.tight_layout()  # Adjust layout to prevent label overlap
    plt.show()


# Example usage with the `animal_data` dictionary
for animal_id in animal_ids:
    synchrony_dict = animal_data[animal_id]
    plot_individual_animal_synchrony(synchrony_dict, animal_id)

# %%


# Load the data from pickle
with open(pickle_file_name, 'rb') as f:
    data = pickle.load(f)
    # Adjust according to session dates in the data
    session_date = list(data.keys())[0]
    unit_spike_trains_dict = data[session_date]['unit_spike_train_dict']
    stim_trains = data[session_date]['stim_trains']
    # Example total duration in seconds (adjust according to your data)
    total_duration = 3600

# Get non-stimulation segments
non_stim_segments = get_non_stim_segments(stim_trains, total_duration)

# Segment the spike trains based on non-stim periods
segmented_spike_trains = {}
for unit_id, spike_train in unit_spike_trains_dict.items():
    segmented_spikes = segment_spike_train(spike_train, non_stim_segments)
    segmented_spike_trains[unit_id] = segmented_spikes

# Prepare the spike trains for synchrony analysis
# Use only the spikes from the non-stim segments
neo_spike_trains = []
for unit_id, segments in segmented_spike_trains.items():
    for segment in segments:
        neo_spike_trains.append(neo.SpikeTrain(segment, t_stop=total_duration))

# Compute the synchrony metric (e.g., cross-correlation or STTC)
sync_profile = sc.spike_time_tiling_coefficient(neo_spike_trains, window=0.01)

# Output the synchrony result
print("Synchrony Result:", sync_profile)
