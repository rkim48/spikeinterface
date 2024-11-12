import util.load_data as load_data
from spikeinterface import full as si
from pathlib import Path
import batch_process.util.file_util as file_util
import matplotlib.pyplot as plt
import numpy as np

# %%


def analyzer_ccg(analyzer, unit_indices):
    ccg_ext = analyzer.get_extension("correlograms")
    data = ccg_ext.get_data()
    t = data[1]
    ccgs = data[0]

    bin_centers = (t[:-1] + t[1:]) / 2
    bin_width = t[1] - t[0]  # Assuming uniform bin widths

    # Choose a subset of units, e.g., units 5 to 9 (adjust this as needed)
    unit_ids = analyzer.unit_ids
    n_units = len(unit_indices)

    # Initialize matrix to store peak values if needed
    ccg_peak_matrix = np.zeros((n_units, n_units))

    # Create subplots grid for visualizing CCGs
    fig, axs = plt.subplots(n_units, n_units, figsize=(5, 5))

    # Loop over the chosen units to create a grid of CCG plots
    for row, i in enumerate(unit_indices):
        for col, j in enumerate(unit_indices):
            axs[row, col].bar(bin_centers, ccgs[i, j, :],
                              width=bin_width, align='center')
            axs[row, col].set_title(f'Unit {unit_ids[i]} & Unit {
                                    unit_ids[j]}', fontsize=8)
            axs[row, col].tick_params(axis='both', which='major', labelsize=6)
            axs[row, col].set_xlim([t[0], t[-1]])  # Optional: set x-limits

    plt.tight_layout()
    plt.show()


# %% Get non-stim period spike trains

# def get_nonstim_ccg(data_folder):
data_folder = "C:\\data\\ICMS92\Behavior\\01-Sep-2023"
df = load_data.get_dataframe(data_folder, make_folder=False)
save_folder = Path(data_folder) / "batch_sort"
analyzer = si.load_sorting_analyzer(
    Path(save_folder) / "merge/hmerge_analyzer_curated.zarr")

df = df[df['stim_timestamps'].apply(lambda x: len(x) > 0)]

unit_ids = analyzer.unit_ids
n_units = len(unit_ids)
# List of lists to store spike trains for each unit
nonstim_spike_trains = [[] for _ in range(n_units)]
start_seg_ts = 0

for row_index, row in df.iterrows():
    # Define end segment based on stimulus timestamps
    end_seg_ts = row['stim_timestamps'][0] - 30 * 10

    # Loop over units and truncate spike trains between start_seg_ts and end_seg_ts
    for unit_index, unit_id in enumerate(analyzer.unit_ids):
        spike_train = analyzer.sorting.get_unit_spike_train(unit_id)

        # Truncate spike train for non-stimulus periods
        trunc_spike_train = spike_train[(
            spike_train > start_seg_ts) & (spike_train < end_seg_ts)]

        # Append the truncated spike train for this unit
        nonstim_spike_trains[unit_index].extend(trunc_spike_train)

    # Update the start timestamp for the next non-stimulus segment
    start_seg_ts = row['stim_timestamps'][-1] + 30 * 10

units_dict_list = {str(unit_id): np.array(
    nonstim_spike_trains[i]) for i, unit_id in enumerate(unit_ids)}

# plt.eventplot(nonstim_spike_trains, linewidths=0.5, linelengths=0.5)
# %%
nonstim_sorting = si.NumpySorting.from_unit_dict(
    units_dict_list=units_dict_list,
    sampling_frequency=30000
)

nonstim_analyzer_path = Path(save_folder) / "merge/nonstim_analyzer.zarr"
nonstim_analyzer = si.create_sorting_analyzer(
    sorting=nonstim_sorting,
    recording=analyzer.recording,
    format="zarr",
    overwrite=True,
    return_scaled=True,  # this is the default to attempt to return scaled
    folder=nonstim_analyzer_path
)

# Compute extensions
nonstim_analyzer.compute("random_spikes")
nonstim_analyzer.compute("waveforms")
nonstim_analyzer.compute("templates")
nonstim_analyzer.compute("template_similarity")
nonstim_analyzer.compute("correlograms", window_ms=50, bin_ms=0.5)
nonstim_analyzer.compute("spike_amplitudes")
nonstim_analyzer.compute(
    "unit_locations", method="monopolar_triangulation", radius_um=200, feature="ptp")

# %%

analyzer_ccg(nonstim_analyzer)


# %% Look at ccgs
animalID = "ICMS92"
tracked_unit_id = 'A'
date_unit_id_dict_A = {
    "30-Aug-2023": [11, 12],
    "01-Sep-2023": [5, 8],
    "06-Sep-2023": [9],
    "08-Sep-2023": [8],
    "12-Sep-2023": [7, 12],
    "14-Sep-2023": [8, 9],
    "19-Sep-2023": [10, 11],
    "21-Sep-2023": [10, 11],
    "25-Sep-2023": [8],
    "27-Sep-2023": [11]
}

tracked_unit_id = 'B'
date_unit_id_dict_B = {
    "30-Aug-2023": [13, 14],
    "01-Sep-2023": [6],
    "06-Sep-2023": [6],
    "08-Sep-2023": [999],
    "12-Sep-2023": [9],
    "14-Sep-2023": [10, 11],
    "19-Sep-2023": [12, 13],
    "21-Sep-2023": [12, 14],
    "25-Sep-2023": [10],
    "27-Sep-2023": [12, 13]
}


# for date_index, (data_folder, unit_ids) in enumerate(date_unit_id_dict.items()):
#     base_path = Path("C:/data/") / animalID / "Behavior"
#     save_folder = base_path / Path(data_folder) / "batch_sort"

#     analyzer = si.load_sorting_analyzer(
#         Path(save_folder) / "merge/hmerge_analyzer_curated.zarr")

#     unit_indices = [np.where(analyzer.unit_ids == unit_id)[
#         0][0] for unit_id in unit_ids]

#     analyzer_ccg(analyzer, unit_indices)

# Merging unit IDs from both dicts per session
for date_index, (data_folder, unit_ids_A) in enumerate(date_unit_id_dict_A.items()):
    # Get unit IDs for the same date from both dicts
    unit_ids_B = date_unit_id_dict_B.get(data_folder, [])

    # Combine unit ids from A and B
    combined_unit_ids = unit_ids_A + unit_ids_B

    # Base path and save folder setup
    base_path = Path("C:/data/") / animalID / "Behavior"
    save_folder = base_path / Path(data_folder) / "batch_sort"

    # Load the analyzer
    analyzer = si.load_sorting_analyzer(
        Path(save_folder) / "merge/hmerge_analyzer_curated.zarr"
    )

    # Find valid unit indices (skip unit_ids that don't exist)
    unit_indices = []
    for unit_id in combined_unit_ids:
        if unit_id in analyzer.unit_ids:
            idx = np.where(analyzer.unit_ids == unit_id)[0][0]
            unit_indices.append(idx)

    # Skip if no valid unit indices are found
    if len(unit_indices) > 1:
        # Perform analysis with valid unit indices
        analyzer_ccg(analyzer, unit_indices)
    else:
        print(f"No valid units found for date: {data_folder}")
