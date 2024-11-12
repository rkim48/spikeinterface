from pathlib import Path
import os
import h5py

from spikeinterface import full as si
import util.file_util as file_util

data_folders = file_util.file_dialog("C://data")
sorted_data_folders = file_util.sort_data_folders(data_folders)
animalID = file_util.get_animal_id(sorted_data_folders[0])
hdf5_file_path = Path(sorted_data_folders[0]).parent / f"{animalID}.h5"  # Fix file path definition

for session_num, data_folder in enumerate(sorted_data_folders):
    animalID = file_util.get_animal_id(data_folder)
    date_string = file_util.get_date_str(data_folder)
    save_folder = Path(data_folder) / "batch_sort"

    stage1_analyzer = si.load_sorting_analyzer(folder=save_folder / "stage1/stage1_analyzer_raw.zarr")
    stage1_analyzer.compute("spike_amplitudes")
    stage1_analyzer.compute("template_similarity")
    stage1_analyzer.compute("unit_locations")

    si.plot_sorting_summary(stage1_analyzer, curation=True, backend="sortingview")
    uri = input("\nPlease enter uri: ")
    curation_sorting = si.apply_sortingview_curation(stage1_analyzer.sorting, uri_or_json=uri)

    unit_ids = curation_sorting.unit_ids
    templates_ext = stage1_analyzer.get_extension("templates")
    ccgs_ext = stage1_analyzer.get_extension("correlograms")
    ccgs, time_bins = ccgs_ext.get_data()
    amp_ext = stage1_analyzer.get_extension("spike_amplitudes")
    amp_data = amp_ext.get_data()
    spike_vector = curation_sorting.to_spike_vector()
    channel_locations = stage1_analyzer.get_channel_locations()
    acg_time_bins = time_bins

    # Ensure we use the same file for all sessions
    with h5py.File(hdf5_file_path, "a") as hdf5_file:
        # Check if session group already exists
        session_group_name = f"session_{session_num + 1}"
        if session_group_name in hdf5_file:
            print(f"Session {session_num + 1} already exists in file.")
            continue  # Skip to next session

        # Create a group for the current session
        session_group = hdf5_file.create_group(session_group_name)

        # Store session metadata
        session_group.attrs["session_date"] = date_string
        session_group.attrs["session_num"] = session_num + 1

        # Store session-level data (common to all units)
        session_group.create_dataset("channel_locations", data=channel_locations)
        session_group.create_dataset("acg_time_bins", data=acg_time_bins)

        # Create a group for unit-specific data within this session
        units_group = session_group.create_group("units")

        # Loop through each unit and store unit-specific data
        for i, unit_id in enumerate(unit_ids):
            unit_group = units_group.create_group(f"unit_{unit_id}")

            is_accepted = curation_sorting.get_unit_property(unit_id, "accept")
            unit_group.attrs["is_accepted"] = is_accepted  # Store as an attribute

            template = templates_ext.get_unit_template(unit_id=unit_id)
            unit_group.create_dataset("template", data=template)

            unit_index = curation_sorting.id_to_index(unit_id)

            # Autocorrelogram (ACG)
            acg = ccgs[unit_index, unit_index, :]
            unit_group.create_dataset("acg", data=acg)

            # Spike timestamps and amplitudes
            ts_indices = spike_vector["unit_index"] == unit_index
            timestamps = spike_vector["sample_index"][ts_indices]
            amplitudes = amp_data[ts_indices]

            unit_group.create_dataset("timestamps", data=timestamps)
            unit_group.create_dataset("amplitudes", data=amplitudes)

    print(f"Data for session {session_num + 1} stored for animal {animalID}")


# %% Access hdf5 file
with h5py.File(hdf5_file_path, "r") as hdf5_file:
    # Print the file structure
    print("HDF5 file structure:")

    def print_structure(name, obj):
        print(name)

    hdf5_file.visititems(print_structure)
# %%
with h5py.File(hdf5_file_path, "r") as hdf5_file:
    # Access session-level data
    session_data = hdf5_file["session_2"]

    # List all units in this session
    units_group = session_data["units"]
    unit_names = list(units_group.keys())

    print("Available units:", unit_names)

    # If you find the correct unit name, access its data:
    for unit_name in unit_names:
        unit_group = units_group[unit_name]
        print(f"Accessing data for {unit_name}")

        # Access attributes like whether the unit is accepted
        is_accepted = unit_group.attrs["is_accepted"]

        # Access datasets like template, timestamps, and amplitudes
        template = unit_group["template"][:]
        timestamps = unit_group["timestamps"][:]
        amplitudes = unit_group["amplitudes"][:]

        # Print or use the loaded data
        print(f"{unit_name} accepted: {is_accepted}")
        print(f"Template shape: {template.shape}")
        print(f"Timestamps: {timestamps[:5]}")  # Print first 5 timestamps
        print(f"Amplitudes: {amplitudes[:5]}")  # Print first 5 amplitudes


# %%
class UnitFeatures:
    # lightweight class that stores important features of
    # units labeled as either accepted or rejected by the user

    def __init__(self):
        pass

    def plot_template(self):
        for idx, wvf in enumerate(template.T):
            plt.plot(wvf + channel_locations[idx, 1])

    def plot_acg(self):
        time_bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
        plt.bar(time_bin_centers, acg, width=np.diff(time_bins), align="center")
