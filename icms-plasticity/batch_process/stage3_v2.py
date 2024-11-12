from scipy.signal import butter, filtfilt
from util.load_data import load_data
import spikeinterface_gui
from util.file_util import *
import umap
import numpy as np
from pathlib import Path
import pickle

from spikeinterface import full as si
from spikeinterface.curation import CurationSorting
from isosplit6 import isosplit6

from batch_process.util.curate_util import *
from batch_process.util.subcluster_util import *
from batch_process.util.misc import align_waveforms
import batch_process.util.template_util as template_util

import warnings

# ignore OMP_NUM_THREADS memory leaks warning
# warnings.filterwarnings("ignore")


def cluster_waveforms(X):
    # X represents truncated primary waveforms for a single unit as detected by Mountainsort.
    # Shape of X: (number of waveforms) x (number of samples per waveform) x 1.
    #
    # These waveforms are assumed to be "undersplit," meaning that the unit detected by Mountainsort
    # may actually consist of multiple distinct clusters or neuron groups that have been grouped together.
    #
    # The purpose of this function is to identify and separate these subclusters by applying UMAP for dimensionality
    # reduction followed by Isosplit6 clustering. The function returns labels for each waveform, indicating its
    # membership in one of the detected subclusters.

    n_neighbors = 50
    min_dist = 0.1
    for iteration in range(1, 6):
        reducer = umap.UMAP(n_neighbors=n_neighbors,
                            min_dist=min_dist, n_jobs=-1)
        wvfs_umap = reducer.fit_transform(X)
        umap_x = wvfs_umap[:, 0]
        umap_y = wvfs_umap[:, 1]
        subcluster_labels = isosplit6(wvfs_umap)
        if len(np.unique(subcluster_labels)) <= 4:
            break
        n_neighbors += 10
        min_dist += 0.1
    else:
        print("Failed to produce <= 4 clusters after 5 iterations.")
    return subcluster_labels, umap_x, umap_y


def process_single_subcluster(
    analyzer,
    unit_id,
    subcluster_id,
    subcluster_labels,
    good_indices,
    waveform_labels,
    final_subcluster_labels,
):
    relative_indices = np.where(subcluster_labels == subcluster_id)[0]
    # Map these indices to their original indices using good_indices
    subcluster_indices = good_indices[relative_indices]

    kept_indices, discarded_indices, subcluster_label = accept_subcluster(
        analyzer, unit_id, subcluster_indices)

    if subcluster_label == "accept":
        if len(kept_indices) > 0:
            waveform_labels[subcluster_indices[kept_indices]] = subcluster_id
        if len(discarded_indices) > 0:
            waveform_labels[subcluster_indices[discarded_indices]] = 0
    else:
        waveform_labels[subcluster_indices] = subcluster_id

    final_subcluster_labels.append((subcluster_id, subcluster_label))


def main(data_folder):
    job_kwargs = dict(n_jobs=5, chunk_duration="1s", progress_bar=True)
    # si.set_global_job_kwargs(**job_kwargs)

    if data_folder:
        data_folders = [data_folder]
    else:
        starting_dir = "C:\\data"
        data_folders = file_dialog(starting_dir=starting_dir)

    for i, data_folder in enumerate(data_folders):
        print("\n###########################################")
        print(f"{data_folder}: {i+1}/{len(data_folders)}")
        print("###########################################")

        cluster_assignments = {}
        save_folder = Path(data_folder) / "batch_sort"
        stage3_path = Path(save_folder) / "stage3"
        create_folder(stage3_path)
        analyzer = si.load_sorting_analyzer(
            folder=save_folder / "stage2/stage2_analyzer.zarr")
        sparse_analyzer = si.load_sorting_analyzer(
            folder=save_folder / "stage2/stage2_analyzer_sparse.zarr")

        # # Compute extensions on analyzer if they don't already exist
        # extensions_to_compute = [
        #     "random_spikes",
        #     "waveforms",
        #     "templates",
        #     "template_similarity",
        #     "correlograms",
        #     "spike_amplitudes",
        #     "unit_locations",
        # ]

        # extension_params = {"random_spikes": {"method": "all"}, "unit_locations": {"method": "center_of_mass"}}

        # if not analyzer.has_extension("random_spikes"):
        #     analyzer.compute(extensions_to_compute, extension_params=extension_params, **job_kwargs)

        unit_ids = analyzer.unit_ids
        cs = CurationSorting(sorting=analyzer.sorting, make_graph=True)
        # template_ch_dict = template_util.get_template_ch(analyzer)

        # for unit_id in unit_ids:
        for unit_id in unit_ids:
            # may be unnecessary since checked in stage1?
            if exclude_artifact_unit(unit_id, analyzer):
                print(f"Unit {unit_id} is an artifact unit.")
                cs.sorting.set_property("artifact", values=[1], ids=[
                                        unit_id], missing_value=0)
                continue
            print(f"Curating waveforms for unit {unit_id}...")
            good_indices, bad_indices = remove_bad_waveforms_A(
                analyzer, unit_id)

            # minimum spike criterion
            if len(good_indices) < 50:
                print(
                    f"Unit {unit_id} after curating has less than 50 spikes.")
                cs.sorting.set_property("few_spikes", values=[1], ids=[
                                        unit_id], missing_value=0)
                continue

            # primary_ch_idx_dense = template_ch_dict[unit_id]["primary_ch_idx_dense"]
            # wvf_ext = analyzer.get_extension("waveforms")
            # wvfs = wvf_ext.get_waveforms_one_unit(unit_id=unit_id)
            # primary_ch_wvfs = wvfs[good_indices, :, primary_ch_idx_dense]
            primary_ch_wvfs = template_util.get_unit_primary_ch_wvfs(
                analyzer, unit_id)

            num_all_wvfs = len(primary_ch_wvfs)
            # get primary channel waveform samples near peak (most informative)
            samples_to_use = np.arange(20, 42)
            good_primary_ch_wvfs = primary_ch_wvfs[good_indices, :]
            trunc_primary_ch_wvfs = good_primary_ch_wvfs[:, samples_to_use]

            # UMAP + Isosplit to identify clusters within the waveforms
            subcluster_labels, umap_x, umap_y = cluster_waveforms(
                trunc_primary_ch_wvfs)

            # Unique cluster IDs
            subcluster_ids = np.unique(subcluster_labels)

            # Initialize an array to store cluster indices, marking bad waveforms with 0
            waveform_labels = np.full(num_all_wvfs, -1, dtype=int)
            waveform_labels[bad_indices] = 0  # Mark bad indices as 0

            # Ensure the sum of good and bad indices matches the total number of waveforms
            assert len(good_indices) + len(bad_indices) == len(waveform_labels)

            # Ensure the number of good waveforms matches the number of cluster labels
            assert len(good_indices) == len(subcluster_labels)

            # Lists to hold cluster labels
            final_subcluster_labels = []

            for subcluster_id in subcluster_ids:  # 1-indexed labels
                process_single_subcluster(
                    analyzer,
                    unit_id,
                    subcluster_id,
                    subcluster_labels,
                    good_indices,
                    waveform_labels,
                    final_subcluster_labels,
                )

            if np.any(waveform_labels == -1):
                print("Error: Some waveforms in waveform_labels were not assigned.")
                unassigned_indices = np.where(waveform_labels == -1)[0]
                print(f"Unassigned indices: {unassigned_indices}")

            # Ensure the length of waveform_labels is correct
            assert len(
                waveform_labels) == num_all_wvfs, "Length of waveform_labels does not match num_all_wvfs"

            new_unit_ids = cs.split(
                split_unit_id=unit_id, indices_list=waveform_labels)

            parent_unit_id_str = "parent_id" + str(unit_id)
            noise_unit_id = new_unit_ids[0]
            cs.sorting.set_property("noise", values=[1], ids=[
                                    noise_unit_id], missing_value=0)
            cs.sorting.set_property(parent_unit_id_str, values=[1], ids=[
                                    noise_unit_id], missing_value=0)

            # Set properties for the subclusters using new unit IDs, excluding the noise unit
            for new_unit_id, (unique_label, template_label) in zip(new_unit_ids[1:], final_subcluster_labels):
                cs.sorting.set_property(template_label, values=[1], ids=[
                                        new_unit_id], missing_value=0)
                cs.sorting.set_property(parent_unit_id_str, values=[1], ids=[
                                        new_unit_id], missing_value=0)

            cluster_assignments[unit_id] = waveform_labels

            plot_umap_subcluster(unit_id, umap_x, umap_y,
                                 subcluster_labels, subcluster_ids)

            plot_clustered_waveforms(
                analyzer,
                unit_id,
                waveform_labels,
                [label for _, label in final_subcluster_labels],
                plot_mean_std=True,
                N=2,
            )

        # save cluster assignments dict
        with open(save_folder / "cluster_assignments_test.pkl", "wb") as file:
            pickle.dump(cluster_assignments, file)

        analyzer = si.create_sorting_analyzer(
            sorting=cs.sorting,
            recording=analyzer.recording,
            format="zarr",
            folder=Path(stage3_path) / "stage3_analyzer_test.zarr",  # Todo
            sparse=False,
            overwrite=True,
            max_spikes_per_unit=None,
        )

        extensions_to_compute = [
            "random_spikes",
            "waveforms",
            "templates",
            "template_similarity",
            "correlograms",
            "spike_amplitudes",
            "unit_locations",
        ]

        extension_params = {
            "unit_locations": {"method": "center_of_mass"},
            "correlograms": {"window_ms": 100, "bin_ms": 0.5},
        }
        analyzer.compute(extensions_to_compute,
                         extension_params=extension_params, **job_kwargs)

        print("Saving sparse version to disk...")

        sparse_curated_analyzer = template_util.save_sparse_analyzer(
            analyzer, method="zarr", job_kwargs=job_kwargs)

        print("\nStage 3 complete.")


# %%

if __name__ == "__main__":
    path_1 = "E:\\data\\ICMS93\\behavior\\30-Aug-2023"
    path_2 = "C:\\data\\ICMS98\\behavior\\01-Sep-2023"
    # 01 sep

    if os.path.exists(path_1):
        data_folder = path_1
    elif os.path.exists(path_2):
        data_folder = path_2
    else:
        data_folder = None  # or raise an error, or assign a default path
        print("Neither directory exists.")

    analyzer = main(data_folder=None)

# %%
# analyzer.compute("random_spikes")
# analyzer.compute("waveforms")
# analyzer.compute("templates")
# si.plot_unit_templates(analyzer, same_axis=True, x_offset_units=True, plot_legend=True, set_title=False)

# %%
# stage3_path = "C:\\data\\ICMS92\\behavior\\01-Sep-2023\\batch_sort\\stage3"
# analyzer = si.load_sorting_analyzer(Path(stage3_path) / "stage3_analyzer.zarr")

# %%
# app = spikeinterface_gui.mkQApp()
# win = spikeinterface_gui.MainWindow(analyzer, curation=True)
# win.show()
# app.exec_()

# %% Create raw analyzer to get raw waveforms
data_folder = "C:\\data\\ICMS92\\behavior\\06-Sep-2023"
analyzer = si.load_sorting_analyzer(Path(stage3_path) / "stage3_analyzer.zarr")

dataloader = load_data(
    data_folder, make_folder=True, save_folder_name="batch_sort", first_N_files=4, server_mount_drive="S:"
)
rec_raw = dataloader.recording

raw_analyzer = si.create_sorting_analyzer(
    sorting=analyzer.sorting,
    recording=rec_raw,
    format="memory",
    sparse=False,
)

raw_analyzer.compute('random_spikes', method='all')
raw_analyzer.compute('waveforms')
raw_analyzer.compute('templates', operators=['average', 'median'])
raw_analyzer.compute('unit_locations')
# %%
# si.plot_spikes_on_traces(raw_analyzer, time_range=[
#                           200, 220], order_channel_by_depth=True, return_scaled=True)


# %%
template_dict = template_util.get_template_ch(analyzer)
raw_template_ext = raw_analyzer.get_extension('templates')
templates = raw_template_ext.get_data(operator='average')


def polynomial_detrend(data, degree=1):
    time = np.arange(20, 40)
    coeffs = np.polyfit(time, data[time], degree)
    trend = np.polyval(coeffs, np.arange(len(data)))
    return data - trend


unit_ids = raw_analyzer.unit_ids
for idx in range(0, 50):
    plt.figure()
    unit_id = unit_ids[idx]

    # plt.plot(templates[idx, :, :])

    ch_template = np.mean(templates[idx, :, :], axis=1)

    primary_ch_idx_sparse = template_dict[unit_id]['primary_ch_idx_sparse']
    primary_ch_idx_dense = template_dict[unit_id]['primary_ch_idx_dense']

    primary_ch_template = templates[idx, :, primary_ch_idx_dense]

    # detrended_template = primary_ch_template - ch_template

    # centered_template = detrended_template - np.mean(detrended_template)

    # detrended_data = polynomial_detrend(centered_template, degree=3)

    plt.plot(templates[idx, :, :], 'k', linewidth=1)

    plt.plot(primary_ch_template, 'r', linewidth=2)

# %%
for unit_id in unit_ids:
    plt.figure(figsize=(4, 3))
    primary_ch_wvfs = template_util.get_unit_primary_ch_wvfs(
        raw_analyzer, unit_id)
    primary_ch_wvfs = primary_ch_wvfs[0: 200, :]

    centered_waveforms = primary_ch_wvfs - \
        primary_ch_wvfs[:, 30].reshape(-1, 1)

    plt.plot(centered_waveforms.T, 'k', linewidth=0.5)
    plt.title(unit_id)

# %%
num_all_wvfs = len(primary_ch_wvfs)
# get primary channel waveform samples near peak (most informative)
samples_to_use = np.arange(20, 42)
good_primary_ch_wvfs = primary_ch_wvfs[good_indices, :]
trunc_primary_ch_wvfs = good_primary_ch_wvfs[:, samples_to_use]

# UMAP + Isosplit to identify clusters within the waveforms
subcluster_labels, umap_x, umap_y = cluster_waveforms(
    trunc_primary_ch_wvfs)

# %%
# # %%
# # Function to apply bandpass filter


# def bandpass_filter(data, lowcut, highcut, fs, order=4):
#     nyquist = 0.5 * fs
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = butter(order, [low, high], btype='band')
#     return filtfilt(b, a, data)


# # Example parameters for the filter
# lowcut = 300  # Low cutoff frequency (Hz)
# highcut = 5000  # High cutoff frequency (Hz)
# fs = 30000  # Sampling frequency (e.g., 30 kHz)


# mean_templates = raw_template_ext.get_data(operator='average')
# median_templates = raw_template_ext.get_data(operator='median')
# unit_to_num_spikes = raw_analyzer.sorting.get_total_num_spikes()

# for idx in range(30, 40):
#     plt.figure()
#     unit_id = unit_ids[idx]

#     # Extract the mean and median templates for the current unit
#     # Assuming mean_templates has shape [units, samples, channels]
#     mean_template = mean_templates[idx, :, :]
#     median_template = median_templates[idx, :, :]

#     primary_ch_idx_sparse = template_dict[unit_id]['primary_ch_idx_sparse']
#     primary_ch_idx_dense = template_dict[unit_id]['primary_ch_idx_dense']

#     # Extract mean and median for the primary channel
#     primary_ch_mean = mean_template[:, primary_ch_idx_dense]
#     primary_ch_median = median_template[:, primary_ch_idx_dense]

#     # Time points (assuming length of the waveform matches the number of samples)
#     time_points = np.arange(primary_ch_mean.shape[0])

#     # Detrend by subtracting the template
#     ch_template = np.median(mean_template, axis=1)
#     detrended_mean = primary_ch_mean - ch_template
#     detrended_median = primary_ch_median - ch_template

#     # Center the detrended data
#     centered_mean = detrended_mean - np.mean(detrended_mean[22:25])
#     centered_median = detrended_median - np.mean(detrended_median[22:25])

#     # Apply the bandpass filter (300 Hz to 5000 Hz)
#     filtered_mean = bandpass_filter(primary_ch_mean, lowcut, highcut, fs)
#     filtered_median = bandpass_filter(primary_ch_median, lowcut, highcut, fs)

#     # Plot the centered mean and median templates
#     plt.plot(time_points, centered_mean, 'r',
#              linewidth=2, label="Centered Mean")
#     plt.plot(time_points, centered_median, 'k',
#              linewidth=2, label="Centered Median")

#     # Plot the filtered templates
#     plt.plot(time_points, filtered_mean, 'b', linestyle='--',
#              linewidth=2, label="Filtered Mean (300-5000 Hz)")
#     plt.plot(time_points, filtered_median, 'g', linestyle='--',
#              linewidth=2, label="Filtered Median (300-5000 Hz)")

#     # Add labels, legend, and title
#     plt.xlabel("Time")
#     plt.ylabel("Amplitude")
#     plt.title(f"{unit_id} num spikes: {unit_to_num_spikes[unit_id]}")
#     plt.legend()
#     plt.show()
