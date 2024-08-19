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

import warnings

# ignore OMP_NUM_THREADS memory leaks warning
# warnings.filterwarnings("ignore")

# %%


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
    template_ch_dict
):
    relative_indices = np.where(subcluster_labels == subcluster_id)[0]
    # Map these indices to their original indices using good_indices
    subcluster_indices = good_indices[relative_indices]

    kept_indices, discarded_indices, subcluster_label = accept_subcluster(
        analyzer, unit_id, subcluster_indices, template_ch_dict
    )

    if subcluster_label == "accept":
        if len(kept_indices) > 0:
            waveform_labels[subcluster_indices[kept_indices]
                            ] = subcluster_id
        if len(discarded_indices) > 0:
            waveform_labels[subcluster_indices[discarded_indices]] = 0
    else:
        waveform_labels[subcluster_indices] = subcluster_id

    final_subcluster_labels.append((subcluster_id, subcluster_label))


def write_stage3_outputs(analyzer, cs, save_folder):

    analyzer = si.create_sorting_analyzer(
        sorting=cs.sorting,
        recording=analyzer.recording,
        format="zarr",
        folder=Path(save_folder) / "stage2/stage2_analyzer.zarr",
        sparse=False,
        overwrite=True,
        max_spikes_per_unit=None,
    )

    # # to avoid large provenance file being saved on disk, create light sorting object
    # heavy_sorting = align_waveforms(we, cs.sorting)
    # si.NpzSortingExtractor.write_sorting(
    #     heavy_sorting, save_folder / "sorting_3.npz")
    # light_sorting = si.read_npz_sorting(save_folder / "sorting_3.npz")
    # # copy properties
    # prop_keys = heavy_sorting.get_property_keys()
    # for prop_key in prop_keys:
    #     prop_mask = heavy_sorting.get_property(prop_key)
    #     light_sorting.set_property(
    #         prop_key, values=prop_mask, ids=heavy_sorting.unit_ids, missing_value=0
    #     )
    # light_sorting.save_to_folder(
    #     folder=save_folder / "sorting_3", overwrite=True)

    # # write waveform extractor to disk using new sorting
    # si.extract_waveforms(
    #     we.recording,
    #     light_sorting,
    #     folder=save_folder / "waveforms_3",
    #     ms_before=1.0,
    #     ms_after=2.0,
    #     sparse=None,
    #     overwrite=True,
    #     max_spikes_per_unit=None,
    #     **job_kwargs,
    # )


def main(debug_folder):
    job_kwargs = dict(n_jobs=5, chunk_duration="1s", progress_bar=True)

    if debug_folder:
        data_folders = [debug_folder]
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

        analyzer.compute("random_spikes", method="all")
        analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)
        analyzer.compute("templates")
        analyzer.compute("template_similarity")
        analyzer.compute("correlograms")
        analyzer.compute("spike_amplitudes")
        analyzer.compute("unit_locations")

        unit_ids = analyzer.unit_ids
        cs = CurationSorting(sorting=analyzer.sorting, make_graph=True)
        template_ch_dict = get_template_ch(analyzer)

        # for unit_id in unit_ids:
        for unit_id in unit_ids[0:3]:
            # may be unnecessary since checked in stage1?
            if exclude_artifact_unit(unit_id, analyzer):
                print(f"Unit {unit_id} is an artifact unit.")
                cs.sorting.set_property(
                    "artifact", values=[1], ids=[unit_id], missing_value=0
                )
                continue
            print(f"Curating waveforms for unit {unit_id}...")
            good_indices, bad_indices = remove_bad_waveforms_A(
                analyzer, unit_id, template_ch_dict)

            # minimum spike criterion
            if len(good_indices) < 50:
                print(
                    f"Unit {unit_id} after curating has less than 50 spikes.")
                cs.sorting.set_property(
                    "few_spikes", values=[1], ids=[unit_id], missing_value=0
                )
                continue

            primary_ch_idx_dense = template_ch_dict[unit_id]["primary_ch_idx_dense"]
            wvf_ext = analyzer.get_extension("waveforms")
            wvfs = wvf_ext.get_waveforms_one_unit(unit_id=unit_id)
            primary_ch_wvfs = wvfs[good_indices, :, primary_ch_idx_dense]

            num_all_wvfs = len(wvfs)
            # get primary channel waveform samples near peak (most informative)
            samples_to_use = np.arange(20, 42)
            trunc_primary_ch_wvfs = primary_ch_wvfs[:, samples_to_use]

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
                    template_ch_dict
                )

            if np.any(waveform_labels == -1):
                print("Error: Some waveforms in waveform_labels were not assigned.")
                unassigned_indices = np.where(waveform_labels == -1)[0]
                print(f"Unassigned indices: {unassigned_indices}")

            # Ensure the length of waveform_labels is correct
            assert (
                len(waveform_labels) == num_all_wvfs
            ), "Length of waveform_labels does not match num_all_wvfs"

            new_unit_ids = cs.split(
                split_unit_id=unit_id, indices_list=waveform_labels
            )

            parent_unit_id_str = "parent_id" + str(unit_id)
            noise_unit_id = new_unit_ids[0]
            cs.sorting.set_property(
                "noise", values=[1], ids=[noise_unit_id], missing_value=0
            )
            cs.sorting.set_property(
                parent_unit_id_str, values=[1], ids=[noise_unit_id], missing_value=0
            )

            # Set properties for the subclusters using new unit IDs, excluding the noise unit
            for new_unit_id, (unique_label, template_label) in zip(
                new_unit_ids[1:], final_subcluster_labels
            ):
                cs.sorting.set_property(
                    template_label, values=[1], ids=[new_unit_id], missing_value=0
                )
                cs.sorting.set_property(
                    parent_unit_id_str, values=[1], ids=[new_unit_id], missing_value=0
                )

            cluster_assignments[unit_id] = waveform_labels

            plot_umap_subcluster(unit_id, umap_x, umap_y,
                                 subcluster_labels, subcluster_ids)
            plot_clustered_waveforms(analyzer, template_ch_dict, unit_id, waveform_labels, [
                label for _, label in final_subcluster_labels], plot_mean_std=True, N=2)

        # save cluster assignments dict
        with open(save_folder / "cluster_assignments.pkl", "wb") as file:
            pickle.dump(cluster_assignments, file)

        analyzer = si.create_sorting_analyzer(
            sorting=cs.sorting,
            recording=analyzer.recording,
            format="zarr",
            folder=Path(stage3_path) / "stage3_analyzer.zarr",
            sparse=False,
            overwrite=True,
            max_spikes_per_unit=None,
        )

        return analyzer


if __name__ == "__main__":
    path_1 = 'E:\\data\\ICMS93\\behavior\\30-Aug-2023'
    path_2 = 'C:\\data\\ICMS93\\behavior\\30-Aug-2023'

    if os.path.exists(path_1):
        debug_folder = path_1
    elif os.path.exists(path_2):
        debug_folder = path_2
    else:
        debug_folder = None  # or raise an error, or assign a default path
        print("Neither directory exists.")

    analyzer = main(debug_folder=debug_folder)

# %%
analyzer.compute("random_spikes")
analyzer.compute("waveforms")
analyzer.compute("templates")
si.plot_unit_templates(analyzer, same_axis=True,
                       x_offset_units=True, plot_legend=True, set_title=False)

# %%
