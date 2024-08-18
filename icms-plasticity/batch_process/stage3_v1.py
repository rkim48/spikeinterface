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
warnings.filterwarnings("ignore")


# %% Attempting to fix the undersplitting problem


def cluster_waveforms(X):
    # X is len(waveforms) x len(samples) x 1
    # returns labels for subclusters for each Mountainsort unit
    n_neighbors = 50
    min_dist = 0.1
    for iteration in range(1, 6):
        reducer = umap.UMAP(n_neighbors=n_neighbors,
                            min_dist=min_dist, n_jobs=-1)
        wvfs_umap = reducer.fit_transform(X)
        x = wvfs_umap[:, 0]
        y = wvfs_umap[:, 1]
        final_labels = isosplit6(wvfs_umap)
        if len(np.unique(final_labels)) <= 4:
            break
        n_neighbors += 10
        min_dist += 0.1
    else:
        print("Failed to produce <= 4 clusters after 5 iterations.")
    return final_labels, x, y


def process_single_subcluster(
    we,
    unit_id,
    subcluster_id,
    final_labels,
    good_idx,
    split_indices_list,
    subcluster_labels,
):
    rel_indices = np.where(final_labels == subcluster_id)[0]
    # Map these indices to their original indices using good_idx
    subcluster_indices = good_idx[rel_indices]

    kept_indices, discarded_indices, template_label = accept_subcluster(
        we, unit_id, subcluster_indices
    )

    if template_label == "accept":
        if len(kept_indices) > 0:
            split_indices_list[subcluster_indices[kept_indices]
                               ] = subcluster_id
        if len(discarded_indices) > 0:
            split_indices_list[subcluster_indices[discarded_indices]] = 0
    else:
        split_indices_list[subcluster_indices] = subcluster_id

    subcluster_labels.append((subcluster_id, template_label))


def write_stage3_outputs(we, cs, save_folder, job_kwargs):
    we = si.extract_waveforms(
        we.recording,
        cs.sorting,
        folder=None,
        mode="memory",
        ms_before=1.0,
        ms_after=2.0,
        sparse=None,
        max_spikes_per_unit=None,
        **job_kwargs,
    )

    # to avoid large provenance file being saved on disk, create light sorting object
    heavy_sorting = align_waveforms(we, cs.sorting)
    si.NpzSortingExtractor.write_sorting(
        heavy_sorting, save_folder / "sorting_3.npz")
    light_sorting = si.read_npz_sorting(save_folder / "sorting_3.npz")
    # copy properties
    prop_keys = heavy_sorting.get_property_keys()
    for prop_key in prop_keys:
        prop_mask = heavy_sorting.get_property(prop_key)
        light_sorting.set_property(
            prop_key, values=prop_mask, ids=heavy_sorting.unit_ids, missing_value=0
        )
    light_sorting.save_to_folder(
        folder=save_folder / "sorting_3", overwrite=True)

    # write waveform extractor to disk using new sorting
    si.extract_waveforms(
        we.recording,
        light_sorting,
        folder=save_folder / "waveforms_3",
        ms_before=1.0,
        ms_after=2.0,
        sparse=None,
        overwrite=True,
        max_spikes_per_unit=None,
        **job_kwargs,
    )


def main():
    starting_dir = "C:\\data"
    data_folders = file_dialog(starting_dir=starting_dir)
    job_kwargs = dict(n_jobs=5, chunk_duration="1s", progress_bar=True)

    for i, data_folder in enumerate(data_folders):
        print("\n###########################################")
        print(f"{data_folder}: {i+1}/{len(data_folders)}")
        print("###########################################")

        cluster_assignments = {}
        # data_folder = 'C:\\data\\ICMS93\\Behavior\\30-Aug-2023'
        save_folder = Path(data_folder) / "batch_sort"
        we = si.load_waveforms(folder=save_folder /
                               "waveforms_2", with_recording=True)
        sorting = we.sorting
        job_kwargs = dict(n_jobs=5, chunk_duration="1s", progress_bar=True)
        # Make new curation sorting object using splits and merges
        unit_ids = we.unit_ids
        cs = CurationSorting(parent_sorting=sorting, make_graph=True)

        for unit_id in unit_ids:  # TEST UNIT
            if exclude_artifact_unit(unit_id, we):
                print(f"Unit {unit_id} is an artifact unit.")
                cs.sorting.set_property(
                    "artifact", values=[1], ids=[unit_id], missing_value=0
                )
                continue
            print(f"Curating waveforms for unit {unit_id}...")
            good_idx, bad_idx = remove_bad_waveforms_A(we, unit_id)

            if len(good_idx) < 50:
                print(
                    f"Unit {unit_id} after curating has less than 50 spikes.")
                cs.sorting.set_property(
                    "few_spikes", values=[1], ids=[unit_id], missing_value=0
                )
                continue

            template_ch_dict = get_template_ch(we)
            primary_ch_idx_dense = template_ch_dict[unit_id]["primary_ch_idx_dense"]
            # use samples near peak (most informative)
            samples_to_use = np.arange(20, 42)

            num_all_wvfs = len(we.get_waveforms(unit_id=unit_id))
            primary_ch_wvfs = we.get_waveforms(unit_id=unit_id)[good_idx]
            trunc_primary_ch_wvfs = primary_ch_wvfs[
                :, samples_to_use, primary_ch_idx_dense
            ]

            # UMAP + isosplit to cluster waveforms
            final_labels, x, y = cluster_waveforms(trunc_primary_ch_wvfs)

            subcluster_ids = np.unique(final_labels)
            split_indices_list = np.full(num_all_wvfs, -1, dtype=int)
            split_indices_list[bad_idx] = 0  # fill with bad indices
            assert len(good_idx) + len(bad_idx) == len(split_indices_list)
            assert len(good_idx) == len(final_labels)

            subcluster_labels = []
            mean_norm_peaks = []

            for subcluster_id in subcluster_ids:  # 1-indexed labels
                process_single_subcluster(
                    we,
                    unit_id,
                    subcluster_id,
                    final_labels,
                    good_idx,
                    split_indices_list,
                    subcluster_labels,
                )

            if np.any(split_indices_list == -1):
                print("Error: Some indices in split_indices_list were not assigned.")
                unassigned_indices = np.where(split_indices_list == -1)[0]
                print(f"Unassigned indices: {unassigned_indices}")

            # Ensure the length of split_indices_list is correct
            assert (
                len(split_indices_list) == num_all_wvfs
            ), "Length of split_indices_list does not match num_all_wvfs"

            new_unit_ids = cs.split(
                split_unit_id=unit_id, indices_list=split_indices_list
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
                new_unit_ids[1:], subcluster_labels
            ):
                cs.sorting.set_property(
                    template_label, values=[1], ids=[new_unit_id], missing_value=0
                )
                cs.sorting.set_property(
                    parent_unit_id_str, values=[1], ids=[new_unit_id], missing_value=0
                )

            cluster_assignments[unit_id] = split_indices_list

            # plot_umap_subcluster(unit_id, x, y, final_labels, subcluster_ids)
            # plot_clustered_waveforms(we, unit_id, split_indices_list, [
            #     label for _, label in subcluster_labels], mean_norm_peaks, plot_mean_std=True, N=2)

        # save cluster assignments dict
        with open(save_folder / "cluster_assignments.pkl", "wb") as file:
            pickle.dump(cluster_assignments, file)

        write_stage3_outputs(we, cs, save_folder, job_kwargs)


# %%
if __name__ == "__main__":
    main()

# %%
