from pathlib import Path
import time
import matplotlib.pyplot as plt

from util.file_util import file_dialog
from util.load_data import load_data
from spikeinterface import full as si
import spikeinterface.preprocessing as sp
from preprocessing.preprocessing_pipelines import new_pipeline3
from batch_process.util.plotting import plot_units_in_batches
from batch_process.util.file_util import create_folder
from batch_process.util.curate_util import *
from batch_process.util.misc import *

# %% Load example data


def main(debug_folder=None):
    if debug_folder:
        data_folders = [debug_folder]
    else:
        starting_dir = "C:\\data"
        data_folders = file_dialog(starting_dir=starting_dir)

    job_kwargs = dict(n_jobs=1, chunk_duration="1s", progress_bar=True)

    for i, data_folder in enumerate(data_folders):
        print("\n###########################################")
        print(f"{data_folder}: {i+1}/{len(data_folders)}")
        print("###########################################")
        start_time = time.time()
        dataloader = load_data(
            data_folder, make_folder=True, save_folder_name="batch_sort", first_N_files=4, server_mount_drive="S:"
        )
        rec = dataloader.recording

        animalID = dataloader.animalID
        channel_ids = dataloader.channel_ids
        all_stim_timestamps = dataloader.all_stim_timestamps
        save_folder = dataloader.save_folder
        fs = dataloader.fs

        # Preprocessing parameters
        flattened_stim = [ts for l in all_stim_timestamps for ts in l]
        offset_stim_ts = [[int(item + 0) for item in sublist] for sublist in all_stim_timestamps]

        art_remove_pre = 0.5
        art_remove_post1 = 1.4
        art_remove_post2 = 1.5
        trend_remove_post_pulse_start = 1.4
        trend_remove_post_pulse_end = 10

        #  Apply preprocessing steps. Since custom preprocessors are trial-based, default time chunking
        # when saving the recording object during sorting is not appropriate. Rather, get traces, convert to rec object,
        # and then use that for sorting.
        rec1 = sp.remove_artifacts(
            rec, flattened_stim, ms_before=art_remove_pre, ms_after=art_remove_post1, mode="zeros"
        )
        rec2 = sp.mean_artifact_subtract(rec1, list_triggers=offset_stim_ts, post_stim_window_ms=10, mode="median")
        rec3 = sp.trend_subtract(
            rec2,
            all_stim_timestamps,
            trend_remove_post_pulse_start,
            trend_remove_post_pulse_end,
            mode="poly",
            poly_order=3,
        )
        rec4 = sp.common_reference(rec3, operator="median", reference="global")
        rec5 = sp.remove_artifacts(
            rec4, flattened_stim, ms_before=art_remove_pre, ms_after=art_remove_post1, mode="cubic"
        )
        rec6 = sp.bandpass_filter(rec5, freq_min=300, freq_max=5000)
        rec7 = sp.remove_artifacts(
            rec6, flattened_stim, ms_before=art_remove_pre, ms_after=art_remove_post2, mode="zeros"
        )

        # Load traces into memory
        print("Loading preprocessed traces into memory...")
        trace7 = rec7.get_traces(return_scaled=False)

        # Convert to recording object
        rec_pre = si.numpyextractors.NumpyRecording(trace7, sampling_frequency=fs, channel_ids=rec.channel_ids)
        rec_pre.set_channel_gains(rec.get_channel_gains())
        rec_pre.set_channel_offsets(rec.get_channel_offsets())
        rec_pre.set_channel_locations(rec.get_channel_locations())

        # Write to .zarr format to spike sort
        stage1_path = Path(save_folder) / "stage1"
        create_folder(stage1_path)
        zarr_folder = stage1_path / "preproc"
        rec_pre.save_to_zarr(folder=zarr_folder, overwrite=True, **job_kwargs)
        rec_pre = si.load_extractor(zarr_folder.with_suffix(".zarr"))

        # Sort
        ms5_params = {
            "scheme": "2",
            "detect_threshold": 5.5,
            "npca_per_channel": 3,
            "npca_per_subdivision": 10,
            "snippet_mask_radius": 0,
            "scheme2_detect_channel_radius": 200,
            "scheme2_training_duration_sec": 300,
            "filter": False,
            "whiten": True,
        }

        sorting = si.run_sorter(
            sorter_name="mountainsort5",
            recording=si.scale(rec_pre, dtype="float"),
            folder=stage1_path / "sorting",
            remove_existing_folder=True,
            verbose=True,
            **ms5_params,
        )

        analyzer = si.create_sorting_analyzer(
            sorting=sorting, recording=rec_pre, format="memory", sparse=False, return_scaled=False
        )

        analyzer.compute("random_spikes", method="all")  # gets all spikes
        analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)
        analyzer.compute("templates")
        analyzer.compute("correlograms")

        # plot figures before curating
        plot_units_in_batches(analyzer, save_dir=stage1_path, ppt_name="test")

        # # curate units and save plotted figures in folder
        good_ids, bad_ids = curate_units(analyzer)
        # save analyzer
        analyzer.select_units(unit_ids=good_ids, format="zarr", folder=stage1_path / "stage1_analyzer")

        # bad_events_dict = get_bad_events_dict(bad_analyzer, bad_ids, sorting)
        # replace_ch_indices = list(bad_events_dict.keys())

        # stage1B_filt = rec_pre
        # for ch_idx in replace_ch_indices:
        #     timestamps = bad_events_dict[ch_idx]
        #     stage1B_filt = si.remove_artifacts(
        #         stage1B_filt,
        #         timestamps,
        #         ms_before=0.25,
        #         ms_after=0.25,
        #         mode="zeros",
        #     )

        # sorting_1B = si.run_sorter(
        #     sorter_name="mountainsort5",
        #     recording=si.scale(stage1B_filt, dtype='float'),
        #     folder=Path(save_folder) / "sorting_1B",
        #     remove_existing_folder=True,
        #     verbose=True,
        #     **ms5_params,
        # )

        # analyzerB = si.create_sorting_analyzer(
        #     sorting=sorting_1B,
        #     recording=stage1B_filt,
        #     format="memory",
        #     sparse=False,
        #     return_scaled=False
        # )

        # analyzerB.compute("random_spikes", method="all")  # gets all spikes
        # analyzerB.compute("waveforms", ms_before=1.0, ms_after=2.0)
        # analyzerB.compute("templates")
        # analyzerB.compute("correlograms")

        # bad_we, bad_ids, good_we = curate_and_plot(
        #     analyzerB,
        #     parent_folder=save_folder,
        #     subfolder_path="results1_B",
        # )

        end_time = time.time()
        time_elapsed = end_time - start_time
        print(f"Time elapsed: {time_elapsed:.2f} seconds")


# %%
if __name__ == "__main__":
    path_1 = "E:\\data\\ICMS93\\behavior\\30-Aug-2023"
    path_2 = "C:\\data\\ICMS93\\behavior\\30-Aug-2023"

    if os.path.exists(path_1):
        debug_folder = path_1
    elif os.path.exists(path_2):
        debug_folder = path_2
    else:
        debug_folder = None  # or raise an error, or assign a default path
        print("Neither directory exists.")

    main(debug_folder=debug_folder)
