from pathlib import Path
import time
import matplotlib.pyplot as plt

from batch_process.util.file_util import file_dialog
from util.load_data import load_data
from spikeinterface import full as si
import spikeinterface.preprocessing as sp
from preprocessing.preprocessing_pipelines import new_pipeline3
from batch_process.util.plotting import plot_units_in_batches
import batch_process.util.file_util as file_util
from batch_process.util.curate_util import *
from batch_process.util.misc import *

#  Load example data


def main(data_folder=None):
    if data_folder:
        data_folders = [data_folder]
    else:
        starting_dir = "C:\\data"
        data_folders = file_dialog(starting_dir=starting_dir)

    job_kwargs = dict(n_jobs=5, chunk_duration="1s", progress_bar=True)

    sorted_data_folders = file_util.sort_data_folders(data_folders)
    for i, data_folder in enumerate(sorted_data_folders):
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
        offset_stim_ts = [[int(item + 0) for item in sublist]
                          for sublist in all_stim_timestamps]

        art_remove_pre = 0.5
        art_remove_post1 = 1.4
        art_remove_post2 = 1.5
        trend_remove_post_pulse_start = 1.4
        trend_remove_post_pulse_end = 10

        # Since custom preprocessors are trial-based, default time chunking
        # when saving the recording object during sorting is not appropriate. Rather, get traces, convert to rec object,
        # and then continue with rest of non trial-based preprocessing.
        rec1 = sp.remove_artifacts(
            rec, flattened_stim, ms_before=art_remove_pre, ms_after=art_remove_post1, mode="zeros"
        )
        rec2 = sp.mean_artifact_subtract(
            rec1, list_triggers=offset_stim_ts, post_stim_window_ms=10, mode="median")
        rec3 = sp.trend_subtract(
            rec2,
            all_stim_timestamps,
            trend_remove_post_pulse_start,
            trend_remove_post_pulse_end,
            mode="poly",
            poly_order=3,
        )

        print("Loading partially preprocessed traces into memory...")
        trace3 = rec3.get_traces(return_scaled=False)
        rec_ts = si.numpyextractors.NumpyRecording(
            trace3, sampling_frequency=fs, channel_ids=rec.channel_ids)
        rec_ts.set_channel_gains(rec.get_channel_gains())
        rec_ts.set_channel_offsets(rec.get_channel_offsets())
        rec_ts.set_channel_locations(rec.get_channel_locations())
        del trace3

        # Write to .zarr format to spike sort
        stage1_path = Path(save_folder) / "stage1"
        file_util.create_folder(stage1_path)
        preproc_zarr_folder = stage1_path / "custom_preproc"

        rec4 = sp.common_reference(
            rec_ts, operator="median", reference="global")
        rec5 = sp.remove_artifacts(
            rec4, flattened_stim, ms_before=art_remove_pre, ms_after=art_remove_post1, mode="cubic"
        )
        rec6 = sp.bandpass_filter(rec5, freq_min=300, freq_max=5000)
        rec7 = sp.remove_artifacts(
            rec6, flattened_stim, ms_before=art_remove_pre, ms_after=art_remove_post2, mode="zeros"
        )
        rec7.save_to_zarr(folder=preproc_zarr_folder,
                          overwrite=True, **job_kwargs)
        # Used for visualizing data (step before whitening)
        rec_preproc = si.load_extractor(
            preproc_zarr_folder.with_suffix(".zarr"))

        # Sort
        ms5_params = {
            "scheme": "2",
            "detect_threshold": 4, # TODO change back to 5.5
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
            recording=si.scale(rec_preproc, dtype="float"),
            remove_existing_folder=True,
            verbose=True,
            **ms5_params,
        )

        # save raw analyzer to disk
        analyzer = si.create_sorting_analyzer(
            sorting=sorting,
            recording=rec_preproc,
            format="zarr",
            folder=Path(stage1_path) / "stage1_analyzer_raw.zarr",
            sparse=False,
            overwrite=True,
        )

        extensions_to_compute = [
            "random_spikes",
            "waveforms",
            "templates",
            "correlograms",
        ]

        extension_params = {"random_spikes": {
            "method": "all"}, "correlograms": {"window_ms": 100}}

        analyzer.compute(extensions_to_compute,
                         extension_params=extension_params, **job_kwargs)

        # spike_vector = analyzer.sorting.to_spike_vector()
        # last_spike = spike_vector[-1][0]
        # rec = analyzer.recording

        # save sparse version of raw analyzer
        sparse_analyzer = template_util.save_sparse_analyzer(
            analyzer, method="zarr", job_kwargs=job_kwargs)

        # plot figures before curating using sparse templates
        plot_units_in_batches(
            sparse_analyzer, save_dir=stage1_path, ppt_name="raw")

        # curate units and save plotted figures in folder
        good_ids, bad_ids = curate_units(analyzer, sparse_analyzer, job_kwargs)

        # save analyzer
        stage1_analyzer_path = stage1_path / "stage1_analyzer_curated.zarr"
        if os.path.isdir(stage1_analyzer_path):
            shutil.rmtree(stage1_analyzer_path)

        curated_analyzer = analyzer.select_units(
            unit_ids=good_ids, format="zarr", folder=stage1_path / "stage1_analyzer_curated"
        )

        extensions_to_compute_curated = [
            "random_spikes",
            "waveforms",
            "templates",
            "template_similarity",
            "correlograms",
            "spike_amplitudes",
            "unit_locations",
        ]

        extension_params_curated = {
            "random_spikes": {"method": "all"},
            "unit_locations": {"method": "center_of_mass"},
            "correlograms": {"window_ms": 100},
        }

        # Compute the necessary extensions for the curated analyzer
        curated_analyzer.compute(extensions_to_compute_curated,
                                 extension_params=extension_params_curated, **job_kwargs)
        # Create a sparse version of the curated analyzer
        sparse_curated_analyzer = template_util.save_sparse_analyzer(
            curated_analyzer, method="zarr", job_kwargs=job_kwargs
        )

        plot_units_in_batches(sparse_curated_analyzer,
                              save_dir=stage1_path, ppt_name="curated")

        # Delete the raw analyzer and its sparse version to save disk space after the curation is complete
        shutil.rmtree(analyzer.folder)
        shutil.rmtree(sparse_analyzer.folder)
        # shutil.rmtree(preproc_zarr_folder.with_suffix(".zarr")) # i think i need this to compute extensions
        del analyzer
        del sparse_analyzer
        del curated_analyzer
        del sparse_curated_analyzer

        end_time = time.time()
        time_elapsed = end_time - start_time
        print(f"{data_folder}: {i+1}/{len(data_folders)} stage 1 complete.")
        print(f"Time elapsed: {time_elapsed:.2f} seconds")

        # curated_analyzer2 = si.load_sorting_analyzer(folder=save_folder / "stage1/stage1_analyzer_curated.zarr")

        # extensions_to_compute = [
        #     "random_spikes",
        #     "waveforms",
        #     "templates",
        #     "template_similarity",
        #     "correlograms",
        #     "spike_amplitudes",
        #     "unit_locations",
        # ]

        # extensions_to_compute = [
        #     "random_spikes",
        #     "waveforms",
        #     "templates",
        #     "template_similarity",
        #     "correlograms",
        #     "spike_amplitudes",
        #     "unit_locations",
        # ]

        # extension_params = {
        #     "unit_locations": {"method": "center_of_mass"},
        # }

        # curated_analyzer.compute(extensions_to_compute, extension_params=extension_params, **job_kwargs)


# %%
if __name__ == "__main__":
    path_1 = "E:\\data\\ICMS93\\behavior\\30-Aug-20231"
    path_2 = "C://data//ICMS100//02-Nov-20231"
    if os.path.exists(path_1):
        data_folder = path_1
    elif os.path.exists(path_2):
        data_folder = path_2
    else:

        data_folder = None  # or raise an error, or assign a default path
        print("Neither directory exists.")

    # main(data_folder=data_folder)
    main()


# %%
# rms_noise_raw_list = []
# rms_noise_filt_list = []

# traces = rec.get_traces(start_frame=0, end_frame=30000 * 5, return_scaled=True)

# for trace in traces.T:
#     rms_noise_raw = np.sqrt(np.mean(trace**2))
#     rms_noise_raw_list.append(rms_noise_raw)


# rec1 = sp.common_reference(rec, operator="median", reference="global")
# rec2 = sp.bandpass_filter(rec1, freq_min=1000, freq_max=14999)

# filt_traces = rec2.get_traces(start_frame=0, end_frame=30000 * 5, return_scaled=True)

# for trace in filt_traces.T:
#     rms_noise_filt = np.sqrt(np.mean(trace**2))
#     rms_noise_filt_list.append(rms_noise_filt)

# data = {
#     "RMS Noise": rms_noise_raw_list + rms_noise_filt_list,
#     "Condition": ["Raw"] * len(rms_noise_raw_list) + ["Filtered"] * len(rms_noise_filt_list),
#     "Channel": list(range(len(rms_noise_raw_list))) * 2,
# }

# df = pd.DataFrame(data)

# plt.figure(figsize=(10, 6))

# # Create a strip plot to show individual data points
# sns.stripplot(x="Condition", y="RMS Noise", data=df, jitter=True, alpha=0.5)

# # Overlay with a box plot to show the distribution
# sns.boxplot(
#     x="Condition",
#     y="RMS Noise",
#     data=df,
#     showcaps=False,
#     boxprops={"facecolor": "None"},
#     showfliers=False,
#     whiskerprops={"linewidth": 0},
# )

# # Show the plot
# plt.title("Comparison of RMS Noise (Raw vs Filtered)")
# plt.show()
