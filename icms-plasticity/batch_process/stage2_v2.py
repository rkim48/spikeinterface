import numpy as np
from spikeinterface import full as si
from util.file_util import *
import os

# from curation_test.curation_module import *
from batch_process.util.curate_util import *
from batch_process.util.misc import *
import batch_process.util.template_util as template_util
import batch_process.util.file_util as file_util


def main(data_folder):
    if data_folder:
        data_folders = [data_folder]
    else:
        starting_dir = "C:\\data"
        data_folders = file_dialog(starting_dir=starting_dir)

    job_kwargs = dict(n_jobs=1, chunk_duration="1s", progress_bar=True)
    si.set_global_job_kwargs(**job_kwargs)

    sorted_data_folders = file_util.sort_data_folders(data_folders)
    for i, data_folder in enumerate(sorted_data_folders):
        print("\n###########################################")
        print(f"{data_folder}: {i+1}/{len(data_folders)}")
        print("###########################################")

        save_folder = Path(data_folder) / "batch_sort"
        stage2_path = Path(save_folder) / "stage2"
        create_folder(stage2_path)
        analyzer = si.load_sorting_analyzer(folder=save_folder / "stage1/stage1_analyzer_curated.zarr")

        # Recompute extensions so sortingview displays correct information
        extensions_to_compute = [
            "random_spikes",
            "waveforms",
            "templates",
            "template_similarity",
            "correlograms",
            "spike_amplitudes",
            "unit_locations",
        ]

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
        }

        analyzer.compute(extensions_to_compute, extension_params=extension_params, **job_kwargs)

        si.plot_sorting_summary(analyzer, curation=True, backend="sortingview")
        uri = input("\nPlease enter uri: ")
        manual_curation_sorting = si.apply_sortingview_curation(analyzer.sorting, uri_or_json=uri)

        keep_idx = np.where(manual_curation_sorting.get_property("accept"))[0]
        unit_ids = manual_curation_sorting.get_unit_ids()
        good_ids = unit_ids[keep_idx]

        # rename units so they are ordered from id 0 to N
        good_sorting = analyzer.sorting.select_units(unit_ids=good_ids, renamed_unit_ids=np.arange(len(good_ids)))

        curated_analyzer = si.create_sorting_analyzer(
            sorting=good_sorting,
            recording=analyzer.recording,
            format="zarr",
            folder=Path(stage2_path) / "stage2_analyzer.zarr",
            sparse=False,
            overwrite=True,
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

        # Get all spikes for stage 3
        extension_params = {"random_spikes": {"method": "all"}, "unit_locations": {"method": "center_of_mass"}}

        curated_analyzer.compute(extensions_to_compute, extension_params=extension_params, **job_kwargs)

        print("Saving sparse version to disk...")

        sparse_curated_analyzer = template_util.save_sparse_analyzer(
            curated_analyzer, method="zarr", job_kwargs=job_kwargs
        )

        print("\nStage 2 complete.")

        # Debugging
        # analyzer.compute("random_spikes")
        # analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)
        # w = analyzer.get_extension("waveform s")
        # wvf = w.get_waveforms_one_unit(analyzer.unit_ids[10])
        # plt.plot(wvf[121,:])
        # plot_units_in_batches(sparse_curated_analyzer, save_dir=stage2_path, ppt_name="curated")


# %%
if __name__ == "__main__":
    path_1 = "E:\\data\\ICMS93\\behavior\\30-Aug-2023"
    path_2 = "C:\\data\\ICMS93\\behavior\\30-Aug-2023"

    # path_2 = "C:\\data\\ICMS92\\behavior\\27-Sep-2023"

    if os.path.exists(path_1):
        data_folder = path_1
    elif os.path.exists(path_2):
        data_folder = path_2
    else:
        data_folder = None  # or raise an error, or assign a default path
        print("Neither directory exists.")

    # main(data_folder=data_folder)
    main(data_folder=None)

# %%

# data_folder = "C:\data\ICMS93\Behavior\30-Aug-2023\batch_sort"
# analyzer = si.load_sorting_analyzer(folder=save_folder / "stage2/stage2_analyzer.zarr")

# spike_vector = analyzer.sorting.to_spike_vector()
# last_spike = spike_vector[-1][0]
# rec = analyzer.recording
# %%
# unit_colors = {unit_id: 'k' for unit_id in analyzer.unit_ids}
# si.plot_unit_templates(analyzer, same_axis=True, x_offset_units=True, unit_colors=unit_colors, plot_legend=False, set_title=False)
# si.plot_unit_templates(analyzer, same_axis=True,
#                        x_offset_units=True, plot_legend=True, set_title=False)
# %%
# analyzer2.compute("random_spikes")
# analyzer2.compute("waveforms")
# analyzer2.compute("templates")
# si.plot_unit_templates(analyzer2, same_axis=True,
#                        x_offset_units=True, plot_legend=True, set_title=False)
