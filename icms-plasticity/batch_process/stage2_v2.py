import numpy as np
from spikeinterface import full as si
from util.file_util import *
# from curation_test.curation_module import *
from batch_process.util.curate_util import *
from batch_process.util.misc import *

# %%


def main(debug_folder):
    if debug_folder:
        data_folders = [debug_folder]
    else:
        starting_dir = "C:\\data"
        data_folders = file_dialog(starting_dir=starting_dir)

    job_kwargs = dict(n_jobs=5, chunk_duration="1s", progress_bar=True)
    for i, data_folder in enumerate(data_folders):
        print("\n###########################################")
        print(f"{data_folder}: {i+1}/{len(data_folders)}")
        print("###########################################")

        save_folder = Path(data_folder) / "batch_sort"
        analyzer = si.load_sorting_analyzer(
            folder=save_folder / "stage1/stage1_analyzer.zarr")

        analyzer.compute("random_spikes")
        analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)
        analyzer.compute("templates")
        analyzer.compute("template_similarity")
        analyzer.compute("correlograms")
        analyzer.compute("spike_amplitudes")
        analyzer.compute("unit_locations")

        si.plot_sorting_summary(analyzer, curation=True, backend="sortingview")
        uri = input("\nPlease enter uri: ")
        manual_curation_sorting = si.apply_sortingview_curation(
            analyzer.sorting, uri_or_json=uri)

        keep_idx = np.where(manual_curation_sorting.get_property("accept"))[0]
        unit_ids = manual_curation_sorting.get_unit_ids()
        good_ids = unit_ids[keep_idx]

        # rename units so they are ordered from id 0 to N
        good_sorting = analyzer.sorting.select_units(
            unit_ids=good_ids, renamed_unit_ids=np.arange(
                len(good_ids))
        )

        analyzer2 = si.create_sorting_analyzer(
            sorting=good_sorting,
            recording=analyzer.recording,
            format="zarr",
            folder=Path(save_folder) / "stage2/stage2_analyzer.zarr",
            sparse=False,
            overwrite=True,
        )

        # Debugging
        # analyzer.compute("random_spikes")
        # analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)
        # w = analyzer.get_extension("waveforms")
        # wvf = w.get_waveforms_one_unit(analyzer.unit_ids[10])
        # plt.plot(wvf[121,:])


if __name__ == "__main__":
    main(debug_folder='C:\\data\\ICMS93\\behavior\\30-Aug-2023')

# %%
# unit_colors = {unit_id: 'k' for unit_id in analyzer.unit_ids}
# si.plot_unit_templates(analyzer, same_axis=True, x_offset_units=True, unit_colors=unit_colors, plot_legend=False, set_title=False)
si.plot_unit_templates(analyzer, same_axis=True,
                       x_offset_units=True, plot_legend=True, set_title=False)
# %%
analyzer2.compute("random_spikes")
analyzer2.compute("waveforms")
analyzer2.compute("templates")
si.plot_unit_templates(analyzer2, same_axis=True,
                       x_offset_units=True, plot_legend=True, set_title=False)
