from pathlib import Path
import os
import h5py
import numpy as np

from util.load_data import load_data
import util.file_util as file_util
import spikeinterface.full as si


# %%
data_folders = file_util.file_dialog("C://data")
sorted_data_folders = file_util.sort_data_folders(data_folders)

for data_folder in sorted_data_folders:
    animalID = file_util.get_animal_id(data_folder)
    save_folder = Path(data_folder) / "batch_sort/merge"
    analyzer = si.load_sorting_analyzer(folder=save_folder / "hmerge_analyzer.zarr")

    extensions_to_compute = [
        "random_spikes",
        "spike_amplitudes",
        "templates",
        "correlograms",
        "template_similarity",
        "unit_locations",
    ]

    analyzer.compute(extensions_to_compute)

    si.plot_sorting_summary(analyzer, curation=True, backend="sortingview")
    uri = input("\nPlease enter uri: ")
    curation_sorting = si.apply_sortingview_curation(analyzer.sorting, uri_or_json=uri)

    keep_idx = np.where(curation_sorting.get_property("accept"))[0]
    unit_ids = curation_sorting.get_unit_ids()
    good_ids = unit_ids[keep_idx]

    # rename units so they are ordered from id 0 to N
    good_sorting = curation_sorting.select_units(unit_ids=good_ids, renamed_unit_ids=np.arange(len(good_ids)))

    curated_analyzer = si.create_sorting_analyzer(
        sorting=good_sorting,
        recording=analyzer.recording,
        format="zarr",
        folder=Path(save_folder) / "hmerge_analyzer_curated.zarr",
        sparse=False,
        overwrite=True,
    )
