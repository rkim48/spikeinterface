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

si.plot_unit_templates()


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

        for unit_id in unit_ids:
            # may be unnecessary since checked in stage1?
            if exclude_artifact_unit(unit_id, we):
                print(f"Unit {unit_id} is an artifact unit.")
                cs.sorting.set_property(
                    "artifact", values=[1], ids=[unit_id], missing_value=0
                )
                continue
            print(f"Curating waveforms for unit {unit_id}...")
            good_idx, bad_idx = remove_bad_waveforms_A(we, unit_id)


if __name__ == "__main__":
    main(debug_folder='C:\\data\\ICMS93\\behavior\\30-Aug-2023')

# %%
analyzer.compute("random_spikes")
analyzer.compute("waveforms")
analyzer.compute("templates")
si.plot_unit_templates(analyzer, same_axis=True,
                       x_offset_units=True, plot_legend=True, set_title=False)
