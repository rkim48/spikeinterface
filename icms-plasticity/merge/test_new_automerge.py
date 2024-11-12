# Test automerge functions
import spikeinterface.curation.auto_merge as auto_merge
from spikeinterface.core import SortingAnalyzer
from pathlib import Path
import spikeinterface.full as si


def main(sorting_analyzer: SortingAnalyzer):
    sorting = sorting_analyzer.sorting
    unit_ids = sorting.unit_ids

    unit1 = unit_ids[15]
    unit2 = unit_ids[19]

    unit_colors = {unit1: "k", unit2: "r"}
    si.plot_amplitudes(sorting_analyzer, [unit1, unit2], unit_colors=unit_colors)

    d = auto_merge.presence_distance(sorting, unit1, unit2)

    print(unit1)
    print(unit2)
    print(d)


if __name__ == "__main__":
    # load example units

    data_path = Path("C:/data/ICMS93/behavior/30-Aug-2023/batch_sort/stage2/stage2_analyzer.zarr")
    analyzer = si.load_sorting_analyzer(data_path)
    analyzer.sorting.register_recording(analyzer.recording)

    # analyzer.compute("templates")
    # analyzer.compute("spike_amplitudes")

    main(analyzer)
