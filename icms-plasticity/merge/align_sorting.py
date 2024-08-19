import numpy as np
from spikeinterface import full as si
from pathlib import Path


def wvf_acg_exporter(we, sorting, file_path):
    unit_dict = {}
    si.compute_correlograms(we)
    if sorting is not None:
        property_keys = sorting.get_property_keys()
        keep_mask = np.where(sorting.get_property("accept") == 1)[0]
        keep_units = sorting.unit_ids[keep_mask]
    else:
        keep_units = we.unit_ids
    we_new = we.select_units(keep_units)
    ccgs, time_bins = si.compute_correlograms(we_new, window_ms=100, bin_ms=0.5)

    for i, unit_id in enumerate(we_new.unit_ids):
        data_dict = {}
        data_dict["acg"] = ccgs[i, i, :]
        data_dict["template"] = template

        unit_dict[unit_id] = data_dict

    # for i, unit_id in enumerate(we_new.unit_ids):
    #     template = we_new.get_template(unit_id)
    #     plt.figure()
    #     plt.plot(template)
    #     plt.axvline(30)
    return unit_dict


# %% Align sorting
def main(we, sorting):
    job_kwargs = dict(n_jobs=5, chunk_duration="1s", progress_bar=True)
    unit_peaks_shift = {}
    for unit_id in we.unit_ids:
        template = we.get_template(unit_id)
        min_values_within_range = np.min(template[20:40, :], axis=0)
        global_min_index = np.argmin(min_values_within_range)
        primary_wvf = template[:, global_min_index]
        peak_idx = np.argmin(primary_wvf)
        shift = 30 - peak_idx
        unit_peaks_shift[unit_id] = shift

    align_sorting = si.align_sorting(sorting, unit_peaks_shift)

    we = si.extract_waveforms(
        we.recording,
        align_sorting,
        folder=None,
        mode="memory",
        ms_before=1.0,
        ms_after=2.0,
        sparse=False,
        overwrite=True,
        max_spikes_per_unit=None,
        **job_kwargs,
    )

    return we


if __name__ == "__main__":
    data_folder = "C:\\data\\ICMS93\\Behavior\\12-Sep-2023"
    save_folder = Path(data_folder) / "batch_sort"
    we = si.load_waveforms(folder=save_folder / "waveforms_3", with_recording=True)
    sorting = si.load_extractor(save_folder / "sorting_3")

    we_new = main(we, sorting)


# %%
data_folder = "C:\\data\\ICMS93\\Behavior\\12-Sep-2023"
save_folder = Path(data_folder) / "batch_sort"
we = si.load_waveforms(folder=save_folder / "waveforms_3", with_recording=True)
# %%
unit_peaks_shift = {}
for unit_id in we.unit_ids:
    template = we.get_template(unit_id)
    min_values_within_range = np.min(template[20:40, :], axis=0)
    global_min_index = np.argmin(min_values_within_range)
    primary_wvf = template[:, global_min_index]
    peak_idx = np.argmin(primary_wvf)

    plt.figure()
    template = we.get_template(unit_id)
    plt.plot(template)
    plt.plot(primary_wvf, linewidth=3)
    plt.axvline(30)

    shift = 30 - peak_idx
    unit_peaks_shift[unit_id] = shift

# %%
sorting = si.load_extractor(save_folder / "sorting_3")
unit_dict = {}

if sorting is not None:
    property_keys = sorting.get_property_keys()
    keep_mask = np.where(sorting.get_property("accept") == 1)[0]
    keep_units = sorting.unit_ids[keep_mask]
else:
    keep_units = we.unit_ids

we_new = we.select_units(keep_units)
si.compute_correlograms(we_new)
ccgs, time_bins = si.compute_correlograms(we_new, window_ms=100, bin_ms=0.5)
# %%
for i, unit_id in enumerate(we_new.unit_ids):
    data_dict = {}
    data_dict["acg"] = ccgs[i, i, :]
    data_dict["bins"] = time_bins
    data_dict["template"] = we_new.get_template(unit_id)
    unit_dict[unit_id] = data_dict

# %% Pickle

with open(save_folder / "cell_type_features.pkl", "wb") as file:
    pickle.dump(unit_dict, file)
