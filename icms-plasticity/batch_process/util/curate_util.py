from batch_process.util.ppt_image_inserter import PPTImageInserter
import numpy as np
from spikeinterface import full as si
from pathlib import Path
from spikeinterface.postprocessing import compute_correlograms

# from batch_process.util.plotting import *
from batch_process.util.file_util import *
from batch_process.util.plotting import plot_units_in_batches
import batch_process.util.template_util as template_util

# %%
# analyzer.sparsity = None
# analyzer.compute("templates")
# a = analyzer.get_extension("templates")
# plt.plot(a.get_unit_template(unit_id=2))


# TEST
# unit_id = 1
# analyzer.sparsity = None
# analyzer.compute("random_spikes")
# analyzer.compute("waveforms")
# analyzer.compute("templates")

# dense_analyzer = analyzer.copy()

# sparse_analyzer = analyzer.copy()
# sparse_analyzer.sparsity = si.compute_sparsity(
#     sparse_analyzer, method="radius", radius_um=60)
# sparse_analyzer.compute("random_spikes")
# sparse_analyzer.compute("waveforms")
# sparse_analyzer.compute("templates")
# sparse_templates = sparse_analyzer.get_extension("templates")

# plt.plot(sparse_templates.get_unit_template(unit_id=unit_id))

# plt.figure()
# plt.title("Dense template")
# dense_templates = dense_analyzer.get_extension("templates")

# plt.plot(dense_templates.get_unit_template(unit_id=unit_id))

# # Test extremum ch id
# template_extremum_ch = si.get_template_extremum_channel(
#     sparse_analyzer, outputs="id")  # 2
# unit_id_to_ch_ids = sparse_analyzer.sparsity.unit_id_to_channel_ids
# primary_ch = template_extremum_ch[unit_id]
# ch_ids = unit_id_to_ch_ids[unit_id]
# all_ch_ids = analyzer.channel_ids
# np.where(ch_ids == str(primary_ch))[0][0]
# np.where(all_ch_ids == str(primary_ch))[0][0]

# plt.plot(templates.get_unit_template(unit_id=unit_id))


def sparse_template_max_amplitude(template):
    max_amplitudes = np.min(template, axis=0)
    ts = np.argmin(template, axis=0)
    return max_amplitudes, ts


# TODO
def get_ts_offsets(sparse_analyzer):
    # analyzer sparsity needs to have radius of 60 um
    unit_ids = sparse_analyzer.unit_ids
    templates_ext = sparse_analyzer.get_extension("templates")

    ts_offsets = {}
    for unit_id in unit_ids:
        template = templates_ext.get_unit_template(unit_id)
        _, ts = sparse_template_max_amplitude(template)
        primary_ts = ts[0]  # Timestamp of the primary channel
        offsets = ts - primary_ts  # Calculate offsets relative to the primary channel
        ts_offsets[unit_id] = np.mean(np.abs(offsets[1:]))
    return ts_offsets


def classify_waveform(new_wvf):
    point1 = new_wvf[6]
    point2 = new_wvf[74]
    return -60 <= point1 <= 100 and -21 <= point2 <= 35


# TODO
def sort_units_by_primary_channel(analyzer, unit_ids_subset):
    ch_loc = analyzer.get_channel_locations()[:, 1]
    ch_ids = analyzer.channel_ids
    ch_idx = analyzer.channel_ids_to_indices(ch_ids)
    ch_idx_to_loc_dict = {idx: loc for idx, loc in zip(ch_idx, ch_loc)}
    unit_id_to_ch_idx_dict = si.get_template_extremum_channel(analyzer, outputs="index")
    # Get the extremum channel location for each unit in the subset
    unit_id_to_extremum_loc_dict = {
        unit_id: ch_idx_to_loc_dict[unit_id_to_ch_idx_dict[unit_id]] for unit_id in unit_ids_subset
    }

    # Sort the unit IDs based on the extremum channel location
    sorted_unit_ids = sorted(
        unit_id_to_extremum_loc_dict.keys(),
        key=lambda unit_id: unit_id_to_extremum_loc_dict[unit_id],
    )[::-1]

    return sorted_unit_ids


def mean_norm_peak(dense_template, sparse_ch_indices, min_idx=30, range_offset=10):
    max_peaks = np.max(
        np.abs(dense_template[min_idx - range_offset : min_idx + range_offset + 1, :]),
        axis=0,
    )
    all_ch_indices = np.arange(dense_template.shape[1])
    # Exclude sparse template channels
    non_sparse_ch_indices = np.setdiff1d(all_ch_indices, sparse_ch_indices)
    max_peaks_non_sparse = max_peaks[non_sparse_ch_indices]
    template_peak = np.min(dense_template[min_idx, :])
    normalized_peaks_non_sparse = max_peaks_non_sparse / np.abs(template_peak)
    top_peaks = np.sort(normalized_peaks_non_sparse)[-5:]
    mnr = np.mean(top_peaks)

    return mnr


def exclude_artifact_unit(unit_id, analyzer):
    # analyzer must be dense
    num_channels = analyzer.get_num_channels()
    sparsity = si.compute_sparsity(analyzer, method="radius", radius_um=120)
    unit_id_to_ch_ind = sparsity.unit_id_to_channel_indices
    sparse_ch_indices = unit_id_to_ch_ind[unit_id]
    templates = analyzer.get_extension("templates")
    dense_template = templates.get_unit_template(unit_id)
    assert dense_template.shape[1] == num_channels, f"Dense template ch num is not {num_channels}!"
    mnr = mean_norm_peak(dense_template, sparse_ch_indices)
    return mnr > 0.3


def curate_units(analyzer, sparse_analyzer, min_spikes=100, job_kwargs={}):
    """
    Curate units after sorting to remove stim/imaging artifacts.
    Look at:
        1) acg 30 ms component
        2) shape of waveform
        3) mean of normalized peaks far from primary channel
        4) spike count

    Returns good and bad unit ids
    """
    unit_ids = analyzer.unit_ids
    # ccgs, bins = analyzer.compute("correlograms", window_ms=100).get_data()

    ccgs, bins = analyzer.get_extension("correlograms").get_data()

    # ccgs, bins = compute_correlograms(we, window_ms=100)
    bins = bins[50:]  # get only positive values since symmetric
    ccgs = ccgs[:, :, 50:]
    acg_range = np.where((bins > 31) & (bins < 35))[0]
    ts_offset_threshold = 10  # samples, max mean sample offset from primary ch peak
    ts_offsets = get_ts_offsets(sparse_analyzer)

    good_ids1 = []
    artifacts = []

    for i, unit_id in enumerate(unit_ids):
        acg = ccgs[i, i, :]
        sort_idx = np.argsort(acg)[::-1]
        # Exclude artifact unit check
        if exclude_artifact_unit(unit_id, analyzer):
            artifacts.append(unit_id)
            continue

        if np.any(np.isin(sort_idx[:5], acg_range)) & (ts_offsets[unit_id] > ts_offset_threshold):
            artifacts.append(unit_id)
        else:
            good_ids1.append(unit_id)

    good_ids2 = []
    noise = []

    # classify by primary waveform
    unit_primary_ch_wvfs_dict = template_util.get_unit_primary_ch_templates(analyzer)
    for unit_id in good_ids1:
        template = unit_primary_ch_wvfs_dict[unit_id]
        if classify_waveform(template):
            good_ids2.append(unit_id)
        else:
            noise.append(unit_id)

    good_ids = sort_units_by_primary_channel(analyzer, good_ids2)
    bad_ids = sort_units_by_primary_channel(analyzer, artifacts + noise)

    return good_ids, bad_ids


def curate_and_plot(analyzer, parent_folder, subfolder_path, save_path=None, curate_only=False):
    sorting = analyzer.sorting

    good_ids, bad_ids = curate_units(analyzer)
    good_sorting = sorting.select_units(unit_ids=good_ids, renamed_unit_ids=good_ids)
    bad_sorting = sorting.select_units(unit_ids=bad_ids, renamed_unit_ids=bad_ids)

    if save_path:
        folder = Path(parent_folder) / save_path
        if folder.exists() and folder.is_dir():
            shutil.rmtree(folder)
        good_we = analyzer.select_units(unit_ids=good_ids, format="zarr", folder=folder)
    else:
        good_analyzer = analyzer.select_units(unit_ids=good_ids)
    bad_analyzer = analyzer.select_units(unit_ids=bad_ids)

    if not curate_only:
        # Check if subfolder_path is provided and non-empty
        if not subfolder_path:
            raise ValueError("subfolder_path cannot be empty.")

        results_path = Path(parent_folder) / subfolder_path
        if results_path.exists() and results_path.is_dir():
            if results_path == Path(parent_folder):
                raise ValueError("Cannot delete the entire parent folder. Please provide a valid subfolder path.")
            # Ensure only subfolder contents are deleted
            shutil.rmtree(results_path)
        results_path.mkdir(parents=True, exist_ok=True)

        plot_units_in_batches(good_analyzer, save_dir=results_path, ppt_name="curated")

    return bad_analyzer, bad_ids, good_analyzer


# TODO
def remove_bad_waveforms_A(analyzer, unit_id, plot_flag=False):
    """
    To remove waveforms with a string of 3 zeros in specific windows (24-30 and 32-45),
    waveforms with values above -40 µV at indices 29, 30, 31, and waveforms
    whose max amplitude channel does not match the template max amplitude channel.
    """

    # Load waveforms and channel info
    wvf_ext = analyzer.get_extension("waveforms")
    wvfs = wvf_ext.get_waveforms_one_unit(unit_id=unit_id)
    num_wvfs = wvfs.shape[0]
    nbefore = wvf_ext.nbefore

    primary_wvfs = template_util.get_unit_primary_ch_wvfs(analyzer, unit_id)
    primary_ch_idx_dense = si.get_template_extremum_channel(analyzer, outputs="index")[unit_id]

    # Find waveforms whose max channel is template primary channel
    has_incorrect_primary_ch = np.zeros(num_wvfs, dtype=bool)
    for i in range(num_wvfs):
        wvf = wvfs[i]
        # Check if the max amplitude channel matches the template max amplitude channel
        max_idx = np.argmin(wvf[nbefore, :])
        if max_idx != primary_ch_idx_dense:
            has_incorrect_primary_ch[i] = True

    extremum_ch_inds = si.get_template_extremum_channel(analyzer, outputs="index")
    sv = analyzer.sorting.to_spike_vector(extremum_channel_inds=extremum_ch_inds)

    unit_index = np.where(analyzer.unit_ids == unit_id)[0][0]
    unit_spike_indices = np.where(sv["unit_index"] == unit_index)[0]
    unit_sv = sv[unit_spike_indices]

    # Define the peak and pre-peak range for the first window
    is_zero_24_30 = primary_wvfs[:, 24:31] == 0
    three_consecutive_zeros_24_30 = np.array(
        [np.any(np.convolve(row.astype(int), [1, 1, 1], mode="valid") == 3) for row in is_zero_24_30]
    )

    # Define the second window from 32 to 45
    is_zero_32_38 = primary_wvfs[:, 32:38] == 0
    three_consecutive_zeros_32_38 = np.array(
        [np.any(np.convolve(row.astype(int), [1, 1, 1], mode="valid") == 3) for row in is_zero_32_38]
    )

    # Check for waveforms with values above -40 µV at inices 29, 30, and 31
    high_amplitude_mask = np.any(primary_wvfs[:, 29:32] > -40, axis=1)

    # Combine masks
    # combined_mask = has_incorrect_primary_ch | three_consecutive_zeros_24_30 | three_consecutive_zeros_32_38 | high_amplitude_mask
    # incorrect_primary_ch_idx = np.where(has_incorrect_primary_ch)[0]
    combined_mask = three_consecutive_zeros_24_30 | three_consecutive_zeros_32_38 | high_amplitude_mask

    good_indices = np.where(~combined_mask)[0]
    bad_indices = np.where(combined_mask)[0]

    if plot_flag:
        validate_remove_bad_waveforms(analyzer, unit_id, primary_ch_idx_dense, good_indices, bad_indices)

    return good_indices, bad_indices


def validate_remove_bad_waveforms(analyzer, unit_id, primary_ch_idx_dense, good_indices, bad_indices):

    wvf_ext = analyzer.get_extension("waveforms")
    wvfs = wvf_ext.get_waveforms_one_unit(unit_id=unit_id)

    good_indices_trunc = good_indices[: np.min([len(good_indices), 500])]
    bad_indices_trunc = bad_indices[: np.min([len(bad_indices), 500])]

    templates = analyzer.get_extension("templates")
    template = templates.get_unit_template(unit_id)

    good_waveforms = wvfs[good_indices_trunc, :, primary_ch_idx_dense].T
    bad_waveforms = wvfs[bad_indices_trunc, :, primary_ch_idx_dense].T

    plt.plot(good_waveforms, "g", linewidth=0.5, alpha=0.2)
    plt.plot(bad_waveforms, "r", linewidth=0.5, alpha=0.2)


# %% Visualize

# ppt_inserter = PPTImageInserter(
#     grid_dims=(2, 2), spacing=(0.05, 0.05), title_font_size=16
# )

# analyzer.sparsity = None

# analyzer.compute("random_spikes", method="all")
# analyzer.compute("waveforms")
# analyzer.compute("templates")

# for unit_id in unit_ids:
#     remove_bad_waveforms_A(analyzer, unit_id, template_ch_dict, plot_flag=True)

#     img_path = "img.png"
#     plt.savefig(str(img_path))
#     ppt_inserter.add_image(str(img_path))
#     plt.close()

# ppt_inserter.save("test.pptx")
