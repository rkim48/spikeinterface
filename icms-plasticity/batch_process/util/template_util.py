import spikeinterface.full as si
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt


def save_sparse_analyzer(analyzer, method="memory", radius_um=60, job_kwargs={}):
    # save sparse analyzer to disk to avoid recomputing extensions
    original_path = analyzer.folder
    stem = original_path.stem
    suffix = original_path.suffix
    new_filename = f"{stem}_sparse{suffix}"
    new_path = original_path.with_name(new_filename)

    if os.path.isdir(new_path):
        shutil.rmtree(new_path)
    sparse_analyzer = analyzer.copy()
    # sparsify
    sparse_analyzer.sparsity = si.compute_sparsity(
        analyzer, method="radius", radius_um=radius_um)

    if method == "memory":
        sparse_analyzer.save_as()
    else:
        sparse_analyzer.save_as(format=method, folder=new_path)
        sparse_analyzer.folder = new_path

    extensions_to_compute = [
        "random_spikes",
        "waveforms",
        "templates",
        "template_similarity",
        "correlograms",
        "spike_amplitudes",
        "unit_locations",
    ]

    extension_params = {"random_spikes": {"method": "all"},
                        "unit_locations": {"method": "center_of_mass"}}
    sparse_analyzer.compute(extensions_to_compute,
                            extension_params=extension_params, **job_kwargs)

    return sparse_analyzer


def get_dense_primary_ch_index(analyzer, unit_id):
    template_extremum_ch = si.get_template_extremum_channel(
        analyzer, outputs="id")
    all_ch_ids = analyzer.channel_ids
    primary_ch = template_extremum_ch[unit_id]
    return np.where(all_ch_ids == str(primary_ch))[0][0]


def plot_mean_waveform_with_std_shading(waveforms, color='k', ax=None):
    # Calculate mean and standard deviation
    mean_template = np.mean(waveforms, axis=0)
    std_template = np.std(waveforms, axis=0)

    # Generate x-values (assuming sampling rate of 30 Hz)
    x_values = np.arange(mean_template.shape[0]) / 30  # Adjust as needed

    # Use the provided axis, or get the current one if None is given
    if ax is None:
        ax = plt.gca()

    # Plot the mean waveform
    ax.plot(x_values, mean_template, label='Mean', color=color)

    # Plot the shaded standard deviation
    ax.fill_between(x_values, mean_template - std_template, mean_template + std_template,
                    color=color, alpha=0.2)

    # Optionally, you can customize other aspects of the plot here
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')


def get_template_ch(analyzer, all_wvfs=True, job_kwargs={}):
    original_path = analyzer.folder
    stem = original_path.stem
    suffix = original_path.suffix
    new_filename = f"{stem}_sparse{suffix}"
    new_path = original_path.with_name(new_filename)

    try:
        sparse_analyzer = si.load_sorting_analyzer(new_path)
    except AssertionError:
        print(f"Sparse analyzer not found at {
              new_path}. Create a sparse analyzer using template.util.save_sparse_analyzer_to_disk().")
        # sparse_analyzer = save_sparse_analyzer(analyzer)

    if sparse_analyzer.sparsity is None:
        print("Error! sparse_analyzer is not actually sparse.")
        return None

    all_ch_ids = sparse_analyzer.channel_ids
    unit_ids = sparse_analyzer.unit_ids

    unit_id_to_ch_ids = sparse_analyzer.sparsity.unit_id_to_channel_ids
    template_extremum_ch = si.get_template_extremum_channel(
        sparse_analyzer, outputs="id")

    template_ch_dict = {}
    for unit_id in unit_ids:
        ch_ids = unit_id_to_ch_ids[unit_id]
        primary_ch = template_extremum_ch[unit_id]
        template_ch_dict[unit_id] = {
            "ch_ids": ch_ids,
            "primary_ch": primary_ch,
            "primary_ch_idx_sparse": np.where(ch_ids == str(primary_ch))[0][0],
            "primary_ch_idx_dense": np.where(all_ch_ids == str(primary_ch))[0][0],
        }
    return template_ch_dict


def get_unit_primary_ch_templates(analyzer):
    # dense analyzer
    if analyzer.sparsity is not None:
        print("Error! sparse_analyzer is not dense.")
        return None

    templates_ext = analyzer.get_extension("templates")

    template_extremum_indices = si.get_template_extremum_channel(
        analyzer, outputs="index")

    unit_primary_ch_wvfs_dict = {}
    for unit_id in analyzer.unit_ids:
        primary_ch_idx = template_extremum_indices[unit_id]
        primary_ch_wvf = templates_ext.get_unit_template(unit_id)[
            :, primary_ch_idx]
        unit_primary_ch_wvfs_dict[unit_id] = primary_ch_wvf
    return unit_primary_ch_wvfs_dict


def get_unit_primary_ch_wvfs(analyzer, unit_id):
    # Return unit id primary channel waveforms using dense analyzer
    num_ch = analyzer.get_num_channels()

    template_ext = analyzer.get_extension("templates")
    template = template_ext.get_unit_template(unit_id)
    wvf_ext = analyzer.get_extension("waveforms")
    wvfs = wvf_ext.get_waveforms_one_unit(unit_id=unit_id)
    if wvfs.shape[2] == num_ch:
        # dense case
        template_extremum_indices = si.get_template_extremum_channel(
            analyzer, outputs="index")
        primary_ch_index = template_extremum_indices[unit_id]
        return wvfs[:, :, primary_ch_index]
    else:
        raise ValueError(
            "Analyzer must be dense. The waveforms do not cover all channels.")


def get_unit_primary_ch_template(analyzer, unit_id):
    if analyzer.sparsity is not None:
        print("Error! sparse_analyzer is not dense.")
        return None

    templates_ext = analyzer.get_extension("templates")
    template_extremum_indices = si.get_template_extremum_channel(
        analyzer, outputs="index")
    primary_ch_idx = template_extremum_indices[unit_id]
    primary_ch_wvf = templates_ext.get_unit_template(unit_id)[
        :, primary_ch_idx]

    return primary_ch_wvf

# %%


# wvfs = get_unit_primary_ch_wvfs(analyzer, unit_id=23)

# plt.plot(wvfs.T, 'k', linewidth=0.5, alpha=0.1)

# #%%

# # When loading analyzer from stage2, it is dense but should handl eiehter case
# # First handle dense case.

# # Template dense, waveform dense, ch idx from template ext ch function
# unit_id = 6

# wvf_ext = analyzer.get_extension("waveforms")
# wvfs = wvf_ext.get_waveforms_one_unit(unit_id=unit_id)[:, :, primary_ch_index]
# plt.plot(wvfs.T)


# #%%
# unit_id = analyzer.unit_ids[0]
# wvfs = get_unit_primary_ch_wvfs(analyzer, unit_id)

# plt.plot(wvfs)
