from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle
from spikeinterface import full as si
from batch_process.util.curate_util import *
from batch_process.util.subcluster_util import *
from batch_process.util.misc import *
from batch_process.util.plotting import *
from spikeinterface.curation import CurationSorting

# %%
"""
Merge test
"""

data_folder = "C:\\data\\ICMS93\\behavior\\06-Oct-2023"
save_folder = Path(data_folder) / "batch_sort"
we = si.load_waveforms(folder=save_folder / "waveforms_3", with_recording=True)
sorting = si.load_extractor(save_folder / "sorting_3")
property_keys = sorting.get_property_keys()

with open(save_folder / "cluster_assignments.pkl", "rb") as file:
    cluster_assignments = pickle.load(file)

keep_mask = np.where(sorting.get_property("accept") == 1)[0]
keep_units = sorting.unit_ids[keep_mask]

template_ch_dict = get_template_ch(we)

# %% Test merge first with new sorting and waveform extractor objects

cs = CurationSorting(parent_sorting=sorting, make_graph=True)
unit_ids = we.unit_ids
unit_id_properties = get_unit_id_properties(sorting)
property_keys = sorting.get_property_keys()
parent_ids = [key for key in property_keys if "parent" in key]
parent_ids = sorted(parent_ids, key=lambda x: int(x.split("parent_id")[1]))

for parent_id in parent_ids:
    parent_property = sorting.get_property(parent_id)
    accept_property = sorting.get_property("accept")

    # Identify units to remove: those with parent property but not accepted
    units_to_remove = unit_ids[(parent_property == 1) & (accept_property != 1)]
    cs.remove_units(units_to_remove)

    # Identify units to merge: those with parent property and accepted
    units_to_merge = unit_ids[(parent_property == 1) & (accept_property == 1)]
    if len(units_to_merge) > 0:
        cs.merge(units_to_merge)

cs.draw_graph()

# cs.sorting.save_to_folder(
#     save_folder / 'sorting_3', overwrite=True)

we_merge1 = si.extract_waveforms(
    we.recording,
    cs.sorting,
    folder=None,
    mode="memory",
    ms_before=1.0,
    ms_after=2.0,
    sparse=True,
    max_spikes_per_unit=100,
)

# %% Test auto merge


# %%


def plot_amplitude_vs_time(parent_id_dict, parent_id, sample_rate=30000, ax=None):
    """
    Plot a scatter plot of amplitude vs. time for all child units under a given parent ID.

    Parameters:
    parent_id_dict : dict
        Dictionary containing parent ID data.
    parent_id : str
        The parent ID to plot data for.
    sample_rate : int
        The sampling rate of the spike times.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        The axis to plot on. If None, a new figure and axis will be created.
    """
    # Extract data for the given parent ID
    child_unit_spike_times = parent_id_dict[parent_id]["spike_times"]
    child_unit_amplitudes = parent_id_dict[parent_id]["amplitudes"]

    if len(child_unit_spike_times) == 0:
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f"Amplitude vs. Time for Parent ID: {parent_id}")

    # Plot each child unit's data
    for i, (spike_times, amplitudes) in enumerate(
        zip(child_unit_spike_times, child_unit_amplitudes)
    ):
        spike_times_sec = spike_times / sample_rate  # Convert spike times to seconds
        ax.scatter(
            spike_times_sec, amplitudes, s=1, label=f"Child Unit {i}", color=f"C{i}"
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    ax.legend()

    if ax is None:
        plt.show()


unit_ids = we.unit_ids
unit_id_properties = get_unit_id_properties(sorting)

parent_ids = [key for key in property_keys if "parent" in key]
parent_ids = sorted(parent_ids, key=lambda x: int(x.split("parent_id")[1]))

parent_id_dict = {}
for parent_id in parent_ids:
    child_unit_mean_wvfs = []
    child_unit_spike_times = []
    child_unit_amplitudes = []

    parent_property = sorting.get_property(parent_id)
    accept_property = sorting.get_property("accept")
    mask = (parent_property == 1) & (accept_property == 1)
    child_ids = unit_ids[mask == 1]

    for child_id in child_ids:
        properties = unit_id_properties[child_id]
        child_wvfs = we.get_waveforms(child_id)
        dense_idx = template_ch_dict[child_id]["primary_ch_idx_dense"]

        mean_wvf = np.mean(child_wvfs[:, :, dense_idx], axis=0)

        amplitudes = child_wvfs[:, 30, dense_idx]
        spike_times = we.sorting.get_unit_spike_train(unit_id=child_id)

        child_unit_mean_wvfs.append(mean_wvf)
        child_unit_spike_times.append(spike_times)
        child_unit_amplitudes.append(amplitudes)

    parent_id_dict[parent_id] = {
        "mean_waveforms": child_unit_mean_wvfs,
        "amplitudes": child_unit_amplitudes,
        "spike_times": child_unit_spike_times,
    }


for parent_id in parent_ids:
    if len(parent_id_dict[parent_id]["spike_times"]) > 0:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        child_mean_wvfs = np.array(parent_id_dict[parent_id]["mean_waveforms"])
        plot_amplitude_vs_time(parent_id_dict, parent_id, ax=axes[0])
        axes[1].plot(child_mean_wvfs.T)
    plt.suptitle(parent_id)


# %% Subcluster similarity check
