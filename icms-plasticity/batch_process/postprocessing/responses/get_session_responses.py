from spikeinterface import full as si
import itertools
import numpy as np
import dill as pickle
import math
import pandas as pd
import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from batch_process.postprocessing.responses import *

import batch_process.postprocessing.stim_response_util as stim_response_util
import batch_process.util.file_util as file_util
import batch_process.util.template_util as template_util

#
"""
Create SessionResponses object and save to disk
"""


def parse_date_from_path(path):
    date_str = path.split("\\")[-1]  # Extract the date part of the string
    return datetime.datetime.strptime(date_str, "%d-%b-%Y")  # Parse the date


def process_unit(unit_id, unique_stim_params, stim_data, timing_params, unit_response):
    """
    Process a single unit's response to the stimulus.

    Args:
        unit_id (int): Unit identifier.
        spike_times (np.ndarray): Array of spike times for the unit.
        unique_stim_params (list): List of unique stimulus parameters (depth, current).
        stim_data (StimData): Data structure containing stimulus times, currents, and depths.
        timing_params (dict): Timing parameters for processing.
        unit_response (UnitResponse): UnitResponse object to store the results.
    """
    for stim_params in unique_stim_params:
        depth, current = stim_params
        stim_timestamps = stim_response_util.get_stim_ts_indices(
            stim_data.timestamps, stim_data.currents, stim_data.depths, current, depth
        )

        if len(stim_timestamps) > 0:
            stim_condition_response = StimConditionResponse(
                stim_timestamps, depth, current, unit_response)

            # Process pulse and train responses
            pulse_response = PulseResponse(
                timing_params, stim_condition_response)
            train_response = TrainResponse(
                timing_params, stim_condition_response)


def save_session_responses(session_responses, save_folder):
    """
    Save the session responses to a file.

    Args:
        session_responses (SessionResponses): SessionResponses object containing the analysis results.
        save_folder (Path): Path to the folder where the results will be saved.
    """
    save_path = Path(save_folder) / "session_responses.pkl"
    with open(save_path, "wb") as file:
        pickle.dump(session_responses, file)
    print(f"Session responses saved to {save_path}")


def main(data_folder=None):
    job_kwargs = dict(n_jobs=5, chunk_duration="1s", progress_bar=True)

    if data_folder:
        sorted_data_folders = [data_folder]
    else:
        data_folders = file_util.file_dialog("C://data")
        sorted_data_folders = file_util.sort_data_folders(data_folders)

    for data_folder in sorted_data_folders:
        timing_params = file_util.load_timing_params(
            path="batch_process/postprocessing/timing_params.json")
        save_folder = Path(data_folder) / "batch_sort"
        date_str = file_util.get_date_str(data_folder)

        # TODO make sure loading appropriate analyzer
        analyzer = si.load_sorting_analyzer(
            Path(save_folder) / "merge/hmerge_analyzer_curated.zarr")
        # analyzer = si.load_sorting_analyzer(Path(save_folder) / "stage1/stage1_analyzer_curated.zarr")
        stim_data = stim_response_util.get_stim_data(data_folder)
        if stim_data is None:
            print(f"Skipping session {data_folder}...")
            continue

        # Extract unique stim channel depths and currents
        unique_depths = list(set(stim_data.depths))
        unique_currents = list(set(stim_data.currents))
        unique_stim_params = list(
            itertools.product(unique_depths, unique_currents))

        # Retrieve unit IDs and spikes
        unit_ids = analyzer.unit_ids

        extensions_to_compute = [
            "random_spikes",
            "waveforms",
            "templates",
            "correlograms",
            "unit_locations"
        ]

        extension_params = {"random_spikes": {
            "method": "all"}, "correlograms": {"window_ms": 100, "bin_ms": 0.5}}

        if not analyzer.has_extension("unit_locations"):
            analyzer.compute(extensions_to_compute,
                             extension_params=extension_params, **job_kwargs)

        # Initialize session response object
        session_responses = SessionResponses(
            data_folder, analyzer, stim_data.timestamps, timing_params)

        for unit_id in unit_ids:
            unit_response = UnitResponse(unit_id, session_responses)
            print(f"Processing unit id: {unit_id}")
            process_unit(unit_id, unique_stim_params, stim_data,
                         timing_params, unit_response)

            # plt.figure()
            # template_util.plot_mean_waveform_with_std_shading(
            #     unit_response.non_stim_waveforms, 'C0')
            # template_util.plot_mean_waveform_with_std_shading(
            #     unit_response.stim_waveforms, 'C1')
            # plt.title(f'Unit id: {unit_id}')

        save_session_responses(session_responses, save_folder)


# %%
# cell_classifier.classify_units_into_cell_types(
#     session_responses.sorting_analyzer)["df"]


# %%
if __name__ == "__main__":

    data_folder = "C:\\data\\ICMS92\\Behavior\\30-Aug-2023"
    data_folder = None
    main(data_folder=data_folder)

# %%
# pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"
# with open(pkl_path, "rb") as file:
#     session_responses = pickle.load(file)

# for unit_id in session_responses.unit_ids:
#     # for unit_id in range(11, 12):
#     ur = session_responses.get_unit_response(unit_id)

#     stim_conditions = ur.show_stim_conditions()

#     # Determine the number of rows and columns for the subplots
#     n_conditions = len(stim_conditions)
#     n_cols = 3  # Define how many columns you want
#     n_rows = int(np.ceil(n_conditions / n_cols))  # Calculate the required rows

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
#     axes = axes.flatten()  # Flatten to easily iterate over the subplots

#     for i, stim_condition in enumerate(stim_conditions):
#         scr = ur.get_stim_response(stim_condition[0], stim_condition[1])

#         ax = axes[i]
#         template_util.plot_mean_waveform_with_std_shading(
#             ur.non_stim_waveforms, color='C0', ax=ax)
#         if len(scr.stim_waveforms) > 0:
#             template_util.plot_mean_waveform_with_std_shading(
#                 scr.stim_waveforms, color='C1', ax=ax)

#         ax.set_title(f'Ch.{stim_condition[0]} at {stim_condition[1]}uA with {
#             len(scr.stim_waveforms)} evoked spikes')

#     # Hide any unused subplots
#     for j in range(i+1, len(axes)):
#         fig.delaxes(axes[j])

#     plt.suptitle(f'Unit: {unit_id}')
#     plt.tight_layout()
#     plt.show()


# %%
# plt.figure()
# template_util.plot_mean_waveform_with_std_shading(ur.non_stim_waveforms, 'C0')
# template_util.plot_mean_waveform_with_std_shading(ur.stim_waveforms, 'C1')
# plt.title(f'Unit id: {unit_id}')

# # %%

# %% DEBUG
data_folder = "C:\\data\\ICMS92\Behavior\\14-Sep-2023"
pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"
with open(pkl_path, "rb") as file:
    session_responses = pickle.load(file)

ur = session_responses.get_unit_response(0)
scr = ur.get_stim_response(9, 3)
tr = scr.train_response
pr = scr.pulse_response

plt.plot(tr.fr_times, tr.firing_rate)
plt.figure()
plt.plot(pr.fr_times, pr.firing_rate)

plt.figure()
plt.plot(ur.primary_channel_template)
plt.plot(ur.non_stim_template)
