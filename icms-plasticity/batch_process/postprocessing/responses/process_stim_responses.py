import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
from pathlib import Path
import batch_process.util.plotting as plot_util
import merge.classify_cell_type as cell_classifier
import os
import batch_process.util.file_util as file_util
import batch_process.util.template_util as template_util

import dill as pickle
import shutil
import re
from collections import defaultdict
from batch_process.util.ppt_image_inserter import PPTImageInserter
import util.file_util as file_util
import batch_process.postprocessing.responses.response_plotting_util as rpu


def process_unit_stim_responses(session_responses, plot_flag=False):
    tparams = session_responses.timing_params
    pre_stim_threshold = 0.5  # Pre-stimulus firing rate threshold to be included
    results = []  # To store results for saving as CSV

    for unit_index, unit_id in enumerate(session_responses.unit_ids):
        print(f"Processing unit id {unit_id}...")
        unit_response = session_responses.get_unit_response(unit_id)
        stim_channels = sorted(
            {ch for ch, _ in unit_response.stim_responses.keys()})

        for stim_channel in stim_channels:
            if stim_channel == 0:
                continue  # Skip stim_channel 0

            print(f"Processing stim_channel {stim_channel}...")
            currents = sorted(
                [curr for ch, curr in unit_response.stim_responses.keys() if ch == stim_channel])
            n_currents = len(currents)

            # Initialize figure and grid for plots if plotting is enabled
            fig, axs = rpu.initialize_plotting(plot_flag, n_currents)

            # Process responses for each current
            process_currents(unit_response, stim_channel, currents, results,
                             plot_flag, axs, fig)

    # Save all results to CSV
    rpu.save_results_to_csv(results, session_responses.sorting_analyzer.folder)


def process_currents(unit_response, stim_channel, currents, results, plot_flag, axs, fig, pre_stim_threshold=0.5):
    color_map = rpu.get_stim_colormap()
    z_scores, latencies, jitters, pulse_locked_list, pre_stim_fr_list = [], [], [], [], []

    for i, stim_current in enumerate(currents):
        stim_response = unit_response.get_stim_response(
            stim_channel=stim_channel, stim_current=stim_current)
        train_response, pulse_response = stim_response.train_response, stim_response.pulse_response

        t_test_dict = train_response.t_test_dict
        z_score_dict = train_response.z_score_dict

        z_scores.append(z_score_dict['z_score'])
        latencies.append(pulse_response.stim_metrics["mean_latency"])
        jitters.append(pulse_response.stim_metrics["std_latency"])
        pulse_locked_list.append(pulse_response.is_pulse_locked)
        pre_stim_fr_list.append(train_response.pre_stim_mean_fr)

    # Skip if no significant modulation or low baseline firing rate
    if should_skip_unit(z_scores, pre_stim_fr_list, pre_stim_threshold):
        print(f"\tSkipping Unit {
              unit_response.unit_id} - Stim Channel D{stim_channel} due to low activity or no modulation.")
        if plot_flag:
            plt.close(fig)
        store_result(stim_response, results, include=False)
        return
    # Process and plot each current

    pulse_line_offset, train_line_offset = 0, 0
    for i, stim_current in enumerate(currents):
        if stim_current == 0:
            continue

        stim_response = unit_response.get_stim_response(
            stim_channel=stim_channel, stim_current=stim_current)

        train_response, pulse_response = stim_response.train_response, stim_response.pulse_response

        process_single_current(stim_response, results, plot_flag,
                               axs, fig, i, pulse_line_offset, train_line_offset)

        pulse_line_offset += len(pulse_response.raster_array)
        train_line_offset += len(train_response.raster_array)

    if plot_flag:
        axs["z_score"].set_xlabel("Current (µA)")
        axs["z_score"].set_xticks(currents)
        axs["z_score"].set_xticklabels([str(curr) for curr in currents])
        axs["z_score"].set_ylabel("Z-Score")


def process_single_current(stim_response, results, plot_flag, axs, fig, i, pulse_line_offset, train_line_offset, pre_stim_threshold=0.5):
    # Store result for this unit and current
    linewidth_factor = 20
    color_map = rpu.get_stim_colormap()
    pulse_response, train_response = stim_response.pulse_response, stim_response.train_response

    store_result(stim_response, results)

    stim_current = stim_response.stim_current
    stim_channel = stim_response.stim_channel

    # Plot data if needed
    if plot_flag:
        rpu.plot_pulse_raster(axs["pulse_raster"], pulse_response,
                              color_map[stim_current], linewidth_factor, pulse_line_offset)

        rpu.plot_train_raster(
            axs["train_raster"], train_response, color_map[stim_current], linewidth_factor, train_line_offset)
        rpu.plot_pulse_firing_rate(
            axs["pulse_fr"], pulse_response, color_map[stim_current])
        rpu.plot_train_firing_rate(
            axs["train_fr"], train_response, color_map[stim_current])
        rpu.plot_probability(
            axs["spike_prob"], pulse_response, color_map[stim_current])

        if i == 0:
            rpu.plot_template(stim_response,
                              stim_response.unit_id, ax=axs["template"])

        # Set alpha based on threshold
        alpha_value = 0.3 if train_response.pre_stim_mean_fr < pre_stim_threshold else 1.0
        axs["z_score"].bar(
            stim_current, stim_response.train_response.z_score, color=color_map[stim_current], alpha=alpha_value)

        alpha_value = 0.3 if train_response.paired_p_val > 0.05 else 1.0
        axs["t"].bar(
            stim_current, stim_response.train_response.paired_t_val, color=color_map[stim_current], alpha=alpha_value)

        finalize_plot(fig, stim_response)


def store_result(stim_response, results, include=True, pre_stim_threshold=0.5):
    pulse_response, train_response = stim_response.pulse_response, stim_response.train_response

    animal_id = file_util.get_animal_id(stim_response.session.session_path)

    result = {
        "animal_id": animal_id,
        "unit_id": stim_response.unit_id,
        "stim_channel": stim_response.stim_channel,
        "stim_current": stim_response.stim_current,
        "unit_location": stim_response.unit_response.unit_location,
        "cell_type": stim_response.unit_response.cell_type,
        "include": include,
    }
    if pulse_response and train_response:
        result.update({
            "spike_prob": np.max(pulse_response.stim_metrics["mean_prob_spike"]),
            "is_pulse_locked": pulse_response.is_pulse_locked,
            "pli": pulse_response.pli,
            "latency": pulse_response.stim_metrics["mean_latency"],
            "jitter": pulse_response.stim_metrics["std_latency"],
            "t_val": train_response.paired_t_val,
            "modulated": train_response.paired_p_val < 0.05,
            "z_score": train_response.z_score,
            "pulse_mean_fr": np.mean(pulse_response.firing_rate),
            "train_mean_fr": train_response.stim_mean_fr,
            "pre_stim_mean_fr": train_response.pre_stim_mean_fr,
            "fr_times":  train_response.fr_times,
            "firing_rate":  train_response.firing_rate,
            "template": stim_response.unit_response.primary_channel_template,
            "num_spikes": len(pulse_response.rel_spike_timestamps),
            "baseline_too_slow": train_response.pre_stim_mean_fr < pre_stim_threshold,
        })
    results.append(result)


def should_skip_unit(z_scores, pre_stim_fr_list, pre_stim_threshold):
    return (all(np.isnan(z) or z == 0 for z in z_scores) or
            all(fr < pre_stim_threshold for fr in pre_stim_fr_list))


def finalize_plot(fig, stim_response):
    unit_id = stim_response.unit_id
    stim_channel = stim_response.stim_channel
    analyzer = stim_response.session.sorting_analyzer
    plt.suptitle(f"Unit {unit_id} - Stim Channel D{stim_channel}")
    plt.tight_layout()
    plt.show()
    # plt.pause(0.1)

    # Save the figure if needed
    rpu.save_plot_and_create_dir(
        fig, analyzer.folder, unit_id, stim_channel)

# %%


# %%
if __name__ == "__main__":
    plt.style.use("default")
    data_folders = file_util.file_dialog("C://data")
    # data_folders = ["C:\\data\\ICMS93\\Behavior\\30-Aug-2023"]

    sorted_data_folders = file_util.sort_data_folders(data_folders)

    for data_folder in sorted_data_folders:

        figures_dir = Path(data_folder) / "batch_sort" / \
            "figures" / "stim_condition_figures"

        pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"
        with open(pkl_path, "rb") as file:
            session_responses = pickle.load(file)

        process_unit_stim_responses(session_responses, plot_flag=False)
        # rpu.make_ppt(data_folder)
