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

import dill as pickle
import shutil
import re
from collections import defaultdict
from batch_process.util.ppt_image_inserter import PPTImageInserter
import util.file_util as file_util


def process_unit_stim_responses(session_responses, plot_flag=True):
    linewidth_factor = 20
    tparams = session_responses.timing_params
    cell_type_df = cell_classifier.classify_units_into_cell_types(
        session_responses.sorting_analyzer)["df"]
    ul_ext = session_responses.sorting_analyzer.get_extension("unit_locations")
    pre_stim_threshold = 0.5  # Pre-stimulus firing rate threshold to be included
    results = []  # To store results for saving as CSV

    for unit_index, unit_id in enumerate(session_responses.unit_ids):
        print(f"Processing unit id {unit_id}...")
        unit_response = session_responses.get_unit_response(unit_id)
        stim_channels = sorted(
            {ch for ch, _ in unit_response.stim_responses.keys()})

        unit_location = ul_ext.get_data()[unit_index]
        cell_type = cell_type_df.loc[unit_id, "cell_type"]

        primary_ch_template = template_util.get_unit_primary_ch_template(
            session_responses.sorting_analyzer, unit_id)

        for stim_channel in stim_channels:
            print(f"Processing channel depth {stim_channel}...")
            if stim_channel == 0:
                continue  # Skip stim_depth 0
            currents = sorted(
                [curr for ch, curr in unit_response.stim_responses.keys() if ch == stim_channel])
            n_currents = len(currents)
            color_map = get_stim_colormap()

            # Initialize figure and GridSpec
            if plot_flag:
                fig = plt.figure(figsize=(8, 6))
                gs = fig.add_gridspec(3, 3, width_ratios=[
                                      2, 2, 1])  # 3 rows, 3 columns

                # Create subplots using the grid specification
                ax_pulse_raster = fig.add_subplot(gs[0, 0])
                ax_pulse_fr = fig.add_subplot(gs[1, 0])
                ax_spike_prob = fig.add_subplot(gs[2, 0])
                ax_latency = fig.add_subplot(gs[2, 1])
                ax_train_raster = fig.add_subplot(gs[0, 1])
                ax_train_fr = fig.add_subplot(gs[1, 1])

                # Use the third column for the template plot
                ax_template = fig.add_subplot(gs[:2, 2])
                ax_z_score = fig.add_subplot(gs[2, 2])

            z_scores = []
            latencies = []
            jitters = []
            pulse_locked_list = []
            pre_stim_fr_list = []
            for i, stim_current in enumerate(currents):
                stim_response = unit_response.get_stim_response(
                    stim_channel=stim_channel, stim_current=stim_current)
                train_response = stim_response.train_response
                pulse_response = stim_response.pulse_response

                z_scores.append(train_response.z_score)
                latencies.append(pulse_response.stim_metrics["mean_latency"])
                jitters.append(pulse_response.stim_metrics["std_latency"])
                pulse_locked_list.append(pulse_response.is_pulse_locked)
                pre_stim_fr_list.append(train_response.pre_stim_mean_fr)

            # Check if all Z-scores are 0 or NaN, or if all pre-stimulus firing rates are below the threshold
            if all(np.isnan(z) or z == 0 for z in z_scores) or all(fr < pre_stim_threshold for fr in pre_stim_fr_list):
                print(f"\tSkipping Unit {
                      unit_id} - Stim Channel D{stim_channel} due to low activity or no modulation.")
                if plot_flag:
                    plt.close()
                results.append(
                    {
                        "unit_id": unit_id,
                        "stim_channel": stim_channel,
                        "stim_current": stim_current,
                        "unit_location": unit_location,
                        "cell_type": cell_type,
                        "include": False,
                    }
                )
                continue

            pulse_line_offset = 0  # Initialize line offset for pulse raster
            train_line_offset = 0  # Initialize line offset for train raster

            for i, stim_current in enumerate(currents):
                if stim_current == 0:
                    if plot_flag:
                        plt.close(fig)
                    z_scores.append(np.nan)
                    continue
                stim_response = unit_response.get_stim_response(
                    stim_channel=stim_channel, stim_current=stim_current)
                pulse_response = stim_response.pulse_response
                train_response = stim_response.train_response

                baseline_metrics = pulse_response.baseline_metrics
                stim_metrics = pulse_response.stim_metrics
                spike_prob = np.max(stim_metrics["mean_prob_spike"])
                latency = stim_metrics["mean_latency"]
                jitter = stim_metrics["std_latency"]

                results.append(
                    {
                        "unit_id": unit_id,
                        "stim_channel": stim_channel,
                        "stim_current": stim_current,
                        "unit_location": unit_location,
                        "cell_type": cell_type,
                        "spike_prob": spike_prob,
                        "is_pulse_locked": pulse_response.is_pulse_locked,
                        "pli": pulse_response.pli,
                        "latency": latency,
                        "jitter": jitter,
                        "z_score": train_response.z_score,



                        "pulse_mean_fr": np.mean(pulse_response.firing_rate),
                        "train_mean_fr": train_response.stim_mean_fr,
                        "pre_stim_mean_fr": train_response.pre_stim_mean_fr,
                        "fr_times": train_response.fr_times,
                        "firing_rate": train_response.firing_rate,
                        "num_spikes": len(pulse_response.rel_spike_timestamps),
                        "template": primary_ch_template,
                        "baseline_too_slow": True if train_response.pre_stim_mean_fr < pre_stim_threshold else False,
                    }
                )

                # Plot raster for pulse response
                pulse_raster_array = pulse_response.raster_array
                num_trials = len(pulse_raster_array)
                if num_trials == 0:
                    pulse_good_linelength = 1
                    pulse_lineoffsets = pulse_line_offset
                else:
                    pulse_good_linelength = np.ceil(
                        len(pulse_raster_array) / linewidth_factor)
                    pulse_lineoffsets = (
                        np.arange(len(pulse_raster_array)) + pulse_line_offset
                    )  # Stack rasters vertically

                if plot_flag:
                    ax_pulse_raster.eventplot(
                        pulse_raster_array,
                        orientation="horizontal",
                        linewidths=1,
                        linelengths=pulse_good_linelength,
                        lineoffsets=pulse_lineoffsets,
                        colors=[color_map[stim_current]],
                    )

                    # Update the pulse line offset for the next current
                    pulse_line_offset += len(pulse_raster_array)

                    ax_pulse_raster.axvspan(
                        0, tparams["post_stim_blank_ms"], color="lightgray", alpha=0.5)
                    ax_pulse_raster.axvspan(
                        tparams["pulse_win_ms"] - tparams["pre_stim_blank_ms"],
                        tparams["pulse_win_ms"],
                        color="lightgray",
                        alpha=0.5,
                    )
                    ax_pulse_raster.set_xlim([0, tparams["pulse_win_ms"]])

                    # Plot train response raster
                    train_raster_array = train_response.raster_array
                    train_good_linelength = np.ceil(
                        len(train_raster_array) / linewidth_factor)
                    train_lineoffsets = (
                        np.arange(len(train_raster_array)) + train_line_offset
                    )  # Stack rasters vertically

                    ax_train_raster.eventplot(
                        train_raster_array,
                        orientation="horizontal",
                        colors=[color_map[stim_current]],
                        linelengths=train_good_linelength,
                        linewidths=1,
                        lineoffsets=train_lineoffsets,
                    )
                    # Update the train line offset for the next current
                    train_line_offset += len(train_raster_array)

                    ax_train_raster.set_xlim([-700, 1400])
                    ax_train_raster.axvline(
                        x=0, color="gray", linestyle="--", linewidth=1)
                    ax_train_raster.axvline(
                        x=700, color="gray", linestyle="--", linewidth=1)

                    ax_pulse_fr.plot(
                        pulse_response.fr_times,
                        pulse_response.firing_rate,
                        color=color_map[stim_current],
                        label=f"{stim_current} µA",
                    )
                    ax_pulse_fr.axvspan(
                        0, tparams["post_stim_blank_ms"], color="lightgray", alpha=0.5)
                    ax_pulse_fr.axvspan(
                        tparams["pulse_win_ms"] - tparams["pre_stim_blank_ms"],
                        tparams["pulse_win_ms"],
                        color="lightgray",
                        alpha=0.5,
                    )

                    ax_pulse_fr.set_xlim([0, tparams["pulse_win_ms"]])

                    ax_train_fr.plot(
                        train_response.fr_times,
                        train_response.firing_rate,
                        color=color_map[stim_current],
                        label=f"{stim_current} µA",
                    )

                    ax_spike_prob.plot(
                        baseline_metrics["bin_centers"], baseline_metrics["mean_prob_spike"], color="k")
                    ax_spike_prob.fill_between(
                        baseline_metrics["bin_centers"],
                        baseline_metrics["mean_prob_spike"] -
                        baseline_metrics["std_prob_spike"],
                        baseline_metrics["mean_prob_spike"] +
                        baseline_metrics["std_prob_spike"],
                        color="k",
                        alpha=0.3,
                    )

                    ax_spike_prob.plot(
                        stim_metrics["bin_centers"], stim_metrics["mean_prob_spike"], color=color_map[stim_current]
                    )
                    ax_spike_prob.fill_between(
                        stim_metrics["bin_centers"],
                        stim_metrics["mean_prob_spike"] -
                        stim_metrics["std_prob_spike"],
                        stim_metrics["mean_prob_spike"] +
                        stim_metrics["std_prob_spike"],
                        color=color_map[stim_current],
                        alpha=0.3,
                    )

                    ax_spike_prob.set_xlim([0, 10])
                    # ax_spike_prob.set_ylim([0, ymax])

                    ax_spike_prob.axvspan(
                        0, tparams["post_stim_blank_ms"], color="lightgray", alpha=0.5)
                    ax_spike_prob.axvspan(
                        tparams["pulse_win_ms"] - tparams["pre_stim_blank_ms"],
                        tparams["pulse_win_ms"],
                        color="lightgray",
                        alpha=0.5,
                    )

            if plot_flag:
                ax_pulse_raster.set_ylabel("Pulses")
                ax_pulse_fr.set_ylabel("Firing Rate (Hz)")
                ax_pulse_fr.set_xlabel("Time (ms)")

                ax_train_raster.set_ylabel("Trials")

                ax_train_fr.set_ylabel("Firing Rate (Hz)")
                ax_train_fr.set_xlabel("Time (ms)")

                ax_spike_prob.set_ylabel("Probability")
                ax_spike_prob.set_xlabel("Time (ms)")

                for i, curr in enumerate(currents):
                    pre_stim_fr = pre_stim_fr_list[i]
                    # Set alpha based on threshold
                    alpha_value = 0.3 if pre_stim_fr < pre_stim_threshold else 1.0
                    ax_z_score.bar(
                        curr, z_scores[i], color=color_map[curr], alpha=alpha_value)
                ax_z_score.set_xlabel("Current (µA)")
                ax_z_score.set_xticks(currents)
                ax_z_score.set_xticklabels([str(curr) for curr in currents])
                ax_z_score.set_ylabel("Z-Score")

                # Initialize a variable to track the maximum y-value to adjust y-limits
                max_y_value = float("-inf")

                for i, (curr, latency, jitter, is_pulse_locked, pre_stim_fr) in enumerate(
                    zip(currents, latencies, jitters,
                        pulse_locked_list, pre_stim_fr_list)
                ):

                    alpha_value = 0.3 if pre_stim_fr < pre_stim_threshold else 1.0

                    ax_latency.errorbar(
                        curr, latency, yerr=jitter, fmt="o", color=color_map[curr], alpha=alpha_value)

                    max_y_value = max(max_y_value, latency + jitter)

                    # Draw an asterisk if the condition is pulse locked
                    if is_pulse_locked:
                        ax_latency.text(
                            curr, latency + jitter + 0.05, "*", ha="center", va="bottom", fontsize=12, color="k"
                        )
                        # Update max_y_value to include the asterisk position
                        max_y_value = max(max_y_value, latency + jitter + 0.3)

                # Set x-ticks to be the currents
                ax_latency.set_xticks(currents)
                ax_latency.set_xticklabels([str(curr) for curr in currents])
                ax_latency.set_xlabel("Current (µA)")
                ax_latency.set_ylabel("Time (ms)")

                # Adjust y-limits to ensure the asterisks stay within the figure
                y_min, y_max = ax_latency.get_ylim()
                # Add a small margin above the max y-value
                ax_latency.set_ylim(y_min, max_y_value + 0.2)

                plot_template(
                    session_responses, unit_id, ax=ax_template)

                plt.suptitle(f"Unit {unit_id} - Stim Channel D{stim_channel}")

                plt.tight_layout()
                plt.show()
                plt.pause(1)

                # Save the figure and store the results
                if plot_flag:
                    save_plot_and_create_dir(
                        fig, session_responses.sorting_analyzer.folder, unit_id, stim_channel)

    # Save the collected results to a CSV file
    save_results_to_csv(results, session_responses.sorting_analyzer.folder)


def get_stim_colormap(max_current=20):
    """
    Generates a colormap for currents, assigning specific colors for currents between 3 to 10 µA,
    and assigning the last color in the palette to 2 µA. Black is assigned for currents outside this range (0, 1, >10 µA).

    Args:
        max_current (int): Maximum current value for assigning colors. Defaults to 20 µA.

    Returns:
        dict: A dictionary mapping currents to specific colors.
    """
    # Define the range of currents that should get a color (3 to 10 µA inclusive)
    valid_currents = list(range(3, 11))

    # Generate a color palette for the valid currents (from 3 to 10 µA)
    palette = sns.color_palette("deep", len(valid_currents))

    # Create a fixed mapping of valid currents (3 to 10 µA) to the color palette
    colormap = {curr: palette[i] for i, curr in enumerate(valid_currents)}

    # Assign the last color in the palette (palette[-1]) to 2 µA
    colormap[2] = palette[-1]

    # Assign black for currents 0, 1, and anything outside the valid range (above 10 µA)
    for invalid_curr in range(0, max_current + 1):
        if invalid_curr < 2 or invalid_curr > 10:
            # Assign black color to invalid currents
            colormap[invalid_curr] = (0, 0, 0)

    return colormap


# %%
if __name__ == "__main__":
    plt.style.use("default")
    data_folders = file_util.file_dialog("C://data")
    # data_folders = ["C:\\data\\ICMS93\\Behavior\\30-Aug-2023"]

    sorted_data_folders = file_util.sort_data_folders(data_folders)

    for data_folder in sorted_data_folders:

        figures_dir = Path(data_folder) / "batch_sort" / \
            "figures" / "stim_condition_figures"

        # Delete the directory if it exists
        # if figures_dir.exists() and figures_dir.is_dir():
        #     shutil.rmtree(figures_dir)
        #     print(f"Deleted existing directory: {figures_dir}")

        pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"
        with open(pkl_path, "rb") as file:
            session_responses = pickle.load(file)

        # Ensure necessary extensions are computed
        required_extensions = [
            "random_spikes",
            "waveforms",
            "templates",
            "template_similarity",
            "correlograms",
            "spike_amplitudes",
            "unit_locations",
        ]
        for ext in required_extensions:
            if not session_responses.sorting_analyzer.has_extension(ext):
                if ext == "correlograms":
                    session_responses.sorting_analyzer.compute(
                        ext, window_ms=100, bin_ms=0.5
                    )
                elif ext == "unit_locations":
                    session_responses.sorting_analyzer.compute(
                        ext, method="center_of_mass"
                    )
                else:
                    session_responses.sorting_analyzer.compute(ext)

        process_unit_stim_responses(session_responses, plot_flag=False)

        # make_ppt(data_folder)

# %%


# unit_response = session_responses.get_unit_response(unit_id=18)
# stim_condition_5uA = unit_response.get_stim_response(stim_channel=12, stim_current=7)

# train_5uA = stim_condition_5uA.train_response
# raster_5uA = train_5uA.raster_array

# # Your existing plot code
# fig, ax = plt.subplots()
# ax.eventplot(
#     raster_5uA,
#     orientation="horizontal",
#     colors="k",
#     linelengths=1,
#     linewidths=1,
#     lineoffsets=1,
# )

# # Set the x-axis limits for the main plot
# ax.set_xlim([-700, 1400])

# # Create a secondary axis above the raster plot
# ax2 = ax.twiny()  # Creates a new x-axis sharing the same y-axis
# ax2.set_xlim(ax.get_xlim())  # Match the x-axis limits with the main plot

# # Hide the secondary axis' ticks and labels
# ax2.tick_params(axis="x", which="both", length=0)
# ax2.set_xticklabels([])

# Draw a horizontal bar from 0 to 700 ms above the raster plot
# ax2.hlines(y=41, xmin=0, xmax=700, colors="k", linewidth=5)

# ax.set_xlabel("Time (ms)")
# ax.set_ylabel("Trial")
