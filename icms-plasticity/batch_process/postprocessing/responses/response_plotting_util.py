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


def extract_unit_id(image_path):
    """
    Extracts the unit ID from the filename.
    Assumes filenames are in the format 'Unit_<unit_id>_*.png'.
    """
    # Use a regular expression to extract the unit number from the filename
    match = re.search(r"Unit_(\d+)_", image_path.name)
    if match:
        return int(match.group(1))
    return None


def make_ppt(session_path):
    date_str = file_util.get_date_str(session_path)
    figures_dir = Path(session_path) / "batch_sort" / \
        "figures" / "stim_condition_figures"

    # Collect all image paths from the directory
    # Adjust the pattern if needed
    image_paths = list(figures_dir.glob("*.png"))

    # Group images by unit ID
    images_by_unit = defaultdict(list)
    for image_path in image_paths:
        unit_id = extract_unit_id(image_path)
        if unit_id is not None:
            images_by_unit[unit_id].append(image_path)

    ppt_inserter = PPTImageInserter(grid_dims=(
        2, 2), spacing=(0.05, 0.05), title_font_size=16)

    # Add images to the PPT, grouping them by unit ID
    for unit_id, image_paths in images_by_unit.items():
        slide_title = f"{date_str}: unit {unit_id}"
        ppt_inserter.add_slide(slide_title)
        for image_path in image_paths:
            # Pass the image path as a string (required by pptx)
            ppt_inserter.add_image(str(image_path), slide_title=slide_title)

    # Save or return the ppt_inserter object as needed
    ppt_inserter.save(Path(session_path) / "batch_sort" /
                      "figures" / "stim_responses.pptx")

    ppt_inserter.save(Path(session_path).parent / "analysis" /
                      f"stim_responses_{Path(session_path).stem}.pptx")


def save_plot_and_create_dir(fig, session_path, unit_id, stim_channel):
    # Define the base directory for saving figures
    figures_dir = Path(session_path).parent.parent / \
        "figures" / "stim_condition_figures"

    # Create the directory if it doesn't exist
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Define the file name and path
    fig_filename = f"Unit_{unit_id}_Stim_Channel_{stim_channel}.png"
    fig_path = figures_dir / fig_filename

    # Save the figure
    fig.savefig(fig_path)
    plt.close(fig)  # Close the figure after saving to free up memory

    print(f"Figure saved to {fig_path}")


def plot_template(stim_response, plot_scalebar=True, y_negative=-80, ax=None):

    session_responses = stim_response.session
    unit_id = stim_response.unit_id

    pre_peak_space = 70
    plot_color = "k"
    x_offset = 0

    analyzer = session_responses.sorting_analyzer
    templates_ext = analyzer.get_extension("templates")

    template_mean = templates_ext.get_unit_template(
        unit_id, operator="average")
    template_std = templates_ext.get_unit_template(unit_id, operator="std")

    ch_loc = analyzer.get_channel_locations()
    ch_idx = analyzer.channel_ids_to_indices(analyzer.channel_ids)

    # first index deep, last index shallow
    depth_order = np.argsort(ch_loc[:, 1])
    abs_depth_id = 31 - np.sort(ch_loc[:, 1]) / 60 + 1

    t1 = 0
    t2 = 181
    ordered_template_mean = template_mean[t1:t2, depth_order]
    ordered_template_std = template_std[t1:t2, depth_order]

    primary_ch = np.argmin(np.min(ordered_template_mean, axis=0))
    channels_to_plot = [primary_ch - 1, primary_ch, primary_ch + 1]

    # Handle edge cases
    if primary_ch == 0:  # If primary channel is at the top
        channels_to_plot = [primary_ch, primary_ch + 1, primary_ch + 2]
    # If primary channel is at the bottom
    elif primary_ch == len(ch_idx) - 1:
        channels_to_plot = [primary_ch - 2, primary_ch - 1, primary_ch]

    peak_indices = [np.argmin(ordered_template_mean[:, ch_idx])
                    for ch_idx in channels_to_plot]

    # Calculate the offsets based on the std values before the peak
    offsets = [i * pre_peak_space for i in range(len(channels_to_plot))]

    for i, ch_idx in enumerate(channels_to_plot):
        # Get the mean and std for the current channel
        depth_id = "D" + str(int(abs_depth_id[ch_idx]))
        mean_waveform = ordered_template_mean[:, ch_idx]
        std_waveform = ordered_template_std[:, ch_idx]
        x = np.arange(mean_waveform.shape[0]) + x_offset

        # Plot the mean waveform
        ax.plot(x, mean_waveform + offsets[i], "k")
        ax.text(5, offsets[i] - 20, depth_id, fontsize=8)

        # Add shading for the std
        ax.fill_between(
            x,  # Use the same x-axis values as for the mean waveform
            mean_waveform - std_waveform + offsets[i],
            mean_waveform + std_waveform + offsets[i],
            color="k",
            alpha=0.2,  # Adjust transparency of the shading
        )

    ax.set_ylim([y_negative, offsets[-1] + 50])
    if plot_scalebar:
        h_pos = (0, y_negative + 10)  # Position for horizontal scale bar
        v_pos = (0, y_negative + 10)  # Position for vertical scale bar
        h_length = 30  # Length of horizontal scale bar (in samples)
        # Length of vertical scale bar (in amplitude units)
        v_length = 100

        plot_util.add_scale_bars_wvf(
            ax,
            h_pos,
            v_pos,
            h_length,
            v_length,
            line_width=1,
            h_label="1 ms",
            v_label="100 uV",
            v_label_x_offset=0,
            v_label_y_offset=50,
            h_label_x_offset=20,
            h_label_y_offset=-10,
        )
    ax.axis("off")


def plot_primary_channel_template(stim_response, plot_scalebar=True, y_negative=-80, ax=None):

    session_responses = stim_response.session
    unit_id = stim_response.unit_id

    plot_color = "k"
    x_offset = 0

    analyzer = session_responses.sorting_analyzer
    templates_ext = analyzer.get_extension("templates")

    template_mean = templates_ext.get_unit_template(
        unit_id, operator="average")
    template_std = templates_ext.get_unit_template(unit_id, operator="std")

    ch_loc = analyzer.get_channel_locations()
    ch_idx = analyzer.channel_ids_to_indices(analyzer.channel_ids)

    # first index deep, last index shallow
    depth_order = np.argsort(ch_loc[:, 1])
    abs_depth_id = 31 - np.sort(ch_loc[:, 1]) / 60 + 1

    t1 = 0
    t2 = 181
    ordered_template_mean = template_mean[t1:t2, depth_order]
    ordered_template_std = template_std[t1:t2, depth_order]

    # Find primary channel as the channel with the minimum peak amplitude
    primary_ch = np.argmin(np.min(ordered_template_mean, axis=0))

    # Get the peak index and depth ID for the primary channel
    depth_id = "D" + str(int(abs_depth_id[primary_ch]))
    mean_waveform = ordered_template_mean[:, primary_ch]
    std_waveform = ordered_template_std[:, primary_ch]
    x = np.arange(mean_waveform.shape[0]) + x_offset

    # Plot the mean waveform for the primary channel
    ax.plot(x, mean_waveform, plot_color)

    # Add shading for the std
    ax.fill_between(
        x,  # Use the same x-axis values as for the mean waveform
        mean_waveform - std_waveform,
        mean_waveform + std_waveform,
        color=plot_color,
        alpha=0.2,  # Adjust transparency of the shading
    )

    # ax.set_ylim([y_negative, y_negative + 100])
    if plot_scalebar:
        h_pos = (0, y_negative + 10)  # Position for horizontal scale bar
        v_pos = (0, y_negative + 10)  # Position for vertical scale bar
        h_length = 30  # Length of horizontal scale bar (in samples)
        v_length = 100  # Length of vertical scale bar (in amplitude units)
        plot_util.add_scale_bars_wvf(
            ax,
            h_pos,
            v_pos,
            h_length,
            v_length,
            line_width=1,
            h_label="1 ms",
            v_label="100 uV",
        )
    ax.axis("off")


def save_results_to_csv(results, session_path):
    # Define the CSV file path
    results_dir = Path(session_path).parent.parent
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "stim_condition_results.csv"
    pkl_path = results_dir / "stim_condition_results.pkl"

    # Convert results dictionary to DataFrame
    df = pd.DataFrame(results)

    # Specify the desired column order
    columns = [
        "animal_id",
        "unit_id",
        "stim_channel",
        "stim_current",
        "unit_location",
        "cell_type",
        "spike_prob",
        "is_pulse_locked",
        "pli",
        "latency",
        "jitter",
        "t_val",
        "modulated",
        "z_score",
        "pulse_mean_fr",
        "train_mean_fr",
        "pre_stim_mean_fr",
        "fr_times",
        "firing_rate",
        "num_spikes",
        "template",
        "baseline_too_slow",
    ]

    for col in columns:
        if col not in df.columns:
            df[col] = np.nan  # Fill missing columns with NaN

    numeric_columns = [
        "unit_id",
        "stim_channel",
        "stim_current",
        "spike_prob",
        "pli",
        "latency",
        "jitter",
        "t_val",
        "z_score",
        "pulse_mean_fr",
        "train_mean_fr",
        "pre_stim_mean_fr",
        "num_spikes",
    ]

    # Apply pd.to_numeric to the numeric columns to ensure they are numeric
    df[numeric_columns] = df[numeric_columns].apply(
        pd.to_numeric, errors='coerce')

    df = df[columns]

    # Save DataFrame to CSV
    df.to_csv(csv_path, index=False)

    df.to_pickle(pkl_path)

    print(f"Results saved to {csv_path}")


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
    valid_currents = list(range(2, 11))

    # Generate a color palette for the valid currents (from 3 to 10 µA)
    palette = sns.color_palette("deep", len(valid_currents))

    # Create a fixed mapping of valid currents (3 to 10 µA) to the color palette
    colormap = {curr: palette[i] for i, curr in enumerate(valid_currents)}

    # Assign black for currents 0, 1, and anything outside the valid range (above 10 µA)
    for invalid_curr in range(0, max_current + 1):
        if invalid_curr < 2 or invalid_curr > 10:
            # Assign black color to invalid currents
            colormap[invalid_curr] = (0, 0, 0)

    return colormap


def initialize_plotting(plot_flag, n_currents):
    if plot_flag:
        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(3, 3, width_ratios=[
                              2, 2, 1])  # 3 rows, 3 columns
        axs = {
            "pulse_raster": fig.add_subplot(gs[0, 0]),
            "pulse_fr": fig.add_subplot(gs[1, 0]),
            "spike_prob": fig.add_subplot(gs[2, 0]),
            "z_score": fig.add_subplot(gs[2, 1]),
            "train_raster": fig.add_subplot(gs[0, 1]),
            "train_fr": fig.add_subplot(gs[1, 1]),
            "template": fig.add_subplot(gs[:2, 2]),
            "t": fig.add_subplot(gs[2, 2])
        }
        return fig, axs
    return None, None

# %%


def plot_train_raster(ax, train_response, color, linewidth_factor, train_line_offset):

    train_raster_array = train_response.raster_array
    train_good_linelength = np.ceil(
        len(train_raster_array) / linewidth_factor)
    train_lineoffsets = (
        np.arange(len(train_raster_array)) + train_line_offset
    )  # Stack rasters vertically

    ax.eventplot(
        train_raster_array,
        orientation="horizontal",
        colors=color,
        linelengths=train_good_linelength,
        linewidths=1,
        lineoffsets=train_lineoffsets,
    )
    # Update the train line offset for the next current
    train_line_offset += len(train_raster_array)

    ax.set_xlim([-200, 900])
    # ax.axvline(
    #     x=0, color="gray", linestyle="--", linewidth=1)
    # ax.axvline(
    #     x=700, color="gray", linestyle="--", linewidth=1)


def plot_train_raster_with_scatter(ax, tr, color, train_line_offset=0):
    train_raster_array = tr.raster_array
    num_trials = len(train_raster_array)
    train_lineoffsets = np.arange(num_trials) + \
        train_line_offset  # Stack rasters vertically

    # Loop over each trial and plot the spike times using scatter
    for i, trial_spike_times in enumerate(train_raster_array):
        # Scatter plot the spikes for each trial
        ax.scatter(
            trial_spike_times,          # X-axis: spike times
            # Y-axis: trial number
            np.full_like(trial_spike_times, train_lineoffsets[i]),
            color=color,
            marker="_",                 # Use a small horizontal line as the marker
            s=5,                        # Marker size
            linewidths=1,               # Thickness of the marker line
        )

    # Update the train line offset
    train_line_offset += num_trials


def plot_train_firing_rate(ax, train_response, color):

    stim_current = train_response.stim_current
    ax.plot(
        train_response.fr_times,
        train_response.firing_rate,
        color=color,
        label=f"{stim_current} µA",
    )

    # ax.axvline(
    #     x=0, color="gray", linestyle="--", linewidth=1)
    # ax.axvline(
    #     x=700, color="gray", linestyle="--", linewidth=1)


# def plot_train_firing_rate(ax, train_response, color):
#     stim_current = train_response.stim_current
#     ax.plot(
#         train_response.fr_times,
#         train_response.firing_rate,
#         color=color,
#         label=f"{stim_current} µA",
#     )

#     # Add black bar from 0 to 700 above the plot to represent the stimulus
#     ax.plot([0, 700], [max(train_response.firing_rate) + 5,
#             max(train_response.firing_rate) + 5], color="black", linewidth=5)

#     # Adjust y-limits based on the firing rate values
#     ax.set_ylim([0, max(train_response.firing_rate) + 10])

#     # Keep the x-axis limits consistent with the raster plot
#     ax.set_xlim([-700, 1400])

#     ax.set_title(f"Stimulus Current: {stim_current} µA")


def plot_pulse_raster(ax, pulse_response, color, linewidth_factor, pulse_line_offset):
    tparams = pulse_response.timing_params
    stim_current = pulse_response.stim_current
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

    ax.eventplot(
        pulse_raster_array,
        orientation="horizontal",
        linewidths=1,
        linelengths=pulse_good_linelength,
        lineoffsets=pulse_lineoffsets,
        colors=color,
    )

    ax.axvspan(
        0, tparams["post_stim_blank_ms"], color="lightgray", alpha=0.5)
    ax.axvspan(
        tparams["pulse_win_ms"] - tparams["pre_stim_blank_ms"],
        tparams["pulse_win_ms"],
        color="lightgray",
        alpha=0.5,
    )
    ax.set_xlim([0, tparams["pulse_win_ms"]])


def plot_pulse_firing_rate(ax, pulse_response, color):

    tparams = pulse_response.timing_params
    stim_current = pulse_response.stim_current
    ax.plot(
        pulse_response.fr_times,
        pulse_response.firing_rate,
        color=color,
        label=f"{stim_current} µA",
    )
    ax.axvspan(
        0, tparams["post_stim_blank_ms"], color="lightgray", alpha=0.5)
    ax.axvspan(
        tparams["pulse_win_ms"] - tparams["pre_stim_blank_ms"],
        tparams["pulse_win_ms"],
        color="lightgray",
        alpha=0.5,
    )

    ax.set_xlim([0, tparams["pulse_win_ms"]])


def plot_probability(ax, pulse_response, color):

    tparams = pulse_response.timing_params
    baseline_metrics = pulse_response.baseline_metrics
    stim_metrics = pulse_response.stim_metrics

    ax.plot(
        baseline_metrics["bin_centers"], baseline_metrics["mean_prob_spike"], color="k")
    ax.fill_between(
        baseline_metrics["bin_centers"],
        baseline_metrics["mean_prob_spike"] -
        baseline_metrics["std_prob_spike"],
        baseline_metrics["mean_prob_spike"] +
        baseline_metrics["std_prob_spike"],
        color="k",
        alpha=0.3,
    )
    ax.plot(stim_metrics["bin_centers"],
            stim_metrics["mean_prob_spike"], color=color)
    ax.fill_between(
        stim_metrics["bin_centers"],
        stim_metrics["mean_prob_spike"] -
        stim_metrics["std_prob_spike"],
        stim_metrics["mean_prob_spike"] +
        stim_metrics["std_prob_spike"],
        color=color,
        alpha=0.3,
    )

    ax.axvspan(
        0, tparams["post_stim_blank_ms"], color="lightgray", alpha=0.5)
    ax.axvspan(
        tparams["pulse_win_ms"] - tparams["pre_stim_blank_ms"],
        tparams["pulse_win_ms"],
        color="lightgray",
        alpha=0.5,
    )
    ax.set_xlim([0, 10])


def plot_z_scores(ax, z_scores, currents, pre_stim_fr_list, color_map, pre_stim_threshold):
    for i, curr in enumerate(currents):
        pre_stim_fr = pre_stim_fr_list[i]
        # Set alpha based on threshold
        alpha_value = 0.3 if pre_stim_fr < pre_stim_threshold else 1.0
        ax.bar(
            curr, z_scores[i], color=color_map[curr], alpha=alpha_value)
    ax.set_xlabel("Current (µA)")
    ax.set_xticks(currents)
    ax.set_xticklabels([str(curr) for curr in currents])
    ax.set_ylabel("Z-Score")
