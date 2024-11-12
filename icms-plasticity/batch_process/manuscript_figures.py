from util.aggregated_weekly_data_plot import plot_aggregated_weekly_data_with_iqr
from batch_process.util.plotting import add_scale_bars_wvf
from util.load_data import load_data
from scipy.stats import norm
from scipy.stats import mannwhitneyu
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle
import matplotlib.cm as cm
from matplotlib.patches import Polygon, Circle
from probeinterface.plotting import plot_probe, plot_probegroup
from probeinterface import Probe, ProbeGroup
import probeinterface as pb
from batch_process.postprocessing.plot_longitudinal_data_v2 import load_session_data
import seaborn as sns  # For color palette
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import batch_process.postprocessing.responses.response_plotting_util as rpu
import numpy as np
import spikeinterface.full as si
from pathlib import Path
import pickle
import batch_process.util.template_util as template_util
import batch_process.postprocessing.stim_response_util as stim_response_util

from matplotlib.lines import Line2D

# %%
linewidth_factor = 40
data_folder = "C:\\data\\ICMS92\Behavior\\01-Sep-2023"
pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"
with open(pkl_path, "rb") as file:
    session_responses = pickle.load(file)


# %%
# fig, ax = plt.subplots(figsize=(6.5, 2.5))
# fig = plt.figure(figsize=(6.5, 4))
fig = plt.figure(figsize=(6, 2.5))

# 3 rows, 2 columns with equal widths
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[3, 1])
axs = {}
# Example of adding subplots to this grid
axs["train_raster"] = fig.add_subplot(gs[0, 0:2])  # Row 0, Column 0
axs["a"] = fig.add_subplot(gs[1, 0:2])  # Row 0, Column 0
# axs["b"] = fig.add_subplot(gs[1, 1])  # Row 0, Column 0

ur = session_responses.get_unit_response(6)  # 6


for i in np.arange(0, 700, 10):
    axs["train_raster"].axvline(i, color='r', linewidth=0.5)

train_line_offset = 0
for idx, current in enumerate(np.arange(3, 6)):
    scr = ur.get_stim_response(9, current)
    tr = scr.train_response

    rpu.plot_train_raster_with_scatter(
        axs["train_raster"], tr, 'k', train_line_offset)
    if idx > -1:
        axs["train_raster"].axhline(
            y=train_line_offset - 0.5, color='k', linewidth=0.5)

    midpoint = train_line_offset + len(tr.raster_array) / 2

    # Add the current label outside the axis (at the midpoint)
    axs["train_raster"].text(
        x=-120,  # Place the text outside the left of the axis
        y=midpoint,
        s=f"{current} µA",  # Text for current value
        va='center',  # Vertically align the text to the center of the segment
        # Horizontally align the text to the right (outside the plot)
        ha='right',
        fontsize=10,
    )
    train_line_offset += len(tr.raster_array)


# ax.set_xlabel('')
axs["train_raster"].set_ylabel('')

# ax.set_xticks([])  # Clear x-ticks
axs["train_raster"].set_yticks([])  # Clear y-ticks
axs["train_raster"].set_ylim([0 - 0.5, train_line_offset + 0.5])
axs["train_raster"].set_xlim([-100, 800])
axs["train_raster"].set_xlabel("Time (ms)")

rec = session_responses.sorting_analyzer.recording

scr = ur.get_stim_response(9, 3)
stim_ts = scr.stim_timestamps

trial = 3  # 3
t1 = trial * 70
t2 = trial * 70 + 10
stim_spikes = scr.stim_spikes[(scr.stim_spikes >= stim_ts[t1]) & (
    scr.stim_spikes < stim_ts[t2])]

ch_ids = rec.get_channel_ids()
ch_locs = rec.get_channel_locations()
sorted_indices = np.argsort(ch_locs[:, 1])[::-1]


traces = rec.get_traces(
    start_frame=stim_ts[t1], end_frame=stim_ts[t2], return_scaled=True)
for i, ch_id in enumerate([3]):
    axs["a"].vlines(x=(stim_ts[t1:t2] - stim_ts[t1]) /
                    30, ymin=-200, ymax=200, colors='r')
    axs["a"].plot((np.arange(traces.shape[0]) / 30), traces[:,
                  ch_id] + i * 250, 'k')  # Convert trace samples to ms

axs["a"].scatter(
    x=(stim_spikes - stim_ts[t1]) / 30,  # Align the spikes to the time window
    # Place the triangles at y=-100 (adjust as needed)
    y=np.full_like(stim_spikes, -300),
    color='r',
    marker='^',  # Upward pointing triangle
    s=100,
    label='Spikes'
)

axs["a"].set_xlabel("Time (ms)")

plt.tight_layout()  # Adjust layout to prevent overlapping

# %% Plot waveforms
t1 = 0
t2 = 70 * 40 - 1
stim_spikes = scr.stim_spikes[(scr.stim_spikes >= stim_ts[t1]) & (
    scr.stim_spikes < stim_ts[t2])]

traces = rec.get_traces(
    start_frame=stim_ts[t1], end_frame=stim_ts[t2], return_scaled=True)
filt_trace = traces[:, 3]

# %% Raw waveforms
dataloader = load_data(
    data_folder, make_folder=True, save_folder_name="batch_sort", first_N_files=4, server_mount_drive="S:"
)
rec_raw = dataloader.recording
raw_traces = rec_raw.get_traces(
    start_frame=stim_ts[t1], end_frame=stim_ts[t2], return_scaled=True)
raw_trace = raw_traces[:, 3]
# %%
stim_spikes_norm = stim_spikes - stim_ts[t1]

raw_spikes = []
preproc_spikes = []

for stim_spike in stim_spikes_norm[0:300]:
    stim_spike = int(stim_spike)  # Ensure it's an integer
    raw_spike = raw_trace[stim_spike - 30 *
                          3: stim_spike + 30 * 3]  # Extract 7 ms window
    filt_spike = filt_trace[stim_spike - 30 * 3: stim_spike + 30 * 3]
    raw_spikes.append(raw_spike)
    preproc_spikes.append(filt_spike)

raw_spikes = np.array(raw_spikes)
preproc_spikes = np.array(preproc_spikes)

# Plot raw and filtered spikes
# plt.subplot(1, 2, 1)
# plt.plot(raw_spikes.T, 'k', linewidth=0.5)
# plt.ylim([-4500, -2000])
# plt.subplot(1, 2, 2)
# plt.plot(filt_spikes.T, 'k', linewidth=0.5)
# plt.ylim([-400, 300])

# %%
# Filter good spikes based on value criteria (not index)
# Example: spikes with peak-to-peak amplitude greater than 3000
good_spike_indices = np.where(
    (np.max(raw_spikes, axis=1) - np.min(raw_spikes, axis=1)) > 3000)
good_spike_indices = np.where(
    (np.argmax(raw_spikes, axis=1) < 30) & (np.min(raw_spikes, axis=1) > -4000))
good_spike_indices = np.where(
    (np.argmax(raw_spikes, axis=1) < 30) & (np.min(raw_spikes, axis=1) > -4000) &
    (np.min(preproc_spikes, axis=1) > -220) & (raw_spikes[:, 50] > -2000) &
    (raw_spikes[:, 10] > -2000))


# Get the good spikes for both raw and filtered
good_raw_spikes = raw_spikes[good_spike_indices]
good_preproc_spikes = preproc_spikes[good_spike_indices]

# fig, axes = plt.subplots(1, 2, figsize=(2.5, 3))
fig, axes = plt.subplots(1, 2, figsize=(4, 2))

# Plot the good raw spikes in the first subplot
axes[0].plot(good_raw_spikes.T, 'k', linewidth=0.5, alpha=0.8)
axes[0].set_ylim([-2600, -1200])
axes[0].set_title("Raw Spikes", fontsize=9)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].set_xlabel('')
axes[0].set_ylabel('')


# Plot the good filtered spikes in the second subplot
axes[1].plot(good_preproc_spikes.T, 'k', linewidth=0.5, alpha=0.8)
axes[1].set_ylim([-430, 200])
axes[1].set_title("Preprocessed Spikes", fontsize=9)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].set_xlabel('')
axes[1].set_ylabel('')

add_scale_bars_wvf(ax=axes[0], h_pos=[40, -2400], v_pos=[40, -2400],
                   h_length=30, v_length=200, v_label_y_offset=170, h_label_y_offset=-20, h_label_x_offset=15, v_label_x_offset=0,
                   h_label="1 ms", v_label="200 μV", font_size=7)
add_scale_bars_wvf(ax=axes[1], h_pos=[40, -350], v_pos=[40, -350],
                   h_length=30, v_length=200, v_label_y_offset=80, h_label_y_offset=-12, h_label_x_offset=15, v_label_x_offset=0,
                   h_label="1 ms", v_label="200 μV", font_size=7)

# Adjust the layout to prevent overlap and make it more readable
plt.tight_layout()

#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

good_spike_indices = np.where(
    (np.max(raw_spikes, axis=1) - np.min(raw_spikes, axis=1)) > 3000)
good_spike_indices = np.where(
    (np.argmax(raw_spikes, axis=1) < 30) & (np.min(raw_spikes, axis=1) > -4000))
good_spike_indices = np.where(
    (np.argmax(raw_spikes, axis=1) < 30) & (np.min(raw_spikes, axis=1) > -4000) &
    (np.min(preproc_spikes, axis=1) > -220) & (raw_spikes[:, 50] > -2000) &
    (raw_spikes[:, 10] > -2000))


# Get the good spikes for both raw and filtered
good_raw_spikes = raw_spikes[good_spike_indices]
good_preproc_spikes = preproc_spikes[good_spike_indices]

# Define the bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Filter parameters
lowcut = 300  # Low cutoff frequency in Hz
highcut = 5000  # High cutoff frequency in Hz
fs = 30000  # Sampling frequency in Hz (adjust to your data's actual rate)

# Rename original "filt_spikes" to "preproc_spikes" for preprocessing only

# Now apply the bandpass filter to raw spikes to get "filt_spikes"
filt_spikes = np.array([bandpass_filter(spike, lowcut, highcut, fs) for spike in good_raw_spikes])


# Create a 1x3 subplot layout for raw, bandpass-filtered raw, and good preprocessed spikes
fig, axes = plt.subplots(1, 3, figsize=(5, 2))

# Plot the raw spikes in the first subplot
axes[0].plot(good_raw_spikes.T, 'k', linewidth=0.5, alpha=0.8)
axes[0].set_ylim([-2600, -1200])
axes[0].set_title("Raw Spikes", fontsize=9)
axes[0].set_xticks([])
axes[0].set_yticks([])

# Plot the bandpass-filtered raw spikes in the second subplot
axes[1].plot(filt_spikes.T, 'k', linewidth=0.5, alpha=0.8)
axes[1].set_ylim([-430, 200])
axes[1].set_title("Bandpass Filtered Spikes", fontsize=9)
axes[1].set_xticks([])
axes[1].set_yticks([])

# Plot the good preprocessed spikes in the third subplot
axes[2].plot(good_preproc_spikes.T, 'k', linewidth=0.5, alpha=0.8)
axes[2].set_ylim([-430, 200])
axes[2].set_title("Preprocessed Spikes", fontsize=9)
axes[2].set_xticks([])
axes[2].set_yticks([])

# Add scale bars for each subplot
add_scale_bars_wvf(ax=axes[0], h_pos=[40, -2400], v_pos=[40, -2400],
                   h_length=30, v_length=200, v_label_y_offset=170, h_label_y_offset=-20, h_label_x_offset=15, v_label_x_offset=0,
                   h_label="1 ms", v_label="200 μV", font_size=7)
bar_x = 40*3
add_scale_bars_wvf(ax=axes[1], h_pos=[bar_x, -350], v_pos=[bar_x, -350],
                   h_length=30, v_length=200, v_label_y_offset=80, h_label_y_offset=-12, h_label_x_offset=15, v_label_x_offset=0,
                   h_label="1 ms", v_label="200 μV", font_size=7)
add_scale_bars_wvf(ax=axes[2], h_pos=[40, -350], v_pos=[40, -350],
                   h_length=30, v_length=200, v_label_y_offset=80, h_label_y_offset=-12, h_label_x_offset=15, v_label_x_offset=0,
                   h_label="1 ms", v_label="200 μV", font_size=7)

# Adjust layout to make it more readable
plt.tight_layout()
plt.show()


# %%
color_map = rpu.get_stim_colormap()
linewidth_factor = 30
data_folder = "C:\\data\\ICMS92\Behavior\\01-Sep-2023"
# data_folder = "C:\\data\\ICMS93\Behavior\\14-Sep-2023"
pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"
with open(pkl_path, "rb") as file:
    session_responses = pickle.load(file)

ur = session_responses.get_unit_response(6)

fig = plt.figure(figsize=(10, 6))
# 3 rows, 2 columns with equal widths
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1])
axs = {}
# Example of adding subplots to this grid
axs["train_raster"] = fig.add_subplot(gs[0, 0])  # Row 0, Column 0
axs["pulse_raster"] = fig.add_subplot(gs[0, 1])  # Row 0, Column 1
axs["train_fr"] = fig.add_subplot(gs[1, 0])  # Row 1, Column 0
axs["pulse_fr"] = fig.add_subplot(gs[1, 1])  # Row 1, Column 1
# axs["fr"] = fig.add_subplot(gs[2:4, 0])  # Row 2, Column 0
# axs["z_score"] = fig.add_subplot(gs[2:4, 1])  # Row 2, Column 1

pulse_line_offset, train_line_offset = 0, 0
for current in np.arange(3, 6):
    scr = ur.get_stim_response(9, current)
    tr = scr.train_response
    pr = scr.pulse_response

    # Plot pulse raster and train raster
    rpu.plot_pulse_raster(
        axs["pulse_raster"], pr, color_map[current], linewidth_factor, pulse_line_offset)
    rpu.plot_train_raster(
        axs["train_raster"], tr, color_map[current], linewidth_factor, train_line_offset)

    # Plot pulse firing rate and train firing rate
    rpu.plot_pulse_firing_rate(axs["pulse_fr"], pr, color_map[current])
    rpu.plot_train_firing_rate(axs["train_fr"], tr, color_map[current])

    # Update the line offsets after plotting each current
    pulse_line_offset += len(pr.raster_array)
    train_line_offset += len(tr.raster_array)

# Add a single black stimulus bar for the entire plot
axs["train_raster"].plot([0, 700], [train_line_offset +
                         5, train_line_offset + 5], color="black", linewidth=4)
axs["train_fr"].plot([0, 700], [max(tr.firing_rate) + 10,
                     max(tr.firing_rate) + 10], color="black", linewidth=4)

# Adjust y-limits for the plots to accommodate the bar and all the data
axs["train_raster"].set_ylim([0, train_line_offset + 10])
axs["train_fr"].set_ylim([0, max(tr.firing_rate) + 20])

axs["train_fr"].set_title("Stim Train Firing Rate", fontsize=10)


axs["train_raster"].set_ylabel("Trial index")
axs["train_raster"].set_xlabel("Time (ms)")
axs["train_raster"].set_title("Stim Train Rasterplot", fontsize=10)

axs["pulse_raster"].set_ylabel("Pulse index")
axs["pulse_raster"].set_xlabel("Time (ms)")
axs["pulse_raster"].set_title("Individual Pulse Rasterplot", fontsize=10)


# Example usage within the plot function
colors = ['C1', 'C2', 'C3']  # Example: 3 different colors for the lines
labels = ['3 µA', '4 µA', '5 µA']  # Labels corresponding to the lines


# Assuming you have your axs['pulse_raster']
add_custom_legend_to_pulse_raster(axs['pulse_fr'], colors, labels)

axs["pulse_fr"].set_ylabel("Firing Rate (Hz)")
axs["pulse_fr"].set_xlabel("Time (ms)")
axs["pulse_fr"].set_title("Pulse Firing Rate", fontsize=10)

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

# %% Firing rate example


def add_custom_legend_to_pulse_raster(ax, colors, labels, loc="upper right", fontsize=6, alpha=1):
    """
    Adds a custom legend to the given axis with the provided colors and labels.

    Args:
        ax (matplotlib axis): The axis where the legend will be added.
        colors (list): A list of colors for the legend lines.
        labels (list): A list of labels corresponding to the colors.
        loc (str, optional): The location of the legend on the plot. Defaults to "upper left".
        fontsize (int, optional): Font size for the legend text. Defaults to 8.
        alpha (float, optional): Transparency of the legend background. Defaults to 0.8.

    Raises:
        ValueError: If `colors` and `labels` do not have the same length.
    """

    # Check that colors and labels are lists and of the same length
    if not isinstance(colors, list) or not isinstance(labels, list):
        raise TypeError("Both 'colors' and 'labels' must be lists.")

    if len(colors) != len(labels):
        raise ValueError("'colors' and 'labels' must have the same length.")

    # Create Line2D objects with the specified colors and labels
    custom_lines = [Line2D([0], [0], color=color, lw=2) for color in colors]

    # Add the legend to the axis
    ax.legend(custom_lines, labels, loc=loc, prop={'size': fontsize})


ur = session_responses.get_unit_response(6)

color_map = rpu.get_stim_colormap()
data_folder = "C:\\data\\ICMS92\Behavior\\01-Sep-2023"
pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"

with open(pkl_path, "rb") as file:
    session_responses = pickle.load(file)


# %%
# fig, ax = plt.subplots(figsize=(3, 2.5))

fig, ax = plt.subplots(figsize=(1.5, 3))

for current in np.arange(3, 6):
    scr = ur.get_stim_response(9, current)
    tr = scr.train_response
    pr = scr.pulse_response
    rpu.plot_train_firing_rate(ax, tr, color_map[current])

ax.plot([0, 700], [max(tr.firing_rate) + 10,
        max(tr.firing_rate) + 10], color="black", linewidth=4)
# ax.set_title("Stim Train Firing Rate", fontsize=10)
# Example usage within the plot function
colors = ['C1', 'C2', 'C3']  # Example: 3 different colors for the lines
labels = ['3 µA', '4 µA', '5 µA']  # Labels corresponding to the lines

# Assuming you have your axs['pulse_raster']
add_custom_legend_to_pulse_raster(ax, colors, labels, loc="center right")
ax.set_xlim([-1000, 1700])
ax.set_ylim([-20, 100])

add_scale_bars_wvf(ax, h_pos=[-800, -5], v_pos=[-800, -5],
                   h_length=200, v_length=10, v_label_x_offset=0, v_label_y_offset=5,
                   h_label_x_offset=400, h_label_y_offset=-2,
                   h_label="200 ms", v_label="10 Hz", font_size=7)

ax.set_axis_off()


# %% Figure 4D


def filter_data(df):
    # Filter data using the correct comparison for boolean and numerical conditions
    for ch in df['stim_channel'].unique():
        ch_df = df[df['stim_channel'] == ch]
        if len(ch_df['days_relative'].unique()) < 5:
            df = df[df['stim_channel'] != ch]

    df = df[
        (df["baseline_too_slow"] == False) &
        (df['significantly_modulated'] == True) &
        (df["z_score"] > 0) &
        (df["z_score"] < 40) &
        (df['num_spikes'] > 150) &
        (df['stim_current'] < 7) &
        (df['stim_current'] > 2)]

    return df


animal_ids = ["ICMS92", "ICMS93", "ICMS98",
              "ICMS100", "ICMS101"]  # Example animal IDs
base_path = Path("C:/data/")  # Replace with your actual base path

# Load session data for each animal
dfs = [load_session_data(animal_id, base_path) for animal_id in animal_ids]

# Concatenate the DataFrames for all animals
df = pd.concat(dfs, ignore_index=True)
df = filter_data(df)
df['weeks_relative'] = (df['days_relative'] // 7).astype(int)

row_counts = df.groupby(['stim_current', 'weeks_relative']
                        ).size().reset_index(name='row_count')
row_counts
# %%

# fig = plt.figure(figsize=(12, 5))
fig, ax = plt.subplots(figsize=(3, 2.5))

p_values_var1, _ = plot_aggregated_weekly_data_with_iqr(
    df, var1='z_score', last_week=4, ax1=ax)

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

# %% Plot average train response to different currents

fig, ax = plt.subplots(figsize=(3, 2.5))

# Loop through each stimulation current (2 µA to 6 µA)
for idx, current in enumerate(np.arange(3, 7)):
    all_frs = []  # List to store firing rate arrays for all weeks combined

    # Loop through all weeks to aggregate firing rates for the current
    for week in np.arange(0, 6):
        week_df = df[(df['stim_current'] == current) &
                     (df['weeks_relative'] == week)]

        # Collect all firing rates for the current and week
        for index, row in week_df.iterrows():
            all_frs.append(row['firing_rate'])  # Append each row's firing rate

    # Ensure that there is data to compute the mean
    if len(all_frs) > 0:
        # Calculate the mean firing rate across all weeks
        mean_fr = np.mean(all_frs, axis=0)
        # Plot the mean firing rate for the current with a unique color for each current
        ax.plot(row['fr_times'], mean_fr, color=f'C{idx+1}',  label=f"{
            current} µA")

# Add plot details
colors = ['C1', 'C2', 'C3', 'C4']  # Example: 3 different colors for the lines
labels = ['3 µA', '4 µA', '5 µA', '6 µA']  # Labels corresponding to the lines
ax.plot([0, 700], [max(mean_fr) + 5,
        max(mean_fr) + 5], color="black", linewidth=4)
add_custom_legend_to_pulse_raster(ax, colors, labels, loc="upper left")
ax.set_xlim([-1000, 1700])
ax.set_ylim([-20, 50])

add_scale_bars_wvf(ax, h_pos=[-800, -5], v_pos=[-800, -5],
                   h_length=200, v_length=10, v_label_x_offset=0, v_label_y_offset=5,
                   h_label_x_offset=200, h_label_y_offset=-2,
                   h_label="200 ms", v_label="10 Hz", font_size=7)

ax.set_axis_off()
# %% Figure 4: Modulation increases with session

"""
Show that modulation score increases over time
"""

# Function to scale the radius of the legend circles


def make_legend_circle(legend, orig_handle, **kwargs):
    return Circle((0, 0),
                  radius=orig_handle.get_radius(),  # Match the size of the plot circles
                  facecolor='white',  # White face for legend circles
                  edgecolor='k')  # Black edge for legend circles


data_folder = "C:\\data\\ICMS92\Behavior\\01-Sep-2023"
# data_folder = "C:\\data\\ICMS98\\22-Nov-2023"

# data_folder = "C:\\data\\ICMS93\Behavior\\14-Sep-2023"
pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"
with open(pkl_path, "rb") as file:
    session_responses = pickle.load(file)

# Choose specific stim condition
stim_condition = (9, 4)  # channel depth index (1-indexed) and current

stim_depth = stim_condition[0]


p = pb.read_probeinterface('util/net32Ch.json')

probe_contour = p.probes[0].probe_planar_contour
probe_contour[:, 0] = probe_contour[:, 0] * 3
probe_contour[:, 1] = probe_contour[:, 1] * 1.2
contact_sites = p.probes[0].contact_positions
# first index is deepest
device_channel_indices = p.probes[0].device_channel_indices

# Create the figure and axis
fig, ax = plt.subplots(figsize=(4, 7))

# Create the polygon and set the desired fill color
# You can change 'skyblue' to any color
polygon = Polygon(probe_contour, closed=True, fill=True, color='#f6f0e0')

# Add the polygon to the axis
ax.add_patch(polygon)

for contact in contact_sites:
    # You can change color
    circle = Circle((0, contact[1]), radius=14, color='#eed79d', fill=True)
    ax.add_patch(circle)

contacts_outside_brain = 5
# ax.axhline(60 * (32 - contacts_outside_brain - 0.5), color='k')

# Set axis limits (optional, adjust based on your data)
ax.set_xlim(-200, 200)
ax.set_ylim(-150, 1200)
ax.set_aspect(aspect=1)  # Reduce this value to compress x-axis more

plt.tight_layout()
# Show the plot
plt.show()

# Plot units on top of probe
max_radius = 50
max_amplitude = 300
amp_to_rad_factor = max_radius / max_amplitude
units = session_responses.unit_ids

stim_contact_site_y = contact_sites[32 - stim_depth]

# Add the stimulation site circle
circle = Circle((0, stim_contact_site_y[1]),
                radius=12, color='#d04f4f', fill=True)
ax.add_patch(circle)

# Create a list of unit data with their radii for sorting
unit_data = []

for unit in units:
    ur = session_responses.get_unit_response(unit)

    scr = ur.get_stim_response(stim_condition[0], stim_condition[1])
    tr = scr.train_response
    z_score = tr.z_score
    modulated = (2 * (1 - norm.cdf(abs(z_score)))) < 0.05

    # modulated = tr.one_samp_p_val < 0.05
    stim_mean_fr = tr.stim_mean_fr

    amplitude = -np.min(ur.primary_channel_template)
    print(amplitude)
    scaled_radius = amplitude * amp_to_rad_factor
    unit_location_y = ur.unit_location[1]
    x_jitter = np.random.normal(0, 5)

    # Store the unit data with radius for sorting later
    unit_data.append(
        (scaled_radius, (x_jitter, unit_location_y), amplitude, stim_mean_fr, modulated))

# Sort the units by scaled_radius (descending  order) so larger units are plotted first
unit_data.sort(key=lambda x: x[0], reverse=True)

# Create a colormap normalization object for the color bar
cnorm = plt.Normalize(vmin=0, vmax=40)  # firing rate from 0 to 80 Hz
sm = plt.cm.ScalarMappable(cmap=cm.gray, norm=cnorm)
sm.set_array([])

# Plot units and adjust face color based on stim_mean_fr
for scaled_radius, (x_jitter, unit_location_y), amplitude, stim_mean_fr, modulated in unit_data:
    # Normalize stim_mean_fr to a value between 0 and 1 for the colormap (0 Hz -> black, 80 Hz -> white)
    normalized_fr = np.clip(stim_mean_fr / 40.0, 0, 1)

    if modulated:
        edgecolor = '#d04f4f'
    else:
        edgecolor = '#878888'

    # Get the grayscale color from the 'gray' colormap
    face_color = cm.gray(normalized_fr)

    # Create and plot the circle
    circle = Circle((x_jitter, unit_location_y), radius=scaled_radius,
                    facecolor=face_color, edgecolor=edgecolor, fill=True)
    ax.add_patch(circle)

# Set axis limits
ax.set_xlim(-70, 70)
ax.set_ylim(-200, 1700)
ax.axis('off')

# Add the color bar, horizontal and positioned below the plot
# cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.05, shrink=0.4)
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.05, shrink=0.5)
cbar.set_label('Stim Firing Rate (Hz)', fontsize=8)

# Add legend for the amplitude (size of circles)
legend_labels = [100, 200, 300]  # Example amplitude labels in µV
legend_circles = [Circle((0, 0), radius=amp_to_rad_factor * label * 0.13, facecolor='white', edgecolor='k')
                  for label in legend_labels]

# ax.legend(legend_circles, [f'{label} µV' for label in legend_labels],
#           handler_map={Circle: HandlerPatch(patch_func=make_legend_circle)},
#           loc='center left', title="Amplitude (µV)", bbox_to_anchor=(1, 0.2),
#           prop={'size': 10}, labelspacing=0.5, borderaxespad=1)

# plt.tight_layout(rect=[0.5, 0.5, 0.5, 0.5])

plt.tight_layout()
# Show the plot
plt.show()

# %%


def filter_data(df):
    # Filter data using the correct comparison for boolean and numerical conditions
    for ch in df['stim_channel'].unique():
        ch_df = df[df['stim_channel'] == ch]
        if len(ch_df['days_relative'].unique()) < 5:
            df = df[df['stim_channel'] != ch]

    df = df[
        (df["baseline_too_slow"] == False) &
        (df['significantly_modulated'] == True) &
        (df["z_score"] > 0) &
        (df["z_score"] < 30) &
        (df['num_spikes'] > 100) &
        (df['stim_current'] < 7) &
        (df['stim_current'] > 2)]

    return df


var1 = "z_score"
animal_ids = ["ICMS92", "ICMS93", "ICMS98", "ICMS100", "ICMS101"]
base_path = Path("C:/data/")
dfs = [load_session_data(animal_id, base_path) for animal_id in animal_ids]
df = pd.concat(dfs, ignore_index=True)
df = filter_data(df)

# Calculate weeks relative
last_week = 4
df['weeks_relative'] = (df['days_relative'] // 7).astype(int)
df = df[df['weeks_relative'] <= last_week]
len_x = len(df['weeks_relative'].unique())

# Filter rows where 'stim_current' equals 'detection_threshold'
df_threshold = df[df['stim_current'] == df['detection_threshold']]

# Calculate median and IQR for each week relative
aggregated_data_var1 = df_threshold.groupby('weeks_relative').apply(
    lambda x: pd.Series({
        'median': x[var1].median(),
        'iqr_low': np.percentile(x[var1], 25),
        'iqr_high': np.percentile(x[var1], 75)
    })
).reset_index()

# Mann-Whitney U test between Week 0 and Week 5
week_0_var1 = df_threshold[df_threshold['weeks_relative'] == 0][var1]
week_last_var1 = df_threshold[df_threshold['weeks_relative'] == 5][var1]
stat, p_val = mannwhitneyu(week_0_var1, week_last_var1, alternative='less')

# Calculate the error bar lengths
lower_error_bar_len = aggregated_data_var1['median'] - \
    aggregated_data_var1['iqr_low']
upper_error_bar_len = aggregated_data_var1['iqr_high'] - \
    aggregated_data_var1['median']

# %%
# Plot median with error bars (IQR)
fig, ax = plt.subplots(figsize=(3, 2.5))  # Fix subplot creation

ax.errorbar(
    aggregated_data_var1['weeks_relative'],  # x-values (weeks)
    aggregated_data_var1['median'],          # y-values (medians)
    yerr=[lower_error_bar_len, upper_error_bar_len],
    fmt='-o',
    capsize=4,
    color='k',
    label="Median with IQR"
)

# Plot individual data points for each animal
for animal_index, animal_id in enumerate(animal_ids):
    df_animal = df_threshold[df_threshold['animal_id'] == animal_id]
    sns.stripplot(
        x='weeks_relative',   # Categorical variable (weeks)
        y=var1,               # Continuous variable (z_score)
        data=df_animal,       # DataFrame
        jitter=0.25,           # Add some jitter for better visibility of overlapping points
        size=3,
        color=f"C{animal_index}",
        label=animal_id,  # Ensure unique label for each animal
    )

# Add significance marker if p-value <= 0.05
if np.round(p_val, 2) <= 0.05 and not aggregated_data_var1.empty:
    last_median = aggregated_data_var1[aggregated_data_var1['weeks_relative']
                                       == 5]['median'].values[0]
    ax.text(5, last_median +
            upper_error_bar_len.iloc[-1] + 1, '*', color='k', fontsize=15, ha='center')

# Add labels and title
ax.set_xlabel('Weeks Relative to First Session', fontsize=8)
ax.set_ylabel('Z-score', fontsize=8)  # Format y-axis label
ax.set_title(
    f'ICMS Train Modulation Score at\n Detection Threshold Current', fontsize=8)

ax.set_xticklabels(np.arange(6), fontsize=8)
ax.tick_params(axis='y', labelsize=8)  # Set y-axis tick font size


# Create a clean legend without duplicates
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels[:-1], handles[:-1]))  # Remove duplicates
ax.legend(unique_labels.values(), unique_labels.keys(), fontsize=5)

# Display plot
plt.tight_layout()
plt.show()

# %% Figure 6 - Pulse response
axs["train_raster"] = fig.add_subplot(gs[0, 0])  # Row 0, Column 0
axs["pulse_raster"] = fig.add_subplot(gs[0, 1])  # Row 0, Column 1
axs["train_fr"] = fig.add_subplot(gs[1, 0])  # Row 1, Column 0
axs["pulse_fr"] = fig.add_subplot(gs[1, 1])  # Row 1, Column 1
# axs["fr"] = fig.add_subplot(gs[2:4, 0])  # Row 2, Column 0
# axs["z_score"] = fig.add_subplot(gs[2:4, 1])  # Row 2, Column 1

pulse_line_offset, train_line_offset = 0, 0
for current in np.arange(3, 6):
    scr = ur.get_stim_response(9, current)
    tr = scr.train_response
    pr = scr.pulse_response

    # Plot pulse raster and train raster
    rpu.plot_pulse_raster(
        axs["pulse_raster"], pr, color_map[current], linewidth_factor, pulse_line_offset)
    rpu.plot_train_raster(
        axs["train_raster"], tr, color_map[current], linewidth_factor, train_line_offset)

    # Plot pulse firing rate and train firing rate
    rpu.plot_pulse_firing_rate(axs["pulse_fr"], pr, color_map[current])
    rpu.plot_train_firing_rate(axs["train_fr"], tr, color_map[current])

    # Update the line offsets after plotting each current
    pulse_line_offset += len(pr.raster_array)
    train_line_offset += len(tr.raster_array)

# Add a single black stimulus bar for the entire plot
axs["train_raster"].plot([0, 700], [train_line_offset +
                         5, train_line_offset + 5], color="black", linewidth=4)
axs["train_fr"].plot([0, 700], [max(tr.firing_rate) + 10,
                     max(tr.firing_rate) + 10], color="black", linewidth=4)

# Adjust y-limits for the plots to accommodate the bar and all the data
axs["train_raster"].set_ylim([0, train_line_offset + 10])
axs["train_fr"].set_ylim([0, max(tr.firing_rate) + 20])

axs["train_fr"].set_title("Stim Train Firing Rate", fontsize=10)


axs["train_raster"].set_ylabel("Trial index")
axs["train_raster"].set_xlabel("Time (ms)")
axs["train_raster"].set_title("Stim Train Rasterplot", fontsize=10)

axs["pulse_raster"].set_ylabel("Pulse index")
axs["pulse_raster"].set_xlabel("Time (ms)")
axs["pulse_raster"].set_title("Individual Pulse Rasterplot", fontsize=10)

# %%

color_map = rpu.get_stim_colormap()
fig, ax = plt.subplots(figsize=(2, 2.5))
for current in np.arange(3, 6):
    scr = ur.get_stim_response(9, current)
    tr = scr.train_response
    pr = scr.pulse_response
    rpu.plot_pulse_firing_rate(ax, pr, color_map[current])


ax.set_title("Stim Pulse-Evoked Firing Rate", fontsize=10)
# Example usage within the plot function
colors = ['C1', 'C2', 'C3']  # Example: 3 different colors for the lines
labels = ['3 µA', '4 µA', '5 µA']  # Labels corresponding to the lines

# Assuming you have your axs['pulse_raster']
add_custom_legend_to_pulse_raster(ax, colors, labels)
ax.set_xlim([0, 10])
ax.set_ylim([0, 240])

add_scale_bars_wvf(ax, h_pos=[-0, 2], v_pos=[-0, 2],
                   h_length=1, v_length=50, v_label_x_offset=0, v_label_y_offset=25,
                   h_label_x_offset=0.5, h_label_y_offset=-5,
                   h_label="1 ms", v_label="50 Hz", font_size=7)

ax.set_axis_off()

# %%


def filter_data(df):
    # Filter data using the correct comparison for boolean and numerical conditions
    for ch in df['stim_channel'].unique():
        ch_df = df[df['stim_channel'] == ch]
        if len(ch_df['days_relative'].unique()) < 5:
            df = df[df['stim_channel'] != ch]

    df = df[
        (df["baseline_too_slow"] == False) &
        # (df['significantly_modulated'] == True) &
        (df['is_pulse_locked'] == True) &
        # (df["z_score"] > 0) &
        # (df["z_score"] < 30) &
        (df['num_spikes'] > 200) &
        (df['stim_current'] < 7) &
        (df['stim_current'] > 2)]

    return df


animal_ids = ["ICMS92", "ICMS93", "ICMS98",
              "ICMS100", "ICMS101"]  # Example animal IDs
base_path = Path("C:/data/")  # Replace with your actual base path

# Load session data for each animal
dfs = [load_session_data(animal_id, base_path) for animal_id in animal_ids]

# Concatenate the DataFrames for all animals
df = pd.concat(dfs, ignore_index=True)
df = filter_data(df)
df['weeks_relative'] = (df['days_relative'] // 7).astype(int)

row_counts = df.groupby(['stim_current', 'weeks_relative']
                        ).size().reset_index(name='row_count')
row_counts
# %% pulse_mean_fr
fig, ax = plt.subplots(figsize=(3, 2.5))

p_values_var1, _ = plot_aggregated_weekly_data_with_iqr(
    df, var1='pulse_mean_fr', last_week=4, ax1=ax)

ax.set_title("ICMS Mean Pulse Firing Rate Across Weeks", fontsize=8)
ax.set_ylabel(f"Firing Rate (Hz)", fontsize=8)
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

# %% Latency
fig, ax = plt.subplots(figsize=(3, 2.5))

p_values_var1, _ = plot_aggregated_weekly_data_with_iqr(
    df, var1='latency', last_week=4, ax1=ax)

ax.set_title("Latency to Peak Across Weeks", fontsize=8)
ax.set_ylabel(f"Latency (ms)", fontsize=8)
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

# %%
var1 = "latency"
animal_ids = ["ICMS92", "ICMS93", "ICMS98", "ICMS100", "ICMS101"]
base_path = Path("C:/data/")
dfs = [load_session_data(animal_id, base_path) for animal_id in animal_ids]
df = pd.concat(dfs, ignore_index=True)
df = filter_data(df)

# Calculate weeks relative
last_week = 4
df['weeks_relative'] = (df['days_relative'] // 7).astype(int)
df = df[df['weeks_relative'] <= last_week]
len_x = len(df['weeks_relative'].unique())

# Filter rows where 'stim_current' equals 'detection_threshold'
df_threshold = df[df['stim_current'] == df['detection_threshold']]

# Calculate median and IQR for each week relative
aggregated_data_var1 = df_threshold.groupby('weeks_relative').apply(
    lambda x: pd.Series({
        'median': x[var1].median(),
        'iqr_low': np.percentile(x[var1], 25),
        'iqr_high': np.percentile(x[var1], 75)
    })
).reset_index()

# Mann-Whitney U test between Week 0 and Week 5
week_0_var1 = df_threshold[df_threshold['weeks_relative'] == 0][var1]
week_last_var1 = df_threshold[df_threshold['weeks_relative'] == 5][var1]
stat, p_val = mannwhitneyu(week_0_var1, week_last_var1, alternative='less')

# Calculate the error bar lengths
lower_error_bar_len = aggregated_data_var1['median'] - \
    aggregated_data_var1['iqr_low']
upper_error_bar_len = aggregated_data_var1['iqr_high'] - \
    aggregated_data_var1['median']

# %%
# Plot median with error bars (IQR)
fig, ax = plt.subplots(figsize=(3, 2.5))  # Fix subplot creation

ax.errorbar(
    aggregated_data_var1['weeks_relative'],  # x-values (weeks)
    aggregated_data_var1['median'],          # y-values (medians)
    yerr=[lower_error_bar_len, upper_error_bar_len],
    fmt='-o',
    capsize=4,
    color='k',
    label="Median with IQR"
)

# Plot individual data points for each animal
for animal_index, animal_id in enumerate(animal_ids):
    df_animal = df_threshold[df_threshold['animal_id'] == animal_id]
    sns.stripplot(
        x='weeks_relative',   # Categorical variable (weeks)
        y=var1,               # Continuous variable (z_score)
        data=df_animal,       # DataFrame
        jitter=0.25,           # Add some jitter for better visibility of overlapping points
        size=3,
        color=f"C{animal_index}",
        label=animal_id,  # Ensure unique label for each animal
    )

# Add significance marker if p-value <= 0.05
if np.round(p_val, 2) <= 0.05 and not aggregated_data_var1.empty:
    last_median = aggregated_data_var1[aggregated_data_var1['weeks_relative']
                                       == last_week]['median'].values[0]
    ax.text(last_week, last_median +
            upper_error_bar_len.iloc[-1] + 1, '*', color='k', fontsize=15, ha='center')

# Add labels and title
ax.set_xlabel('Weeks Relative to First Session', fontsize=8)
ax.set_ylabel('Firing Rate (Hz)', fontsize=8)  # Format y-axis label
ax.set_title(
    f'ICMS Train Modulation Score at\n Detection Threshold Current', fontsize=8)

ax.set_xticklabels(np.arange(last_week + 1), fontsize=8)
ax.tick_params(axis='y', labelsize=8)  # Set y-axis tick font size


# Create a clean legend without duplicates
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels[:-1], handles[:-1]))  # Remove duplicates
ax.legend(unique_labels.values(), unique_labels.keys(), fontsize=5)

# Display plot
plt.tight_layout()
plt.show()

# %% Unit count


def filter_data(df):
    # Filter data using the correct comparison for boolean and numerical conditions
    for ch in df['stim_channel'].unique():
        ch_df = df[df['stim_channel'] == ch]
        if len(ch_df['days_relative'].unique()) < 5:
            df = df[df['stim_channel'] != ch]

    df['weeks_relative'] = (df['days_relative'] // 7).astype(int)

    df = df[
        (df['weeks_relative'] < 5) &
        (df["baseline_too_slow"] == False) &
        (df['significantly_modulated'] == True) &
        (df["z_score"] > 0) &
        (df["z_score"] < 80) &
        # (df["is_pulse_locked"] == True) &
        # (
        #     # If z_score > 0, must be pulse-locked
        #     ((df["z_score"] > 0) & (df["is_pulse_locked"] == True)) |
        #     # Allow negative or zero z_scores without pulse-locked condition
        #     (df["z_score"] < 0)
        # ) &
        (df['num_spikes'] > 200) &
        (df['stim_current'] < 7) &
        (df['stim_current'] > 2)]

    return df


for animal_id in animal_ids:
    base_path = Path("C:/data/")  # Replace with your base path
    original_df = load_session_data(animal_id, base_path)
    df = filter_data(original_df)

    relative_days = df['days_relative'].unique()
    relative_weeks = df['weeks_relative'].unique()

    modulated_unit_counts = []
    all_unit_counts = []
    pulse_locked_unit_counts = []

    for relative_week in relative_weeks:
        original_session_df = original_df[original_df['weeks_relative']
                                          == relative_week]
        session_df = df[df['weeks_relative'] == relative_week]

        pulse_locked_df = session_df[session_df['is_pulse_locked'] == True]

        all_unit_counts.append(len(original_session_df['unit_id'].unique()))
        modulated_unit_counts.append(len(session_df['unit_id'].unique()))
        pulse_locked_unit_counts.append(
            len(pulse_locked_df['unit_id'].unique()))

    plt.scatter(all_unit_counts, label='All')
    plt.scatter(modulated_unit_counts, label='Modulated')
    plt.scatter(pulse_locked_unit_counts, label='Pulse-locked')
    plt.legend()
    plt.xlabel("Session index")
    plt.ylabel("Unit count")
    plt.title(animal_id)

# %%


def plot_animal_unit_counts_by_week(animal_ids, base_path):
    # Dictionary to store aggregated data by count type for each week
    weekly_counts = {
        'All': [[] for _ in range(6)],  # Assuming max 6 weeks
        'Modulated': [[] for _ in range(6)],
        'Pulse-locked': [[] for _ in range(6)]
    }

    for animal_id in animal_ids:
        original_df = load_session_data(animal_id, base_path)
        original_df['weeks_relative'] = original_df['days_relative'] // 7
        df = filter_data(original_df)

        relative_weeks = df['weeks_relative'].unique()

        for relative_week in relative_weeks:
            original_session_df = original_df[original_df['weeks_relative']
                                              == relative_week]
            session_df = df[df['weeks_relative'] == relative_week]
            pulse_locked_df = session_df[session_df['is_pulse_locked'] == True]

            # Append counts to the respective week list in weekly_counts
            weekly_counts['All'][relative_week].append(
                len(original_session_df['unit_id'].unique()))
            weekly_counts['Modulated'][relative_week].append(
                len(session_df['unit_id'].unique()))
            weekly_counts['Pulse-locked'][relative_week].append(
                len(pulse_locked_df['unit_id'].unique()))

    # Plot each count type with different colors and a slight x-offset for clarity
    plt.figure(figsize=(2, 2))
    categories = ['All', 'Modulated', 'Pulse-locked']
    categories = ['All', 'Modulated']

    offsets = [-0.1, 0, 0.1]  # Small offset for each category to avoid overlap

    # Store median values for each category and week
    median_values_by_category = {category: [] for category in categories}

    for i, category in enumerate(categories):
        color = f'C{i}'
        offset = offsets[i]

        # Loop through each week to plot individual counts per animal and calculate medians
        for week in range(len(weekly_counts[category])):
            counts = weekly_counts[category][week]
            week_index = np.full(len(counts), week) + \
                offset  # Offset x values for scatter
            plt.scatter(week_index, counts, color=color, alpha=0.7)

            # Plot the median for the current week at the center of the week and store it
            if len(counts) > 0:
                median_value = np.median(counts)
                median_values_by_category[category].append(median_value)
                plt.plot(week, median_value, marker='o',
                         color=color, markersize=6)
            else:
                median_values_by_category[category].append(
                    np.nan)  # Handle weeks with no data

        # Connect the median points for each category with lines
        plt.plot(range(len(median_values_by_category[category])),
                 median_values_by_category[category], color=color, linewidth=2)

    # Customize legend to display only scatter points
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=category,
                              markerfacecolor=f'C{i}', markersize=6) for i, category in enumerate(categories)]
    plt.legend(handles=legend_elements, fontsize='small', loc='upper right')

    # Plot settings
    plt.xticks(ticks=np.arange(0, len(weekly_counts['All'])), labels=[
               f"W{i}" for i in range(len(weekly_counts['All']))])
    plt.ylabel("Unit Count")
    # plt.title("Median Unit Counts per Week Across All Animals")
    plt.tight_layout()
    plt.show()


# %%
# Usage
base_path = Path("C:/data/")  # Replace with your base path
plot_animal_unit_counts_by_week(animal_ids, base_path)
plt.xlim([-0.5, 4.5])
