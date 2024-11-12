from util.aggregated_weekly_data_plot import plot_aggregated_weekly_data_with_iqr
from batch_process.postprocessing.plot_longitudinal_data_v2 import load_session_data
import seaborn as sns  # For color palette
from scipy.stats import bootstrap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import batch_process.postprocessing.responses.response_plotting_util as rpu
from scipy.stats import ttest_ind
import numpy as np
import spikeinterface.full as si
from pathlib import Path
import pickle
import batch_process.util.template_util as template_util
import batch_process.postprocessing.stim_response_util as stim_response_util

from matplotlib.lines import Line2D


def add_custom_legend_to_pulse_raster(ax, colors, labels, loc="upper left", fontsize=8, alpha=0.8):
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
    custom_lines = [Line2D([0], [0], color=color, lw=4) for color in colors]

    # Add the legend to the axis
    ax.legend(custom_lines, labels, loc=loc, prop={'size': fontsize})


def bootstrap_ci(data, confidence_level=0.95, n_resamples=1000):
    """Computes the bootstrapped confidence interval for the median."""
    data = np.array(data.dropna())  # Ensure no NaN values
    if len(data) == 0:
        return np.nan, np.nan  # Return NaNs if there's no data

    # Define the statistic function for median
    stat_fun = np.median

    # Perform bootstrapping to compute confidence interval
    res = bootstrap((data,), stat_fun, confidence_level=confidence_level,
                    n_resamples=n_resamples, method='basic')

    return res.confidence_interval.low, res.confidence_interval.high


def plot_aggregated_weekly_data_with_bootstrapped_ci(df, var1="z_score", var2="pulse_mean_fr", animal_id="Aggregated", ax1=None, ax2=None):
    color_map = rpu.get_stim_colormap()

    if ax1 is None or ax2 is None:
        raise ValueError("Both ax1 and ax2 must be provided.")

    # Create a new column for 'weeks_relative' by dividing 'days_relative' by 7 and flooring it
    df['weeks_relative'] = (df['days_relative'] // 7).astype(int)

    # Initialize p-values dictionaries
    p_values_var1 = {}
    p_values_var2 = {}

    # Get unique stimulation currents
    stim_currents = sorted(df['stim_current'].unique())

    # Plot for var1 with 95% CI
    for stim_current in stim_currents:
        current_data_var1 = df[df['stim_current'] == stim_current]

        week_0_var1 = current_data_var1[current_data_var1['weeks_relative'] == 0][var1]
        week_last_var1 = current_data_var1[current_data_var1['weeks_relative'] == 5][var1]

        # Perform Mann-Whitney U test
        stat, p_val = mannwhitneyu(
            week_0_var1, week_last_var1, alternative='less')
        p_values_var1[stim_current] = p_val

        aggregated_data_var1 = current_data_var1.groupby('weeks_relative').apply(
            lambda x: pd.Series({
                'median': x[var1].median(),
                'ci_low': bootstrap_ci(x[var1])[0],
                'ci_high': bootstrap_ci(x[var1])[1]
            })
        ).reset_index()

        # Plot the median
        ax1.plot(
            aggregated_data_var1['weeks_relative'],
            aggregated_data_var1['median'],
            marker="o",
            label=f"{stim_current} µA",
            color=color_map[stim_current]
        )

        # Add 95% CI as error bars
        ax1.fill_between(
            aggregated_data_var1['weeks_relative'],
            aggregated_data_var1['ci_low'],
            aggregated_data_var1['ci_high'],
            color=color_map[stim_current],
            alpha=0.2  # Adjust alpha for transparency
        )

        # Add star if p <= 0.05 near the last data point
        if np.round(p_val, 2) <= 0.05:
            last_median = aggregated_data_var1[aggregated_data_var1['weeks_relative']
                                               == 5]['median'].values[0]
            ax1.text(5.1, last_median - 0.5, '*',
                     color=color_map[stim_current], fontsize=20)

    # Replace underscores with spaces for better formatting
    var1_label = var1.replace('_', ' ').capitalize()
    ax1.set_title(f"Median {var1_label.capitalize()
                            } with 95% CI Across Weeks", fontsize=10)
    ax1.set_ylabel(f"Median {var1_label}")
    ax1.set_xlabel("Weeks Relative to First Session")
    ax1.set_xticks(sorted(df['weeks_relative'].unique()))

    # Plot for var2 with 95% CI
    for stim_current in stim_currents:
        current_data_var2 = df[df['stim_current'] == stim_current]

        week_0_var2 = current_data_var2[current_data_var2['weeks_relative'] == 0][var2]
        week_last_var2 = current_data_var2[current_data_var2['weeks_relative'] == 5][var2]

        # Perform Mann-Whitney U test
        stat, p_val = mannwhitneyu(
            week_0_var2, week_last_var2, alternative='less')
        p_values_var2[stim_current] = p_val

        aggregated_data_var2 = current_data_var2.groupby('weeks_relative').apply(
            lambda x: pd.Series({
                'median': x[var2].median(),
                'ci_low': bootstrap_ci(x[var2])[0],
                'ci_high': bootstrap_ci(x[var2])[1]
            })
        ).reset_index()

        # Plot the median
        ax2.plot(
            aggregated_data_var2['weeks_relative'],
            aggregated_data_var2['median'],
            marker="o",
            label=f"{stim_current} µA",
            color=color_map[stim_current]
        )

        # Add 95% CI as error bars
        ax2.fill_between(
            aggregated_data_var2['weeks_relative'],
            aggregated_data_var2['ci_low'],
            aggregated_data_var2['ci_high'],
            color=color_map[stim_current],
            alpha=0.2
        )

        # Add star if p <= 0.05 near the last data point
        if np.round(p_val, 2) <= 0.05:
            last_median = aggregated_data_var2[aggregated_data_var2['weeks_relative']
                                               == 5]['median'].values[0]
            ax2.text(5.1, last_median - 0.5, '*',
                     color=color_map[stim_current], fontsize=20)

    # Replace underscores with spaces for better formatting
    var2_label = var2.replace('_', ' ').capitalize()
    ax1.set_xlim([-0.4, 5.4])
    ax2.set_xlim([-0.4, 5.4])
    ax2.set_title(
        f"Median Pulse Window Firing Rate with 95% CI Across Weeks", fontsize=10)
    ax2.set_ylabel(f"Median {var2_label}")
    ax2.set_xlabel("Weeks Relative to First Session")
    ax2.set_xticks(sorted(df['weeks_relative'].unique()))

    # Add legend if needed
    ax1.legend(loc="best")
    ax2.legend(loc="best")

    plt.tight_layout()
    return p_values_var1, p_values_var2


def plot_aggregated_weekly_data(df, var1="z_score", var2="pulse_mean_fr", animal_id="Aggregated", ax1=None, ax2=None):
    color_map = rpu.get_stim_colormap()
    if ax1 is None or ax2 is None:
        raise ValueError("Both ax1 and ax2 must be provided.")

    # Create a new column for 'weeks_relative' by dividing 'days_relative' by 7 and flooring it
    df['weeks_relative'] = (df['days_relative'] // 7).astype(int)

    # Get unique stimulation currents
    stim_currents = sorted(df['stim_current'].unique())

    # Prepare to store p-values for significance tests
    p_values_var1 = {}
    p_values_var2 = {}

    # Plot for var1 with shading
    for stim_current in stim_currents:
        current_data_var1 = df[df['stim_current'] == stim_current]

        # Extract data points for week 0 and week 5 (or last week)
        week_0_var1 = current_data_var1[current_data_var1['weeks_relative'] == 0][var1]
        week_last_var1 = current_data_var1[current_data_var1['weeks_relative'] == 5][var1]

        # Perform Mann-Whitney U test
        stat, p_val = mannwhitneyu(
            week_0_var1, week_last_var1, alternative='less')
        p_values_var1[stim_current] = p_val

        # Aggregate data for plotting
        aggregated_data_var1 = current_data_var1.groupby(
            ['weeks_relative'])[var1].agg(['mean', 'sem']).reset_index()

        # Plot var1
        ax1.plot(
            aggregated_data_var1['weeks_relative'],
            aggregated_data_var1['mean'],
            marker="o",
            label=f"{stim_current} µA",
            color=color_map[stim_current]
        )
        # Add shading (fill between the mean ± SEM)
        ax1.fill_between(
            aggregated_data_var1['weeks_relative'],
            aggregated_data_var1['mean'] - aggregated_data_var1['sem'],
            aggregated_data_var1['mean'] + aggregated_data_var1['sem'],
            color=color_map[stim_current],
            alpha=0.2  # Adjust alpha for transparency
        )

        # Add star if p <= 0.05 near the last data point
        if np.round(p_val, 2) <= 0.05:
            last_mean = aggregated_data_var1[aggregated_data_var1['weeks_relative']
                                             == 5]['mean'].values[0]
            ax1.text(5.1, last_mean - 0.5, '*',
                     color=color_map[stim_current], fontsize=20)

    # Replace underscores with spaces for better formatting
    var1_label = var1.replace('_', ' ').capitalize()
    ax1.set_title(f"Mean Z-Score Across Weeks", fontsize=10)
    ax1.set_ylabel(f"Z-Score")
    ax1.set_xlabel("Weeks Relative to First Session")
    ax1.set_xticks(sorted(df['weeks_relative'].unique()))

    # Plot for var2 with shading
    for stim_current in stim_currents:
        current_data_var2 = df[df['stim_current'] == stim_current]

        # Extract data points for week 0 and week 5 (or last week)
        week_0_var2 = current_data_var2[current_data_var2['weeks_relative'] == 0][var2]
        week_last_var2 = current_data_var2[current_data_var2['weeks_relative'] == 5][var2]

        # Perform Mann-Whitney U test
        stat, p_val = mannwhitneyu(
            week_0_var2, week_last_var2, alternative='less')
        p_values_var2[stim_current] = p_val

        # Aggregate data for plotting
        aggregated_data_var2 = current_data_var2.groupby(
            ['weeks_relative'])[var2].agg(['mean', 'sem']).reset_index()

        # Plot var2
        ax2.plot(
            aggregated_data_var2['weeks_relative'],
            aggregated_data_var2['mean'],
            marker="o",
            label=f"{stim_current} µA",
            color=color_map[stim_current]
        )
        # Add shading (fill between the mean ± SEM)
        ax2.fill_between(
            aggregated_data_var2['weeks_relative'],
            aggregated_data_var2['mean'] - aggregated_data_var2['sem'],
            aggregated_data_var2['mean'] + aggregated_data_var2['sem'],
            color=color_map[stim_current],
            alpha=0.2  # Adjust alpha for transparency
        )

        # Add star if p <= 0.05 near the last data point
        if np.round(p_val, 2) <= 0.05:
            last_mean = aggregated_data_var2[aggregated_data_var2['weeks_relative']
                                             == 5]['mean'].values[0]
            ax2.text(5.1, last_mean - 0.5, '*',
                     color=color_map[stim_current], fontsize=20)

    # Replace underscores with spaces for better formatting
    var2_label = var2.replace('_', ' ').capitalize()
    ax2.set_xlim([-0.4, 5.4])
    ax1.set_xlim([-0.4, 5.4])
    ax2.set_title(f"Mean Pulse Window Firing Rate Across Weeks", fontsize=10)
    ax2.set_ylabel(f"Firing Rate (Hz)")
    ax2.set_xlabel("Weeks Relative to First Session")
    ax2.set_xticks(sorted(df['weeks_relative'].unique()))

    return p_values_var1, p_values_var2


# %%
color_map = rpu.get_stim_colormap()
linewidth_factor = 30
data_folder = "C:\\data\\ICMS92\Behavior\\01-Sep-2023"
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

# colors = ['C1', 'C2']  # Example: 3 different colors for the lines
# labels = ['3 µA', '4 µA']  # Labels corresponding to the lines

# Assuming you have your axs['pulse_raster']
add_custom_legend_to_pulse_raster(axs['pulse_fr'], colors, labels)

axs["pulse_fr"].set_ylabel("Firing Rate (Hz)")
axs["pulse_fr"].set_xlabel("Time (ms)")
axs["pulse_fr"].set_title("Pulse Firing Rate", fontsize=10)

# p_values_var1, p_values_var2 = plot_aggregated_weekly_data_with_bootstrapped_ci(
#     df, var1='abs_t_val', ax1=axs["fr"], ax2=axs["z_score"])

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

# %%%
fig, ax = plt.subplots()  # Use plt.subplots() to get both fig and ax
# Assuming rpu.plot_template requires an axis to plot on
rpu.plot_primary_channel_template(stim_response=scr, y_negative=-220, ax=ax)

# %%
fig = plt.figure(figsize=(8, 10))
# 3 rows, 2 columns with equal widths
gs = fig.add_gridspec(4, 2, width_ratios=[1, 1])
axs = {}
# Example of adding subplots to this grid
axs["train_raster"] = fig.add_subplot(gs[0, 0])  # Row 0, Column 0
axs["pulse_raster"] = fig.add_subplot(gs[0, 1])  # Row 0, Column 1
axs["train_fr"] = fig.add_subplot(gs[1, 0])  # Row 1, Column 0
axs["pulse_fr"] = fig.add_subplot(gs[1, 1])  # Row 1, Column 1
axs["fr"] = fig.add_subplot(gs[2:4, 0])  # Row 2, Column 0
axs["z_score"] = fig.add_subplot(gs[2:4, 1])  # Row 2, Column 1
p_values_var1, p_values_var2 = plot_aggregated_weekly_data_with_iqr(
    df, var1='z_score', var2='latency', ax1=axs["fr"], ax2=axs["z_score"])

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

# %%


def filter_data(df):
    # Filter data using the correct comparison for boolean and numerical conditions
    for ch in df['stim_channel'].unique():
        ch_df = df[df['stim_channel'] == ch]
        if len(ch_df['days_relative'].unique()) < 5:
            df = df[df['stim_channel'] != ch]

    df = df.copy()  # Make an explicit copy of the DataFrame
    df.loc[:, "abs_t_val"] = df["t_val"].abs()  # Set the abs_t_val column

    df = df[
        # (df['cell_type'] == "Narrow Interneuron Cell") &
        (df["baseline_too_slow"] == False) &
        (df['modulated'] == True) &
        (df["t_val"] < 60) &

        # Ensure t_val > 0 has specific conditions
        # (
        #     ((df["t_val"] > 0) &
        #      (df["is_pulse_locked"] == True) &
        #      # If t_val > 0, must be pulse-locked and have more than 150 spikes
        #      (df["num_spikes"] > 150)) |

        #     # Allow t_val <= 0 without pulse-locked or num_spikes conditions
        #     (df["t_val"] <= 0)
        # ) &

        (df['stim_current'] < 7) &
        (df['stim_current'] > 2)
    ]

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

fig = plt.figure(figsize=(12, 5))
# 3 rows, 2 columns with equal widths
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
axs = {}
# Example of adding subplots to this grid

axs["fr"] = fig.add_subplot(gs[0, 0])  # Row 2, Column 0
axs["z_score"] = fig.add_subplot(gs[0, 1])  # Row 2, Column 1
p_values_var1, p_values_var2 = plot_aggregated_weekly_data_with_iqr(
    df, var1='abs_t_val', ax1=axs["fr"], ax2=axs["z_score"])

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()
# %% Stats


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

# fig = plt.figure(figsize=(12, 5))
fig = plt.figure(figsize=(10, 5))
# 3 rows, 2 columns with equal widths
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
axs = {}
# Example of adding subplots to this grid

axs["fr"] = fig.add_subplot(gs[0, 0])  # Row 2, Column 0
axs["z_score"] = fig.add_subplot(gs[0, 1])  # Row 2, Column 1
p_values_var1, p_values_var2 = plot_aggregated_weekly_data_with_iqr(
    df, var1='z_score', ax1=axs["fr"], ax2=axs["z_score"])

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

# %%

# Set up color palette for weeks
palette = sns.color_palette("deep", 6)  # 6 colors for 6 weeks

for current in np.arange(3, 8):
    plt.figure()

    all_medians = {}  # Dictionary to store median firing rates for each week

    for week in np.arange(0, 6):
        week_df = df[(df['stim_current'] == current) &
                     (df['weeks_relative'] == week)]

        all_frs = []  # List to store firing rate arrays for each row

        for index, row in week_df.iterrows():
            all_frs.append(row['firing_rate'])  # Append each row's firing rate

        # Ensure that there is data to compute the median
        if len(all_frs) > 0:
            # Calculate the median firing rate across trials
            median_fr = np.median(all_frs, axis=0)
            all_medians[week] = median_fr
            # Plot the median firing rate for the current week
            plt.plot(row['fr_times'], median_fr, label=f"Week {
                     week}", color=palette[week])

    plt.legend()
    plt.title(f"Median Firing Rate for Stim Current: {current} µA")
    plt.xlabel("Time (ms)")
    plt.ylabel("Median Firing Rate (Hz)")
    plt.show()
# %%


# Set up color palette for currents
# 5 colors for 5 currents (2, 3, 4, 5, 6 µA)
palette = sns.color_palette("deep", len(np.arange(3, 7)))

plt.figure()

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
        plt.plot(row['fr_times'], mean_fr, color=f'C{idx+1}',  label=f"{
                 current} µA")

        print(current)

# Add plot details
plt.legend()
plt.title("Mean Firing Rate Across All Weeks for Different Stimulation Currents")
plt.xlabel("Time (ms)")
plt.ylabel("Mean Firing Rate (Hz)")
plt.show()

# %%

# Set up color palette for currents
# 5 colors for 5 currents (2, 3, 4, 5, 6 µA)
palette = sns.color_palette("husl", len(np.arange(3, 7)))

plt.figure()

# Loop through each stimulation current (2 µA to 6 µA)
for idx, current in enumerate(np.arange(3, 7)):
    all_frs_pos = []  # List to store firing rates where t_val > 0
    all_frs_neg = []  # List to store firing rates where t_val < 0

    # Loop through all weeks to aggregate firing rates for the current
    for week in np.arange(0, 6):
        week_df = df[(df['stim_current'] == current) &
                     (df['weeks_relative'] == week)]

        # Separate firing rates based on t_val
        week_df_pos = week_df[week_df['t_val'] > 0]
        week_df_neg = week_df[week_df['t_val'] < 0]

        # Collect firing rates where t_val > 0
        for index, row in week_df_pos.iterrows():
            all_frs_pos.append(row['firing_rate'])

        # Collect firing rates where t_val < 0
        for index, row in week_df_neg.iterrows():
            all_frs_neg.append(row['firing_rate'])

    # Ensure that there is data to compute the mean
    if len(all_frs_pos) > 0:
        # Calculate the mean firing rate where t_val > 0
        mean_fr_pos = np.mean(all_frs_pos, axis=0)
        # Plot the mean firing rate for t_val > 0 with a unique color for each current
        plt.plot(row['fr_times'], mean_fr_pos, label=f"{
                 current} µA (t_val > 0)", linestyle='-', color=palette[idx])

    if len(all_frs_neg) > 0:
        # Calculate the mean firing rate where t_val < 0
        mean_fr_neg = np.mean(all_frs_neg, axis=0)
        # Plot the mean firing rate for t_val < 0 with a dashed line for each current
        plt.plot(row['fr_times'], mean_fr_neg, label=f"{
                 current} µA (t_val < 0)", linestyle='--', color=palette[idx])

# Add plot details
plt.legend()
plt.title("Mean Firing Rate for t_val > 0 and t_val < 0 Across Currents")
plt.xlabel("Time (ms)")
plt.ylabel("Mean Firing Rate (Hz)")
plt.show()

# %%

# Set up color palette for weeks
palette = sns.color_palette("deep", 6)  # 6 colors for 6 weeks

for current in np.arange(3, 7):
    plt.figure()

    # Loop through each week (0 to 5)
    for week in np.arange(0, 6):
        week_df = df[(df['stim_current'] == current) &
                     (df['weeks_relative'] == week)]

        all_frs_pos = []  # List to store firing rates where t_val > 0
        all_frs_neg = []  # List to store firing rates where t_val < 0

        # Separate firing rates based on t_val
        for index, row in week_df.iterrows():
            if row['t_val'] > 0:
                # Append each row's firing rate where t_val > 0
                all_frs_pos.append(row['firing_rate'])
            else:
                # Append each row's firing rate where t_val < 0
                all_frs_neg.append(row['firing_rate'])

        # Plot median firing rate for t_val > 0
        if len(all_frs_pos) > 0:
            # Calculate the median firing rate across trials (t_val > 0)
            median_fr_pos = np.median(all_frs_pos, axis=0)
            plt.plot(row['fr_times'], median_fr_pos, label=f"Week {
                     week} (t_val > 0)", color=palette[week], linestyle='-')

        # Plot median firing rate for t_val < 0
        if len(all_frs_neg) > 0:
            # Calculate the median firing rate across trials (t_val < 0)
            median_fr_neg = np.median(all_frs_neg, axis=0)
            plt.plot(row['fr_times'], median_fr_neg, label=f"Week {
                     week} (t_val < 0)", color=palette[week], linestyle='--')

    # Add plot details
    plt.legend()
    plt.title(f"Median Firing Rate for Stim Current: {current} µA")
    plt.xlabel("Time (ms)")
    plt.ylabel("Median Firing Rate (Hz)")
    plt.show()

# %%

# Set up color palette for weeks
palette = sns.color_palette("deep", 7)  # 6 colors for 6 weeks

# Initialize figure
plt.figure()

# Loop through each week (0 to 5) and aggregate data across all currents
for week in np.arange(0, 6):
    all_frs_pos = []  # List to store firing rates where t_val > 0 across all currents
    all_frs_neg = []  # List to store firing rates where t_val < 0 across all currents

    # Loop through each current (2 to 6 µA)
    for current in np.arange(2, 7):
        week_df = df[(df['stim_current'] == current) &
                     (df['weeks_relative'] == week)]

        # Separate firing rates based on t_val
        for index, row in week_df.iterrows():
            if row['t_val'] > 0:
                # Append each row's firing rate where t_val > 0
                all_frs_pos.append(row['firing_rate'])
            else:
                # Append each row's firing rate where t_val < 0
                all_frs_neg.append(row['firing_rate'])

    # Plot median firing rate for t_val > 0 (color-coded by week)
    if len(all_frs_pos) > 0:
        # Calculate the median firing rate (t_val > 0)
        median_fr_pos = np.mean(all_frs_pos, axis=0)
        plt.plot(row['fr_times'], median_fr_pos, label=f"Week {
                 week}", color=palette[week], linestyle='-')

    # Plot median firing rate for t_val < 0 (no legend for dashed lines)
    if len(all_frs_neg) > 0:
        # Calculate the median firing rate (t_val < 0)
        median_fr_neg = np.mean(all_frs_neg, axis=0)
        plt.plot(row['fr_times'], median_fr_neg,
                 color=palette[week], linestyle='--')

# Add plot details
plt.legend()  # Legend only for solid lines (t_val > 0)
plt.title("Mean Firing Rate Averaged Across Currents")
plt.xlabel("Time (ms)")
plt.ylabel("Mean Firing Rate (Hz)")
plt.show()

# %%

# Set up color palette for weeks
palette = sns.color_palette("deep", 7)  # 6 colors for 6 weeks

# Initialize figure
plt.figure()

# Loop through each week (0 to 5) and aggregate data across all currents, without separating t_val
for week in np.arange(0, 6):
    all_frs = []  # List to store firing rates across all currents and all t_val

    # Loop through each current (2 to 6 µA)
    for current in np.arange(2, 10):
        week_df = df[(df['stim_current'] == current) &
                     (df['weeks_relative'] == week)]

        # Collect all firing rates regardless of t_val
        for index, row in week_df.iterrows():
            all_frs.append(row['firing_rate'])  # Append each row's firing rate

    # Plot mean firing rate for the week (color-coded by week)
    # Calculate the mean firing rate across all trials
    mean_fr = np.mean(all_frs, axis=0)
    plt.plot(row['fr_times'], mean_fr, label=f"Week {
             week}", color=palette[week], linestyle='-', linewidth=2)

# Add plot details
plt.legend()  # Legend showing each week
plt.title("Mean Firing Rate Averaged Across Currents")
plt.xlabel("Time (ms)")
plt.ylabel("Mean Firing Rate (Hz)")
plt.show()

# %% Show stage2 manually curated unit vs merge stage final unit
data_folder = "C:\\data\\ICMS92\Behavior\\01-Sep-2023"

save_folder = Path(data_folder) / "batch_sort"

analyzer = si.load_sorting_analyzer(
    folder=save_folder / "stage2/stage2_analyzer.zarr")
wvf_ext = analyzer.load_extension('waveforms')

# %% Merge
save_folder = Path(data_folder) / "batch_sort"

analyzer = si.load_sorting_analyzer(
    folder=save_folder / "merge/hmerge_analyzer_curated.zarr")

wvf_ext = analyzer.load_extension('waveforms')


# %%


# Stimulus timing parameters
timing_params = file_util.load_timing_params(
    path="batch_process/postprocessing/timing_params.json")
stim_data = stim_response_util.get_stim_data(data_folder)

stim_timestamps = np.array(
    [item for sublist in stim_data.timestamps for item in sublist])

window_samples = timing_params['pulse_post_samples']
stim_spikes = []
stim_spike_indices = []

# Check which spikes occur within the stimulus window
for idx, spike_ts in enumerate(spike_train):
    is_stim_related = np.any(
        (spike_ts > stim_timestamps) & (spike_ts <= stim_timestamps + window_samples))

    if is_stim_related:
        stim_spikes.append(spike_ts)
        stim_spike_indices.append(idx)

stim_spikes = np.array(stim_spikes)
stim_spike_indices = np.array(stim_spike_indices)

# Loop over each unit to plot stimulus-related and non-stimulus-related spikes
for unit_id in analyzer.unit_ids[0:5]:
    # Get waveforms and spike train for the current unit
    wvfs = template_util.get_unit_primary_ch_wvfs(analyzer, unit_id)
    spike_train = analyzer.sorting.get_unit_spike_train(unit_id)

    # Initialize lists for stimulus-related and non-stimulus-related spikes
    stim_spike_wvfs = []
    non_stim_spike_wvfs = []

    # Split the waveforms into stimulus-related and non-stimulus-related categories
    for i, spike_ts in enumerate(spike_train):
        if i in stim_spike_indices:
            # Append stimulus-related waveforms
            stim_spike_wvfs.append(wvfs[i])
        else:
            # Append non-stimulus-related waveforms
            non_stim_spike_wvfs.append(wvfs[i])

    # Convert lists to arrays for plotting
    stim_spike_wvfs = np.array(stim_spike_wvfs)
    non_stim_spike_wvfs = np.array(non_stim_spike_wvfs)

    # Handle edge cases where there might be no stim or non-stim spikes
    if stim_spike_wvfs.shape[0] == 0:
        print(f"No stimulus-related spikes found for unit {unit_id}")
        continue

    if non_stim_spike_wvfs.shape[0] == 0:
        print(f"No non-stimulus-related spikes found for unit {unit_id}")
        continue

    plt.figure(figsize=(12, 6))

    # Plot stimulus-related spikes (left subplot)
    plt.subplot(1, 2, 1)
    if stim_spike_wvfs.size > 0:
        # Plot all available stim-related waveforms in red
        plt.plot(stim_spike_wvfs[300:500, :].T, 'r', alpha=0.5, linewidth=0.5)
    plt.title(f"Stimulus-Related Spikes for Unit {unit_id}")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude (µV)")
    plt.ylim([-600, 300])

    # Plot non-stimulus-related spikes (right subplot)
    plt.subplot(1, 2, 2)
    if non_stim_spike_wvfs.size > 0:
        # Plot all available non-stim waveforms in black
        plt.plot(non_stim_spike_wvfs[300:500, :].T,
                 'k', alpha=0.5, linewidth=0.5)
    plt.title(f"Non-Stimulus-Related Spikes for Unit {unit_id}")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude (µV)")
    plt.ylim([-600, 300])

    plt.tight_layout()
    plt.show()
    # %%
data_folder = "C:\\data\\ICMS92\Behavior\\30-Aug-2023"
save_folder = Path(data_folder) / "batch_sort"
# Load units from stage2 (manually curated) and merge stage
stage2_analyzer = si.load_sorting_analyzer(
    folder=save_folder / "stage2/stage2_analyzer.zarr")
merge_analyzer = si.load_sorting_analyzer(
    folder=save_folder / "merge/hmerge_analyzer_curated.zarr")

# Iterate over the first 5 units in stage2
for stage2_unit_id in stage2_analyzer.unit_ids[0:10]:
    # Get the spike train for the stage2 unit
    stage2_spike_train = stage2_analyzer.sorting.get_unit_spike_train(
        stage2_unit_id)

    best_match = None
    max_common_spikes = 0

    # Iterate over all units in the merge stage to find the best match
    for merge_unit_id in merge_analyzer.unit_ids:
        # Get the spike train for the merge unit
        merge_spike_train = merge_analyzer.sorting.get_unit_spike_train(
            merge_unit_id)

        # Find common spikes between stage2 and merge unit based on timestamps
        common_spikes = np.intersect1d(
            stage2_spike_train, merge_spike_train, assume_unique=True)

        # Track the merge unit with the most common spike timestamps
        if len(common_spikes) > max_common_spikes:
            max_common_spikes = len(common_spikes)
            best_match = merge_unit_id

    # Print out the best matching unit in merge for the current stage2 unit
    print(f"Stage2 Unit {stage2_unit_id} matches Merge Unit {
          best_match} with {max_common_spikes} common spikes")

# %%


def get_stim_and_nonstim_wvfs(wvfs, spike_train, max_threshold=300):
    # Initialize lists for stimulus-related and non-stimulus-related spikes
    stim_spike_wvfs = []
    non_stim_spike_wvfs = []

    # Split the waveforms into stimulus-related and non-stimulus-related categories
    for i, spike_ts in enumerate(spike_train):
        # Filter by the threshold (max value must be <= max_threshold)
        # Use absolute value for both positive and negative thresholds
        if np.max(np.abs(wvfs[i])) <= max_threshold:
            if i in stim_spike_indices:
                # Append stimulus-related waveforms
                stim_spike_wvfs.append(wvfs[i])
            else:
                # Append non-stimulus-related waveforms
                non_stim_spike_wvfs.append(wvfs[i])

    # Convert lists to arrays for plotting
    stim_spike_wvfs = np.array(stim_spike_wvfs)
    non_stim_spike_wvfs = np.array(non_stim_spike_wvfs)

    return stim_spike_wvfs, non_stim_spike_wvfs


# %%

# Assuming you have the matched pairs of units between stage2 and merge
matched_pairs = [(2, 0),  (6, 7), (8, 1)]

matched_pairs = [(2, 0), (7, 3), (8, 4), (9, 5)]
matched_pairs = [(2, 0)]
# matched_pairs = [(10, 6)]


# Function to add scale bars
def add_scale_bars(ax, h_length, v_length, h_label="1 ms", v_label="100 µV"):
    """ Adds horizontal and vertical scale bars to the axes. """
    h_pos = (10, -400)  # Position of horizontal scale bar
    v_pos = (10, -400)  # Position of vertical scale bar

    # Horizontal scale bar (time)
    ax.plot([h_pos[0], h_pos[0] + h_length], [h_pos[1], h_pos[1]], 'k-', lw=2)
    ax.text(h_pos[0] + h_length / 2, h_pos[1] -
            50, h_label, ha='center', fontsize=8)

    # Vertical scale bar (amplitude)
    ax.plot([v_pos[0], v_pos[0]], [v_pos[1], v_pos[1] + v_length], 'k-', lw=2)
    ax.text(v_pos[0] - 4, v_pos[1] + v_length / 2, v_label,
            va='center', fontsize=8, rotation='vertical')


# Iterate over the matched pairs of units
for stage2_unit_id, merge_unit_id in matched_pairs:
    # Retrieve waveforms for the stage2 unit
    stage2_wvfs = template_util.get_unit_primary_ch_wvfs(
        stage2_analyzer, stage2_unit_id)
    stage2_spike_train = stage2_analyzer.sorting.get_unit_spike_train(
        stage2_unit_id)

    # Retrieve waveforms for the merge unit
    merge_wvfs = template_util.get_unit_primary_ch_wvfs(
        merge_analyzer, merge_unit_id)
    merge_spike_train = merge_analyzer.sorting.get_unit_spike_train(
        merge_unit_id)

    # Get stimulus-related and non-stimulus-related waveforms
    stim_spike_wvfs_stage2, non_stim_spike_wvfs_stage2 = get_stim_and_nonstim_wvfs(
        stage2_wvfs, stage2_spike_train)
    stim_spike_wvfs_merge, non_stim_spike_wvfs_merge = get_stim_and_nonstim_wvfs(
        merge_wvfs, merge_spike_train)

    min_len = np.min([len(stim_spike_wvfs_stage2), len(non_stim_spike_wvfs_stage2),
                      len(stim_spike_wvfs_merge),  len(non_stim_spike_wvfs_merge)])

    # Set up 2x2 grid for subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    alpha = 0.6
    linewidth = 0.2

    wvf_range = np.arange(0, np.min([min_len, 1500]))

    # Plot stimulus-related waveforms for stage2 (top-left)
    axs[0, 0].plot(stim_spike_wvfs_stage2[wvf_range, :].T,
                   'C3', alpha=alpha, linewidth=linewidth)
    axs[0, 0].set_title(f"Raw unit waveforms: post-stim")
    axs[0, 0].set_ylim([-500, 300])
    axs[0, 0].axis('off')  # Hide axis

    # Plot non-stimulus-related waveforms for stage2 (bottom-left)
    axs[1, 0].plot(non_stim_spike_wvfs_stage2[wvf_range, :].T,
                   'k', alpha=alpha, linewidth=linewidth)
    axs[1, 0].set_title(f"Raw unit waveforms: spontaneous")
    axs[1, 0].set_ylim([-500, 300])
    axs[1, 0].axis('off')  # Hide axis
    # add_scale_bars(axs[1, 0], h_length=50, v_length=100)  # Add scale bars

    # Plot stimulus-related waveforms for merge (top-right)
    axs[0, 1].plot(stim_spike_wvfs_merge[wvf_range, :].T,
                   'C3', alpha=alpha, linewidth=linewidth)
    axs[0, 1].set_title(f"Processed unit waveforms: post-stim")
    axs[0, 1].set_ylim([-500, 300])
    axs[0, 1].axis('off')  # Hide axis
    # 50 samples for time, 100 µV for amplitude
    add_scale_bars(axs[1, 0], h_length=30, v_length=100)
    # add_scale_bars(axs[0, 1], h_length=50, v_length=100)  # Add scale bars

    # Plot non-stimulus-related waveforms for merge (bottom-right)
    axs[1, 1].plot(non_stim_spike_wvfs_merge[wvf_range, :].T,
                   'k', alpha=alpha, linewidth=linewidth)
    axs[1, 1].set_title(f"Processed unit waveforms: spontaneous")
    axs[1, 1].set_ylim([-500, 300])
    axs[1, 1].axis('off')  # Hide axis
    # add_scale_bars(axs[1, 1], h_length=50, v_length=100)  # Add scale bars

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()
