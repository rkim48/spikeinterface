import os
import numpy as np
import matplotlib.pyplot as plt
import batch_process.util.file_util as file_util
from pathlib import Path
import pandas as pd
from datetime import datetime
import batch_process.postprocessing.responses.response_plotting as plot_util
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import linregress


def process_data_folder(folder):
    """Process a single data folder and return its DataFrame."""
    print(f"Processing data folder {folder}...")
    save_folder = os.path.join(folder, "batch_sort")
    if not save_folder:
        return None

    animalID = file_util.get_animal_id(save_folder)
    date_str = file_util.get_date_str(save_folder)

    df_path = Path(save_folder) / "stim_condition_results.csv"
    if not os.path.exists(df_path):
        return None

    df = pd.read_csv(df_path)
    df["date_str"] = date_str
    df["date"] = datetime.strptime(date_str, "%d-%b-%Y")
    return df


def load_session_data(animal_id, base_path):
    """
    Loads the session data for a given animal ID.

    Args:
        animal_id (str): The animal ID to filter by.
        base_path (Path): The base directory where session data is stored.

    Returns:
        pd.DataFrame: Combined DataFrame with session data for the animal.
    """
    if (animal_id == "ICMS92") or (animal_id == "ICMS93"):
        session_files = base_path.glob(
            f"{animal_id}/Behavior/*/batch_sort/stim_condition_results.csv")
    else:
        session_files = base_path.glob(
            f"{animal_id}/*/batch_sort/stim_condition_results.csv")

    df_list = []
    for session_file in session_files:
        df = pd.read_csv(session_file)
        session_name = session_file.parent.parent.name
        # Extract session name from the path
        df["session"] = session_file.parent.parent.name
        session_date = pd.to_datetime(session_name, format="%d-%b-%Y")
        df["session_date"] = session_date
        df_list.append(df)

    # Combine all session DataFrames
    combined_df = pd.concat(df_list, ignore_index=True)

    # Order by session date
    combined_df = combined_df.sort_values(
        by="session_date").reset_index(drop=True)

    # Calculate days relative to the first session (1-indexed)
    combined_df["days_relative"] = (
        combined_df["session_date"] - combined_df["session_date"].min()).dt.days + 1

    return combined_df


def plot_longitudinal_data(animal_id, base_path):
    """
    Plots longitudinal data of Z-scores and Pulse Mean FR across sessions.

    Args:
        animal_id (str): The animal ID to plot.
        base_path (Path): The base directory where session data is stored.
    """
    # Load data
    df = load_session_data(animal_id, base_path)

    # Filter data
    df = df[(df["include"]) & (df["z_score"] > 0)
            & (df["is_pulse_locked"] == True)]

    # Calculate median Z-scores and Pulse Mean FR
    median_z_scores = df.groupby(["days_relative", "stim_channel", "stim_current"])[
        "z_score"].median().reset_index()
    median_pulse_fr = (
        df.groupby(["days_relative", "stim_channel", "stim_current"])[
            "pulse_mean_fr"].median().reset_index()
    )

    # Get unique channels and currents
    stim_channels = sorted(median_z_scores["stim_channel"].unique())
    stim_currents = sorted(median_z_scores["stim_current"].unique())

    # Create subplots
    num_channels = len(stim_channels)
    fig, axes = plt.subplots(num_channels, 2, figsize=(
        12, 3 * num_channels), sharex=True)

    if num_channels == 1:
        axes = [axes]  # Ensure axes is iterable if only one subplot

    for i, stim_channel in enumerate(stim_channels):
        # Plot Z-Scores
        ax_z = axes[i, 0] if num_channels > 1 else axes[0]
        channel_data_z = median_z_scores[median_z_scores["stim_channel"]
                                         == stim_channel]

        for stim_current in stim_currents:
            current_data_z = channel_data_z[channel_data_z["stim_current"]
                                            == stim_current]
            ax_z.plot(
                current_data_z["days_relative"], current_data_z["z_score"], marker="o", label=f"{stim_current} µA"
            )

        ax_z.set_title(f"Channel {stim_channel} - Z-Scores")
        ax_z.set_ylabel("Median Z-Score")
        ax_z.set_xlabel("Days Relative to First Session")

        # Plot Pulse Mean FR
        ax_fr = axes[i, 1] if num_channels > 1 else axes[1]
        channel_data_fr = median_pulse_fr[median_pulse_fr["stim_channel"]
                                          == stim_channel]

        for stim_current in stim_currents:
            current_data_fr = channel_data_fr[channel_data_fr["stim_current"]
                                              == stim_current]
            ax_fr.plot(
                current_data_fr["days_relative"],
                current_data_fr["pulse_mean_fr"],
                marker="o",
                label=f"{stim_current} µA",
            )

        ax_fr.set_title(f"Channel {stim_channel} - Pulse Mean FR")
        ax_fr.set_ylabel("Median Pulse FR (Hz)")
        ax_fr.set_xlabel("Days Relative to First Session")

    # Create a single legend for all subplots
    handles, labels = ax_z.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(
        0.85, 0.5), title="Stimulation Current")

    plt.suptitle(animal_id)
    # Adjust the right padding to make space for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


def plot_firing_rate_comparison(df, animal_id):
    """
    Plots pre-stim and stim firing rates as a paired bar plot.
    Each session is represented on the x-axis, and the y-axis shows firing rates.

    Args:
        df (pd.DataFrame): DataFrame containing pre-stim and stim firing rates.
    """
    # Melt the DataFrame to have a long format
    df_melted = df.melt(
        id_vars=["unit_id", "stim_channel", "stim_current", "days_relative"],
        value_vars=["pre_stim_mean_fr", "train_mean_fr"],
        var_name="condition",
        value_name="firing_rate",
    )

    # Set up the figure
    plt.figure(figsize=(12, 6))

    # Plot the barplot
    sns.barplot(
        data=df_melted,
        x="days_relative",
        y="firing_rate",
        hue="condition",
        palette={"pre_stim_mean_fr": "skyblue", "train_mean_fr": "salmon"},
        dodge=True,
    )

    plt.title(f"{animal_id} Pre-Stim vs. Stim Firing Rates by Session")
    plt.xlabel("Days")
    plt.ylabel("Firing Rate (Hz)")
    plt.tight_layout()
    plt.show()


def plot_prestim_vs_stim_firing_rates(df, animal_id):
    """
    Plots pre-stimulus vs. stimulus firing rates on scatter plots for each channel.

    Args:
        df (pd.DataFrame): DataFrame containing pre-stim and stim firing rates.
        animal_id (str): The animal ID for the title.
    """
    # Filter data to only include units marked as include=True
    df = df[df["include"]]

    # Get unique stimulation channels
    stim_channels = sorted(df["stim_channel"].unique())

    # Create individual joint plots for each channel
    for stim_channel in stim_channels:
        channel_data = df[df["stim_channel"] == stim_channel]

        # Create a jointplot with density plots on the axes
        g = sns.jointplot(
            data=channel_data,
            x="pre_stim_mean_fr",
            y="train_mean_fr",
            hue="stim_current",
            palette="viridis",
            kind="scatter",
            marginal_kws=dict(fill=True),
        )

        # Plot the diagonal line (x=y)
        max_rate = max(channel_data["pre_stim_mean_fr"].max(
        ), channel_data["train_mean_fr"].max())
        g.ax_joint.plot([0, max_rate], [0, max_rate],
                        linestyle="--", color="gray")

        # Set titles and labels
        g.fig.suptitle(
            f"Animal {animal_id} - Channel {stim_channel} Pre-Stim vs. Stim Firing Rates")
        g.set_axis_labels("Pre-Stim Firing Rate (Hz)", "Stim Firing Rate (Hz)")

        plt.tight_layout()
        plt.show()


# Group by cell type
def plot_prestim_vs_stim_firing_rates_for_cell_type(df, animal_id, margin=0.05):
    # Calculate global min and max for consistent axis limits, with margin adjustment

    global_min = min(df["pre_stim_mean_fr"].min(), df["pulse_mean_fr"].min())
    global_max = max(df["pre_stim_mean_fr"].max(), df["pulse_mean_fr"].max())

    # Adjust global_min to add a margin
    global_min = global_min - (global_max - global_min) * margin
    global_max = global_max + (global_max - global_min) * margin

    grouped = df.groupby("cell_type")
    filenames = []
    for i, (cell_type, group) in enumerate(grouped):
        # Create a jointplot with scatter and density plots
        group = group.dropna(
            subset=["pre_stim_mean_fr", "train_mean_fr"])
        # Filter only pulse-locked data

        g = sns.jointplot(
            data=group,
            x="pre_stim_mean_fr",
            y="train_mean_fr",
            hue="stim_current",
            palette="viridis",
            kind="scatter",  # or 'kde' if you want contour plots
            size=3
        )

        # Set the same x and y limits for all plots
        g.ax_joint.set_xlim(global_min, 50)
        g.ax_joint.set_ylim(global_min, global_max)

        slope, intercept, r_value, p_value, std_err = linregress(
            group["pre_stim_mean_fr"], group["train_mean_fr"])

        x_vals = np.array(g.ax_joint.get_xlim())
        y_vals = intercept + slope * x_vals
        g.ax_joint.plot(x_vals, y_vals, linestyle="--",
                        color="red", label=f"R² = {r_value**2:.2f}")

        # Plot the diagonal line (x=y)
        g.ax_joint.plot([global_min, global_max], [global_min,
                        global_max], linestyle="--", color="gray")

        # Set titles and labels
        g.set_axis_labels("Pre-Stim Mean FR", "Stim Mean FR")
        # g.fig.suptitle(f"{cell_type}", y=0.9)
        g.fig.suptitle(
            f"{cell_type} (R² = {r_value**2:.2f} Slope = {slope:.2f})", y=0.9)

        # Save the figure
        filename = f"celltype_{cell_type}.png"
        g.savefig(filename)
        filenames.append(filename)
        plt.close(g.fig)

    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, len(filenames))

    for i, filename in enumerate(filenames):
        img = plt.imread(filename)
        ax = fig.add_subplot(gs[i])
        ax.imshow(img)
        ax.axis("off")  # Hide the axes

    plt.suptitle(animal_id)
    plt.tight_layout()
    plt.savefig(f"{animal_id}_combined.png")
    plt.show()


def plot_prestim_vs_stim_firing_rates_for_cell_type_all_animals():
    animal_id1 = "ICMS92"  # Replace with your animal ID
    animal_id2 = "ICMS93"  # Replace with your second animal ID
    animal_id3 = "ICMS98"  # Replace with your third animal ID
    animal_id4 = "ICMS100"  # Replace with your third animal ID
    animal_id5 = "ICMS101"  # Replace with your third animal ID

    base_path = Path("C:/data/")  # Replace with your base path

    df1 = load_session_data(animal_id1, base_path)
    df2 = load_session_data(animal_id2, base_path)
    df3 = load_session_data(animal_id3, base_path)
    df4 = load_session_data(animal_id4, base_path)
    df5 = load_session_data(animal_id5, base_path)

    # Concatenate the DataFrames
    df = pd.concat([df1, df2, df3, df4, df5])
    df = df.reset_index(drop=True)

    plot_prestim_vs_stim_firing_rates_for_cell_type(df, "Aggregated")


def fix_list_format(value):
    """
    Fixes the formatting of a list by splitting the string on whitespace
    and converting it into a Python list.
    """
    if isinstance(value, str):
        # Remove brackets and split the string by whitespace
        value = value.strip("[]").split()
        # Convert the elements to floats
        try:
            value = [float(v) for v in value]
        except ValueError:
            value = np.nan  # If conversion fails, return NaN
    return value


def plot_prestim_vs_stim_firing_rates_by_unit_location(df, animal_id, margin=0.05):
    import ast

    # Convert the string representation of lists to actual lists
    df["unit_location"] = df["unit_location"].apply(fix_list_format)
    # Extract the second value from the unit_location list (handle missing values)
    df["unit_location_extracted"] = df["unit_location"].apply(
        lambda x: x[1] if len(x) > 1 else np.nan)

    # Ensure that the extracted unit locations are numeric
    df["unit_location_extracted"] = pd.to_numeric(
        df["unit_location_extracted"], errors="coerce")

    # Define the bins for unit locations as numeric ranges
    bins = [0, 150, 300, 500, 800, 1500]
    bin_labels = ["0-150", "150-300", "300-500", "500-800", "800-1500"]

    # Bin the extracted unit locations
    df["unit_location_bin"] = pd.cut(
        df["unit_location_extracted"], bins=bins, labels=bin_labels, include_lowest=True)

    # Calculate global min and max for consistent axis limits, with margin adjustment
    global_min = min(df["pre_stim_mean_fr"].min(), df["train_mean_fr"].min())
    global_max = max(df["pre_stim_mean_fr"].max(), df["train_mean_fr"].max())

    # Adjust global_min to add a margin
    global_min = global_min - (global_max - global_min) * margin
    global_max = global_max + (global_max - global_min) * margin

    grouped = df.groupby("unit_location_bin")
    filenames = []
    for i, (location_bin, group) in enumerate(grouped):
        # Create a jointplot with scatter and density plots
        g = sns.jointplot(
            data=group,
            x="pre_stim_mean_fr",
            y="train_mean_fr",
            hue="stim_current",
            palette="viridis",
            kind="scatter",  # or 'kde' if you want contour plots
        )

        # Set the same x and y limits for all plots
        g.ax_joint.set_xlim(global_min, global_max)
        g.ax_joint.set_ylim(global_min, global_max)

        # Plot the diagonal line (x=y)
        g.ax_joint.plot([global_min, global_max], [global_min,
                        global_max], linestyle="--", color="gray")

        # Set titles and labels
        g.set_axis_labels("Pre-Stim Mean FR", "Train Mean FR")
        g.fig.suptitle(f"Location {location_bin} µm", y=0.9)

        # Save the figure
        filename = f"location_{location_bin}.png"
        g.savefig(filename)
        filenames.append(filename)
        plt.close(g.fig)

    # Combine the individual plots into one figure
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, len(filenames))

    for i, filename in enumerate(filenames):
        img = plt.imread(filename)
        ax = fig.add_subplot(gs[i])
        ax.imshow(img)
        ax.axis("off")  # Hide the axes

    plt.suptitle(f"Animal ID: {animal_id}")
    plt.tight_layout()
    plt.savefig(f"{animal_id}_combined_locations.png")
    plt.show()


def plot_prestim_vs_stim_firing_rates_for_depths_all_animals():
    animal_id1 = "ICMS92"  # Replace with your animal ID
    animal_id2 = "ICMS93"  # Replace with your second animal ID
    animal_id3 = "ICMS98"  # Replace with your third animal ID
    animal_id4 = "ICMS100"  # Replace with your third animal ID
    animal_id5 = "ICMS101"  # Replace with your third animal ID

    base_path = Path("C:/data/")  # Replace with your base path

    df1 = load_session_data(animal_id1, base_path)
    df2 = load_session_data(animal_id2, base_path)
    df3 = load_session_data(animal_id3, base_path)
    df4 = load_session_data(animal_id4, base_path)
    df5 = load_session_data(animal_id5, base_path)

    # Concatenate the DataFrames
    df = pd.concat([df1, df2, df3, df4, df5])
    df = df.reset_index(drop=True)

    plot_prestim_vs_stim_firing_rates_by_unit_location(df, "Aggregated")


# Group by cell type

def plot_prestim_vs_stim_firing_rates_for_cell_type_and_current(df, animal_id, margin=0.05):
    # Calculate global min and max for consistent axis limits, with margin adjustment
    global_min = min(df["pre_stim_mean_fr"].min(), df["train_mean_fr"].min())
    global_max = max(df["pre_stim_mean_fr"].max(), df["train_mean_fr"].max())

    # Adjust global_min to add a margin
    global_min = global_min - (global_max - global_min) * margin
    global_max = global_max + (global_max - global_min) * margin

    # Filter the dataframe to include only stim currents 3, 4, 5, 6
    df_filtered = df[df["stim_current"].isin([3, 4, 5, 6])]

    # Get unique stim currents (should only be 3, 4, 5, 6) and cell types
    unique_stim_currents = sorted(df_filtered["stim_current"].unique())
    unique_cell_types = df_filtered["cell_type"].unique()

    # Number of rows = number of unique stim currents (3, 4, 5, 6)
    num_rows = len(unique_stim_currents)
    num_cols = len(unique_cell_types)

    # Create a grid of subplots dynamically based on the available stim currents and cell types
    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=(4 * num_cols, 2 * num_rows))
    fig.tight_layout(pad=5.0)  # Add some padding between plots

    # Group the dataframe by both `cell_type` and `stim_current`
    grouped = df_filtered.groupby(["cell_type", "stim_current"])

    for (cell_type, stim_current), group in grouped:
        # Drop NaN values
        group = group.dropna(subset=["pre_stim_mean_fr", "train_mean_fr"])

        # Get the row and column index for the subplot
        row_idx = unique_stim_currents.index(stim_current)
        col_idx = list(unique_cell_types).index(cell_type)

        ax = axes[row_idx, col_idx]  # Select the correct subplot

        # Create scatter plot directly on the subplot `ax`
        sns.scatterplot(
            data=group,
            x="pre_stim_mean_fr",
            y="train_mean_fr",
            ax=ax,
            palette="viridis",
            size=5,
            legend=False
        )

        # Set consistent axis limits
        ax.set_xlim(global_min, 50)
        ax.set_ylim(global_min, global_max)

        # Linear regression and plot on the same subplot
        if len(group) >= 2:
            slope, intercept, r_value, p_value, std_err = linregress(
                group["pre_stim_mean_fr"], group["train_mean_fr"])
            x_vals = np.array(ax.get_xlim())
            y_vals = intercept + slope * x_vals
            ax.plot(x_vals, y_vals, linestyle="--",
                    color="red")

        # Plot the diagonal line (x=y)
        ax.plot([global_min, global_max], [global_min, global_max],
                linestyle="--", color="gray")

        # Set titles and labels
        ax.set_title(f"{cell_type}, Current: {stim_current} R² = {
                     r_value**2:.2f} Slope = {slope:.2f}", fontsize=10)

        ax.set_xlabel("Pre-Stim Mean FR")
        ax.set_ylabel("Stim Mean FR")

    # Save and show the figure
    plt.suptitle(animal_id)
    plt.tight_layout()
    plt.savefig(f"{animal_id}_combined.png")
    plt.show()


def plot_prestim_vs_stim_firing_rates_for_cell_type_and_currents_all_animals():
    animal_id1 = "ICMS92"  # Replace with your animal ID
    animal_id2 = "ICMS93"  # Replace with your second animal ID
    animal_id3 = "ICMS98"  # Replace with your third animal ID
    animal_id4 = "ICMS100"  # Replace with your third animal ID
    animal_id5 = "ICMS101"  # Replace with your third animal ID

    base_path = Path("C:/data/")  # Replace with your base path

    df1 = load_session_data(animal_id1, base_path)
    df2 = load_session_data(animal_id2, base_path)
    df3 = load_session_data(animal_id3, base_path)
    df4 = load_session_data(animal_id4, base_path)
    df5 = load_session_data(animal_id5, base_path)

    # Concatenate the DataFrames
    df = pd.concat([df1, df2, df3, df4, df5])
    df = df.reset_index(drop=True)

    plot_prestim_vs_stim_firing_rates_for_cell_type_and_current(
        df, "Aggregated")


# %%
# plot_prestim_vs_stim_firing_rates_for_cell_type_all_animals()
animal_id = "ICMS92"  # Replace with your animal ID
base_path = Path("C:/data/")  # Replace with your base path
df = load_session_data(animal_id, base_path)
# plot_prestim_vs_stim_firing_rates_for_cell_type_and_current(
#     df, animal_id)
plot_prestim_vs_stim_firing_rates_for_cell_type_and_currents_all_animals()

# plot_prestim_vs_stim_firing_rates_for_cell_type(df, animal_id)
# plot_prestim_vs_stim_firing_rates_for_cell_type(df, animal_id)
# plot_prestim_vs_stim_firing_rates_for_cell_type_all_animals()
# plot_prestim_vs_stim_firing_rates_for_cell_type_all_animals()
#
