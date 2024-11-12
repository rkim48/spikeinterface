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
import pickle
from scipy.stats import norm


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
            f"{animal_id}/Behavior/*/batch_sort/stim_condition_results.pkl")
    else:
        session_files = base_path.glob(
            f"{animal_id}/*/batch_sort/stim_condition_results.pkl")

    df_list = []
    for session_file in session_files:
        with open(session_file, 'rb') as file:
            df = pickle.load(file)
        session_name = session_file.parent.parent.name
        # Extract session name from the path
        df["session"] = session_file.parent.parent.name
        session_date = pd.to_datetime(session_name, format="%d-%b-%Y")
        df["session_date"] = session_date
        df_list.append(df)

    # Combine all session DataFrames
    df = pd.concat(df_list, ignore_index=True)

    df = df.dropna()

    # Order by session date
    df = df.sort_values(
        by="session_date").reset_index(drop=True)

    # Calculate days relative to the first session (1-indexed)
    df["days_relative"] = (
        df["session_date"] - df["session_date"].min()).dt.days + 1

    # preprocess
    df = add_sig_modulated_to_responses(df)
    # df = filter_data(df)
    df = get_stim_train_segment_firing_rate(df)
    df = add_time_to_peak_to_responses(df)
    # df = correct_df_unit_locations(df, animal_id)
    df = add_threshold_to_responses(df, animal_id)

    return df


def correct_df_unit_locations(df, animal_id):
    scale = 0.8666

    # Determine translation based on animal_id
    match animal_id:
        case "ICMS92":
            translation = -104
        case "ICMS93":
            translation = -208
        case "ICMS98":
            translation = -52
        case "ICMS100":
            translation = 0
        case "ICMS101":
            translation = -156
        case _:
            print("Error: invalid animal_id")
            return

    # Apply scale and translation only to the second coordinate (y-position)
    df['unit_location'] = df['unit_location'].apply(
        # Modify y-position, leave x-position unchanged
        lambda loc: [loc[0], loc[1] * scale + translation]
    )

    # Remove rows where the second coordinate (y-position) is less than 0
    # df_remove = df[df['unit_location'].apply(lambda loc: loc[1] < 0)]
    df = df[df['unit_location'].apply(lambda loc: loc[1] >= 0)]

    return df


def add_threshold_to_responses(df, animal_id):
    import ast
    blocks_csv_path = os.path.join(
        "batch_process/postprocessing/behavioral_data", f"{animal_id}_blocks.csv")
    blocks_df = pd.read_csv(blocks_csv_path)
    for session in blocks_df['Session'].unique():
        session_rows = df[df['session'] == os.path.basename(session)]
        session_block_rows = blocks_df[blocks_df['Session'] == session]
        avg_thresholds_str = session_block_rows['Avg Thresholds'].iloc[0]
        if pd.isna(avg_thresholds_str):
            continue  # Skip this session if Avg Thresholds is NaN
        avg_thresholds = ast.literal_eval(avg_thresholds_str)
        chA = session_block_rows['ChA'].iloc[0]
        chB = session_block_rows['ChB'].iloc[0]
        chC = session_block_rows['ChC'].iloc[0]
        chA_threshold = avg_thresholds[0]
        chB_threshold = avg_thresholds[1]
        chC_threshold = avg_thresholds[2]
        for i, row in session_rows.iterrows():
            stim_channel = row['stim_channel']
            if stim_channel == chA:
                df.loc[i, 'detection_threshold'] = chA_threshold
            elif stim_channel == chB:
                df.loc[i, 'detection_threshold'] = chB_threshold
            elif stim_channel == chC:
                df.loc[i, 'detection_threshold'] = chC_threshold
    return df


def add_sig_modulated_to_responses(df):
    significantly_modulated = []
    for index, row in df.iterrows():
        z_score = row['z_score']
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        if p_value < 0.05:
            significantly_modulated.append(True)
        else:
            significantly_modulated.append(False)
    df['significantly_modulated'] = significantly_modulated
    return df


def add_time_to_peak_to_responses(df):
    time_to_peak_list = []
    for index, row in df.iterrows():
        fr_times = row['fr_times']  # Assuming this is a list of time points
        frs = row['firing_rate']  # Assuming this is a list of firing rates
        if isinstance(fr_times, np.ndarray) and isinstance(frs, np.ndarray) and len(fr_times) == len(frs):
            valid_bins = np.where((np.array(fr_times) >= 0)
                                  & (np.array(fr_times) <= 700))[0]
            if len(valid_bins) > 0:
                peak_index = np.argmax(np.array(frs)[valid_bins])
                time_to_peak = np.array(fr_times)[valid_bins][peak_index]
            else:
                time_to_peak = np.nan  # No valid times in the range
        else:
            time_to_peak = np.nan  # If data is not valid
        time_to_peak_list.append(time_to_peak)
    df['time_to_peak'] = time_to_peak_list
    return df


def get_stim_train_segment_firing_rate(df, fraction_range=[0, 0.5]):
    if df.empty or 'fr_times' not in df.columns:
        print("DataFrame is empty or 'fr_times' column is missing.")
        return df
    fr_times = df['fr_times'].iloc[0]
    N = len(df)  # Loop over all rows in the DataFrame
    first_half_avg_fr = []
    second_half_avg_fr = []
    for i in range(N):
        stim_fr = df['firing_rate'].iloc[i]
        # Get the firing rate for the time window between 0 and 700 ms
        train_seg = stim_fr[(fr_times > 0) & (fr_times < 700)]
        stim_times = fr_times[(fr_times > 0) & (fr_times < 700)]
        # Divide the time segment into two halves
        bin_range = (len(stim_times) * np.array(fraction_range)).astype(int)
        first_half = train_seg[bin_range[0]:bin_range[1]]
        second_half = train_seg[bin_range[1]:]
        # Calculate the average firing rate for each half
        avg_first_half = np.mean(first_half) if len(first_half) > 0 else np.nan
        avg_second_half = np.mean(second_half) if len(
            second_half) > 0 else np.nan
        # Append the results to their respective lists
        first_half_avg_fr.append(avg_first_half)
        second_half_avg_fr.append(avg_second_half)
    # Add the new columns to the DataFrame
    df['first_half_avg_fr'] = first_half_avg_fr
    df['second_half_avg_fr'] = second_half_avg_fr
    return df


def plot_longitudinal_data(df, var1="z_score", var2="pulse_mean_fr"):
    """
    Plots longitudinal data of Z-scores and Pulse Mean FR across sessions.

    Args:
        animal_id (str): The animal ID to plot.
        base_path (Path): The base directory where session data is stored.
    """

    # Calculate median Z-scores and Pulse Mean FR
    median_z_scores = df.groupby(["days_relative", "stim_channel", "stim_current"])[
        var1].median().reset_index()
    median_pulse_fr = (
        df.groupby(["days_relative", "stim_channel", "stim_current"])[
            var2].median().reset_index()
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
                current_data_z["days_relative"], current_data_z[var1], marker="o", label=f"{stim_current} µA"
            )

        ax_z.set_title(f"Channel {stim_channel} - {var1}")
        ax_z.set_ylabel(f"Median {var1}")
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
                current_data_fr[var2],
                marker="o",
                label=f"{stim_current} µA",
            )

        ax_fr.set_title(f"Channel {stim_channel} - {var2}")
        ax_fr.set_ylabel(f"Median {var2}")
        ax_fr.set_xlabel("Days Relative to First Session")

    # Create a single legend for all subplots
    handles, labels = ax_z.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(
        0.85, 0.5), title="Stimulation Current")

    plt.suptitle(animal_id)
    # Adjust the right padding to make space for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


def plot_aggregated_longitudinal_data(df):
    """
    Plots aggregated longitudinal data of Z-scores and Pulse Mean FR across sessions.

    Args:
        df (DataFrame): The aggregated data for all channels and all sessions.
    """

    print("taking abs value of z score")
    # Compute median of absolute z-scores
    aggregated_z_scores = df.groupby(["days_relative", "stim_current"])[
        "z_score"].apply(lambda x: x.abs().mean()).reset_index()

    aggregated_pulse_fr = df.groupby(["days_relative", "stim_current"])[
        "pulse_mean_fr"].mean().reset_index()

    # Get unique stimulation currents
    stim_currents = [2, 3, 4, 5, 6, 7]

    # Create a figure with 2 subplots: one for Z-scores and one for Pulse Mean FR
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    # Plot Z-Scores
    ax_z = axes[0]
    for stim_current in stim_currents:
        current_data_z = aggregated_z_scores[aggregated_z_scores["stim_current"] == stim_current]

        # Plot line
        ax_z.plot(
            current_data_z["days_relative"],
            current_data_z["z_score"],
            label=f"{stim_current} µA"
        )

        # Plot individual points
        ax_z.scatter(
            current_data_z["days_relative"],
            current_data_z["z_score"],
            marker="o", s=50, zorder=5  # 's' for marker size, 'zorder' brings points to the front
        )

    ax_z.set_title("Aggregated Z-Scores Across Channels")
    ax_z.set_ylabel("Median Z-Score")
    ax_z.set_xlabel("Days Relative to First Session")

    # Plot Pulse Mean FR
    ax_fr = axes[1]
    for stim_current in stim_currents:
        current_data_fr = aggregated_pulse_fr[aggregated_pulse_fr["stim_current"] == stim_current]

        # Plot line
        ax_fr.plot(
            current_data_fr["days_relative"],
            current_data_fr["pulse_mean_fr"],
            label=f"{stim_current} µA"
        )

        # Plot individual points
        ax_fr.scatter(
            current_data_fr["days_relative"],
            current_data_fr["pulse_mean_fr"],
            marker="o", s=50, zorder=5  # 's' for marker size, 'zorder' brings points to the front
        )

    ax_fr.set_title("Aggregated Pulse Mean FR Across Channels")
    ax_fr.set_ylabel("Median Pulse FR (Hz)")
    ax_fr.set_xlabel("Days Relative to First Session")

    # Create a single legend for both subplots
    handles, labels = ax_z.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right",
               title="Stimulation Current")

    # Adjust the layout and add a title
    plt.suptitle("Aggregated Longitudinal Data")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


def plot_longitudinal_all_animals():
    animal_ids = ["ICMS92", "ICMS93", "ICMS98",
                  "ICMS100", "ICMS101"]  # Example animal IDs
    base_path = Path("C:/data/")  # Replace with your actual base path

    # Load session data for each animal
    dfs = [load_session_data(animal_id, base_path) for animal_id in animal_ids]

    # Concatenate the DataFrames for all animals
    df = pd.concat(dfs, ignore_index=True)

    # Plot the aggregated longitudinal data
    plot_aggregated_longitudinal_data(df)


def plot_prestim_vs_stim_firing_rates_for_cell_type(df, animal_id, y_value='train_mean_fr', margin=0.05, stim_currents=[3, 4, 5, 6]):
    # Filter dataframe to only include the specified stim currents
    df = df[df['stim_current'].isin(stim_currents)]

    # Calculate global min and max for consistent axis limits, with margin adjustment
    global_min = min(df["pre_stim_mean_fr"].min(), df[y_value].min())
    global_max = max(df["pre_stim_mean_fr"].max(), df[y_value].max())

    # Adjust global_min and global_max to add a margin
    global_min = global_min - (global_max - global_min) * margin
    global_max = global_max + (global_max - global_min) * margin

    grouped = df.groupby("cell_type")
    filenames = []

    for i, (cell_type, group) in enumerate(grouped):
        # Drop NaNs
        group = group.dropna(subset=["pre_stim_mean_fr", y_value])

        # Create a jointplot
        g = sns.jointplot(
            data=group,
            x="pre_stim_mean_fr",
            y=y_value,
            hue="stim_current",
            palette="viridis",
            kind="scatter",
            joint_kws={'s': 20, 'alpha': 0.8}
        )

        # Set consistent axis limits
        g.ax_joint.set_xlim(global_min, 40)
        g.ax_joint.set_ylim(global_min, 80)

        # Plot the diagonal line (x=y)
        g.ax_joint.plot([global_min, global_max], [global_min,
                        global_max], linestyle="--", color="gray")

        # Set titles and labels
        g.set_axis_labels("Pre-Stim Mean FR", y_value)
        g.fig.suptitle(f"{cell_type}", y=0.9)

        # Save the figure
        filename = f"celltype_{cell_type}.png"
        g.savefig(filename)
        filenames.append(filename)
        plt.close(g.fig)

    # Combine the saved plots into a single figure
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


def plot_prestim_vs_stim_firing_rates_for_cell_type_all_animals(y_value="train_mean_fr"):
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
    df = filter_data(df)

    plot_prestim_vs_stim_firing_rates_for_cell_type(df, "Aggregated", y_value)


def plot_prestim_vs_stim_firing_rates_for_cell_type_and_other_var(df, animal_id, var="stim_current", margin=0.05):
    # Calculate global min and max for consistent axis limits, with margin adjustment
    global_min = min(df["pre_stim_mean_fr"].min(), df["train_mean_fr"].min())
    global_max = max(df["pre_stim_mean_fr"].max(), df["train_mean_fr"].max())

    # Adjust global_min to add a margin
    global_min = global_min - (global_max - global_min) * margin
    global_max = global_max + (global_max - global_min) * margin

    # Filter the dataframe to include only stim currents 3, 4, 5, 6
    if var == "stim_current":
        df_filtered = df[df["stim_current"].isin([3, 4, 5, 6])]
    elif var == "detection_threshold":
        df_filtered = df[df["detection_threshold"].isin([2, 3, 4, 5])]

    # Get unique stim currents (should only be 3, 4, 5, 6) and cell types
    unique_var_values = sorted(df_filtered[var].unique())
    unique_cell_types = df_filtered["cell_type"].unique()

    # Number of rows = number of unique stim currents (3, 4, 5, 6)
    num_rows = len(unique_var_values)
    num_cols = len(unique_cell_types)

    # Create a grid of subplots dynamically based on the available stim currents and cell types
    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=(4 * num_cols, 2 * num_rows))
    fig.tight_layout(pad=5.0)  # Add some padding between plots

    # Group the dataframe by both `cell_type` and `stim_current`
    grouped = df_filtered.groupby(["cell_type", var])

    for (cell_type, var_i), group in grouped:
        # Drop NaN values
        group = group.dropna(subset=["pre_stim_mean_fr", "train_mean_fr"])

        # Get the row and column index for the subplot
        row_idx = unique_var_values.index(var_i)
        col_idx = list(unique_cell_types).index(cell_type)

        ax = axes[row_idx, col_idx]  # Select the correct subplot

        # Create scatter plot directly on the subplot `ax`
        sns.scatterplot(
            data=group,
            x="pre_stim_mean_fr",
            y="train_mean_fr",
            ax=ax,
            palette="viridis",
            s=10,  # Size of the scatter points
            alpha=0.8,  # Transparency for better visibility
            legend=False
        )

        # Set consistent axis limits
        ax.set_xlim(global_min, 40)
        ax.set_ylim(global_min, 100)

        # Linear regression and plot on the same subplot
        if len(group) >= 2:
            slope, intercept, r_value, p_value, std_err = linregress(
                group["pre_stim_mean_fr"], group["train_mean_fr"])
            x_vals = np.array(ax.get_xlim())
            y_vals = intercept + slope * x_vals
            ax.plot(x_vals, y_vals, linestyle="--", color="red")

            # Add slope and R² to the title of each subplot with regular font size
            ax.set_title(
                f"R² = {r_value**2:.2f}, Slope = {slope:.2f}",
                fontsize=10
            )

        # Plot the diagonal line (x=y)
        ax.plot([global_min, global_max], [global_min, global_max],
                linestyle="--", color="gray")

        # Only set the ylabel (current) for the first column in each row
        if col_idx == 0:
            ax.set_ylabel(f"{int(var_i)} uA")

        # Only set the title (cell type) for the top row, but place it above the entire column with larger font size
        if row_idx == 0:
            ax.set_title(f"{cell_type}\nR² = {
                         r_value**2:.2f}, Slope = {slope:.2f}", fontsize=10)

        # Remove repetitive x and y labels except for the bottom-left plot
        if col_idx > 0:
            ax.set_ylabel("")
        if row_idx < num_rows - 1:
            ax.set_xlabel("")
        else:
            if col_idx == num_cols - 1:
                ax.set_xlabel("Pre-Stim Mean FR")
                ax.set_ylabel("Stim Mean FR")
            else:
                ax.set_xlabel("")

    # Save and show the figure
    plt.suptitle(animal_id)
    plt.tight_layout()
    plt.savefig(f"{animal_id}_combined.png")
    plt.show()


def plot_prestim_vs_stim_firing_rates_for_cell_type_and_var_all_animals(var):
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

    plot_prestim_vs_stim_firing_rates_for_cell_type_and_other_var(
        df, "Aggregated", var)


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
    df["unit_location"] = df["unit_location"].apply(fix_list_format)
    df["unit_location_extracted"] = df["unit_location"].apply(
        lambda x: x[1] if len(x) > 1 else np.nan)
    df["unit_location_extracted"] = pd.to_numeric(
        df["unit_location_extracted"], errors="coerce")
    # Define the bins for unit locations as numeric ranges
    bins = [0, 100, 400, 550, 800, 1000, 1500]
    bin_labels = ["0-100", "100-400", "400-550",
                  "550-800", "800-1000", "1000-1500"]
    df["unit_location_bin"] = pd.cut(
        df["unit_location_extracted"], bins=bins, labels=bin_labels, include_lowest=True)
    global_min = min(df["pre_stim_mean_fr"].min(), df["train_mean_fr"].min())
    global_max = max(df["pre_stim_mean_fr"].max(), df["train_mean_fr"].max())
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
        g.ax_joint.set_xlim(global_min, 50)
        g.ax_joint.set_ylim(global_min, 100)

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
    gs = gridspec.GridSpec(2, int(len(filenames) / 2))

    for i, filename in enumerate(filenames):
        img = plt.imread(filename)
        ax = fig.add_subplot(gs[i])
        ax.imshow(img)
        ax.axis("off")  # Hide the axes

    plt.suptitle(f"Animal ID: {animal_id}")
    plt.tight_layout()
    plt.savefig(f"{animal_id}_combined_locations.png")
    plt.show()


def plot_prestim_vs_stim_firing_rates_by_unit_location_all_animals():
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

    plot_prestim_vs_stim_firing_rates_by_unit_location(
        df, "Aggregated")


def plot_prestim_vs_stim_firing_rates_for_stim_current_and_threshold(df, animal_id, margin=0.05):
    """
    Plots the pre-stim vs stim firing rates for cell types while comparing stim_current and detection_threshold.
    Args:
        df: DataFrame with firing rate data.
        animal_id: ID of the animal.
        margin: Margin to add to axis limits.
    """
    # Calculate global min and max for consistent axis limits, with margin adjustment
    global_min = min(df["pre_stim_mean_fr"].min(), df["train_mean_fr"].min())
    global_max = max(df["pre_stim_mean_fr"].max(), df["train_mean_fr"].max())

    # Adjust global_min to add a margin
    global_min = global_min - (global_max - global_min) * margin
    global_max = global_max + (global_max - global_min) * margin

    # Filter the dataframe to include only stim currents and thresholds
    df_filtered = df[df["stim_current"].isin([3, 4, 5, 6])]
    df_filtered = df_filtered[df_filtered["detection_threshold"].isin([
                                                                      3, 4, 5, 6])]

    # Get unique stim currents and detection thresholds
    unique_stim_currents = sorted(df_filtered["stim_current"].unique())
    unique_detection_thresholds = sorted(
        df_filtered["detection_threshold"].unique())

    # Number of rows = number of unique stim currents
    # Number of columns = number of unique detection thresholds
    num_rows = len(unique_stim_currents)
    num_cols = len(unique_detection_thresholds)

    # Create a grid of subplots dynamically based on the available stim currents and detection thresholds
    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=(2 * num_cols, 1.5 * num_rows))
    fig.tight_layout(pad=1.0)  # Add some padding between plots

    # Group the dataframe by `stim_current` and `detection_threshold`
    grouped = df_filtered.groupby(["stim_current", "detection_threshold"])

    for (stim_current, detection_threshold), group in grouped:
        # Drop NaN values
        group = group.dropna(subset=["pre_stim_mean_fr", "train_mean_fr"])

        # Get the row and column index for the subplot
        row_idx = unique_stim_currents.index(stim_current)
        col_idx = unique_detection_thresholds.index(detection_threshold)

        ax = axes[row_idx, col_idx]  # Select the correct subplot

        # Create scatter plot directly on the subplot `ax`
        sns.scatterplot(
            data=group,
            x="pre_stim_mean_fr",
            y="train_mean_fr",
            ax=ax,
            s=10,  # Size of the scatter points
            alpha=0.8,  # Transparency for better visibility
            legend=False
        )

        # Set consistent axis limits
        ax.set_xlim(global_min, 25)
        ax.set_ylim(global_min, 80)

        # Plot the diagonal line (x=y)
        ax.plot([global_min, global_max], [global_min, global_max],
                linestyle="--", color="gray")

        # Set stim current label (y-axis) for every row
        if col_idx == 0:
            ax.set_ylabel(f"Current: {stim_current}")
            ax.set_xlabel("")
        elif (col_idx == num_cols - 1) and (row_idx == num_rows - 1):
            ax.set_ylabel("Stim Mean FR")
            ax.set_xlabel("Pre-Stim Mean FR")
        else:
            ax.set_ylabel("")
            ax.set_xlabel("")

    # Add detection thresholds as column titles above the plots
    for col_idx, detection_threshold in enumerate(unique_detection_thresholds):
        axes[0, col_idx].annotate(f"Threshold: {int(detection_threshold)}",
                                  xy=(0.5, 1.2), xycoords='axes fraction', ha='center', va='center',
                                  fontsize=11)

    # Set super title and save the figure
    # plt.suptitle(f"Animal ID: {animal_id}", fontsize=16)
    # Adjust layout to make space for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{animal_id}_combined.png")
    plt.show()


def plot_prestim_vs_stim_firing_rates_by_stim_current_and_threshold_all_animals():
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
    df = filter_data(df)

    plot_prestim_vs_stim_firing_rates_for_stim_current_and_threshold(
        df, "Aggregated")


def plot_EI_ratio(df):
    """
    Plots the E/I ratio (Excitatory/Inhibitory) across weeks and stimulation currents.

    Args:
        df (DataFrame): A dataframe containing the columns 'weeks_relative', 'stim_current', 'cell_type'
                        where 'cell_type' can be 'pyramidal' for excitatory or some other type for inhibitory.
    """
    # Get unique weeks and currents
    df['weeks_relative'] = (df['days_relative'] // 7).astype(int)
    weeks = sorted(df['weeks_relative'].unique())
    stim_currents = [2, 3, 4, 5, 6, 7]

    # Dictionary to store E/I ratios for each week and current
    EI_ratios = {current: [] for current in stim_currents}

    # Loop over each week
    for week in weeks:
        df_week = df[df['weeks_relative'] == week]

        # Loop over each current
        for current in stim_currents:
            df_current = df_week[df_week['stim_current'] == current]

            # Initialize counts
            E_count = 0  # Excitatory (Pyramidal)
            I_count = 0  # Inhibitory (Non-pyramidal)

            # Classify cells and count
            for _, row in df_current.iterrows():
                if row['cell_type'] == 'Pyramidal Cell':
                    E_count += 1
                else:
                    I_count += 1

            # Calculate E/I ratio (avoid division by zero)
            if I_count > 0:
                EI_ratio = E_count / I_count
            else:
                EI_ratio = np.nan  # Set to NaN if no inhibitory cells found

            # Append the ratio to the dictionary
            EI_ratios[current].append(EI_ratio)

    # Plotting the E/I ratio
    plt.figure(figsize=(10, 6))

    for current in stim_currents:
        plt.plot(weeks, EI_ratios[current], marker='o', label=f'{current} µA')

    plt.xlabel('Weeks Relative to First Session')
    plt.ylabel('E/I Ratio (Excitatory/Inhibitory)')
    plt.title('E/I Ratio Across Weeks and Stimulation Currents')
    plt.legend(title='Stimulation Current')
    plt.tight_layout()
    plt.show()


def plot_cell_type_feature_hist(df, feature):
    """
    Plots histograms of a specific feature for three cell types: Pyramidal, Wide Interneuron, and Narrow Interneuron.

    Args:
        df (DataFrame): The dataframe containing the data, including the 'cell_type' and 'stim_current' columns.
        feature (str): The feature (column name) to plot histograms for.
    """
    # Filter the data for each cell type
    pyr_df = df[df['cell_type'] == "Pyramidal Cell"]
    wide_int_df = df[df['cell_type'] == "Wide Interneuron"]
    narrow_int_df = df[df['cell_type'] == "Narrow Interneuron"]

    # List of dataframes for convenience
    cell_type_dfs = [pyr_df, wide_int_df, narrow_int_df]
    cell_type_labels = ["Pyramidal Cell",
                        "Wide Interneuron", "Narrow Interneuron"]

    # Stimulation currents
    stim_currents = [2, 3, 4, 5, 6, 7]
    # Create color map for different currents
    colors = plt.cm.viridis(np.linspace(0, 1, len(stim_currents)))

    # Create subplots for each cell type (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

    for i, (cell_df, cell_label) in enumerate(zip(cell_type_dfs, cell_type_labels)):
        ax = axes[i]

        # Loop through each stimulation current and plot a histogram
        for j, current in enumerate(stim_currents):
            current_df = cell_df[cell_df['stim_current'] == current]
            values = current_df[feature].dropna()  # Drop NaNs to avoid errors

            # Normalize the values
            counts, bins = np.histogram(values, bins=20)
            normalized_counts = counts / counts.sum()  # Normalization step

            # Plot the normalized bar plot
            ax.bar(bins[:-1], normalized_counts, width=np.diff(bins),
                   alpha=0.7, color=colors[j], label=f'{current} µA')

        # Set titles and labels
        ax.set_title(f'{cell_label}')
        ax.set_xlabel(f'{feature.replace("_", " ").capitalize()}')
        # Add ylabel only for the first subplot
        ax.set_ylabel('Normalized frequency' if i == 0 else '')

    # Create a shared legend for all plots
    fig.legend([f'{current} µA' for current in stim_currents],
               loc='center right', title="Stimulation Current")

    # Adjust layout and display the plot
    plt.suptitle(f'Normalized histograms of {feature.replace(
        "_", " ").capitalize()} by Cell Type and Stimulation Current', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


def plot_waveforms_by_cell_type_and_week(df):
    """
    Plots unique unit waveforms by cell type and week with appropriate axis labels.

    Args:
        df (DataFrame): DataFrame containing unit information, cell types, templates, and weeks.
    """
    # Filter the data for each cell type
    pyr_df = df[df['cell_type'] == "Pyramidal Cell"]
    wide_int_df = df[df['cell_type'] == "Wide Interneuron"]
    narrow_int_df = df[df['cell_type'] == "Narrow Interneuron"]

    # List of dataframes and corresponding labels for cell types
    cell_type_dfs = [pyr_df, wide_int_df, narrow_int_df]
    cell_type_labels = ["Pyramidal Cell",
                        "Wide Interneuron", "Narrow Interneuron"]

    # Unique weeks (0 to 5)
    weeks = [0, 1, 2, 3, 4, 5]

    # Create a 3x6 grid for subplots
    fig, axes = plt.subplots(3, 6, figsize=(10, 6), sharey=True)

    for row_idx, (cell_df, cell_label) in enumerate(zip(cell_type_dfs, cell_type_labels)):
        for col_idx, week in enumerate(weeks):
            ax = axes[row_idx, col_idx]

            # Filter for the current week and get unique units by "unit_id"
            week_df = cell_df[cell_df['weeks_relative']
                              == week].drop_duplicates(subset='unit_id')

            # Plot the waveforms for each unique unit in the current week
            for _, row in week_df.iterrows():
                # Assuming 'template' contains the waveform
                waveform = row['template']
                # Time axis in ms (30 samples per ms)
                time_axis = np.linspace(0, len(waveform) / 30, len(waveform))
                ax.plot(time_axis, waveform, alpha=0.7)

            # Set titles and labels
            if col_idx == 0:
                # Manually place the cell type label to the left
                ax.text(-0.15, 0.5, f"{cell_label}", fontsize=8, rotation=90,
                        transform=ax.transAxes, va='center')

            if row_idx == 0:
                ax.set_title(f"Week {week}", fontsize=12)  # Week title on top

            # Set y-axis ticks for amplitude only on the last column
            if col_idx == 5:
                ax.set_yticks([-200, 0, 50])  # Example y-ticks in µV
                ax.yaxis.set_label_position("right")
                ax.set_ylabel("Amplitude (µV)", fontsize=10)
                ax.yaxis.tick_right()

            # Set x-axis labels only for the bottom-left subplot
            if row_idx == 2 and col_idx == 0:
                ax.set_xlabel('Time (ms)')
                ax.set_xticks([0, len(waveform) / 60, len(waveform) / 30])
                ax.set_xticklabels(
                    [0, len(waveform) / 60, len(waveform) // 30])
            else:
                ax.set_xticks([])  # Hide x-ticks for all other subplots

            # Remove y-tick labels for non-leftmost subplots
            if col_idx != 5:
                ax.set_yticklabels([])

    # Add a super title and adjust the layout
    plt.suptitle("Unit Waveforms by Cell Type and Week", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def compute_within_between_procrustes_distances(df):
    from scipy.spatial import procrustes
    """
    Computes Procrustes distances and categorizes them into "within-group" and "between-group."
    """
    # Filter and prepare data
    df_filtered = df[df["stim_current"].isin([3, 4, 5, 6, 7])]
    df_filtered = df_filtered[df_filtered["detection_threshold"].isin([
                                                                      3, 4, 5, 6, 7])]

    unique_conditions = list(df_filtered.groupby(
        ["stim_current", "detection_threshold"]).groups.keys())

    distances = []
    labels = []

    for i, condition_a in enumerate(unique_conditions):
        for j, condition_b in enumerate(unique_conditions):
            if i < j:  # Only calculate once
                df_a = df_filtered[(df_filtered["stim_current"] == condition_a[0]) &
                                   (df_filtered["detection_threshold"] == condition_a[1])]
                df_b = df_filtered[(df_filtered["stim_current"] == condition_b[0]) &
                                   (df_filtered["detection_threshold"] == condition_b[1])]

                if df_a.empty or df_b.empty:
                    continue  # Skip if there's no data for either condition

                # Prepare 2D data for Procrustes analysis
                points_a = np.stack(
                    [df_a["pre_stim_mean_fr"].values, df_a["train_mean_fr"].values], axis=1)
                points_b = np.stack(
                    [df_b["pre_stim_mean_fr"].values, df_b["train_mean_fr"].values], axis=1)

                if np.isnan(points_a).any() or np.isnan(points_b).any():
                    continue  # Skip invalid centroids

                # Ensure equal number of points by subsampling
                min_points = min(len(points_a), len(points_b))
                points_a = points_a[:min_points]
                points_b = points_b[:min_points]

                # Compute Procrustes distance
                _, _, disparity = procrustes(points_a, points_b)

                # Check if this is a "within-group" or "between-group" comparison
                if condition_a[0] == condition_a[1] and condition_b[0] == condition_b[1]:
                    category = 'Within-Group'
                else:
                    category = 'Between-Group'

                distances.append(disparity)
                labels.append(category)

    return distances, labels


def compute_centroid_distances(df):
    from scipy.spatial.distance import euclidean

    """
    Computes centroid distances and categorizes them into "within-group" and "between-group,"
    while removing empty distances or self-comparisons.
    """
    # Filter and prepare data
    df_filtered = df[df["stim_current"].isin([3, 4, 5, 6, 7])]
    df_filtered = df_filtered[df_filtered["detection_threshold"].isin([
                                                                      3, 4, 5, 6, 7])]

    unique_conditions = list(df_filtered.groupby(
        ["stim_current", "detection_threshold"]).groups.keys())

    distances = []
    labels = []

    for i, condition_a in enumerate(unique_conditions):
        for j, condition_b in enumerate(unique_conditions):
            # Skip self-comparisons (when comparing the same condition)
            if i < j:  # Only calculate for distinct pairs and avoid duplicates
                df_a = df_filtered[(df_filtered["stim_current"] == condition_a[0]) &
                                   (df_filtered["detection_threshold"] == condition_a[1])]
                df_b = df_filtered[(df_filtered["stim_current"] == condition_b[0]) &
                                   (df_filtered["detection_threshold"] == condition_b[1])]

                # Check if either DataFrame is empty
                if df_a.empty or df_b.empty:
                    continue  # Skip if there's no data for either condition

                # Calculate centroids
                centroid_a = [df_a["pre_stim_mean_fr"].mean(),
                              df_a["train_mean_fr"].mean()]
                centroid_b = [df_b["pre_stim_mean_fr"].mean(),
                              df_b["train_mean_fr"].mean()]

                # Skip if centroids contain NaN values (i.e., if no valid data for centroid calculation)
                if np.isnan(centroid_a).any() or np.isnan(centroid_b).any():
                    continue  # Skip invalid centroids

                # Compute Euclidean distance between the centroids
                centroid_distance = euclidean(centroid_a, centroid_b)

                # Check if this is a "within-group" or "between-group" comparison
                if condition_a[0] == condition_a[1] and condition_b[0] == condition_b[1]:
                    category = 'Within-Group'
                else:
                    category = 'Between-Group'

                distances.append(centroid_distance)
                labels.append(category)

    return distances, labels


def plot_similarity_comparison(distances, labels):
    """
    Plots a swarm plot comparing the within-group and between-group similarities.
    """
    data = pd.DataFrame({'Distance': distances, 'Category': labels})

    plt.figure(figsize=(10, 6))
    sns.swarmplot(x='Category', y='Distance',
                  data=data, palette="Set2", size=5)
    plt.title(
        "Procrustes Distance Comparison Between Within-Group and Between-Group")
    plt.ylabel("Procrustes Distance")
    plt.xlabel("Group")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Category', y='Distance', data=data,
                   inner=None, palette="Set2", alpha=0.5)
    sns.swarmplot(x='Category', y='Distance', data=data,
                  color='k', alpha=0.7, size=5)
    plt.title(
        "Procrustes Distance Comparison Between Within-Group and Between-Group")
    plt.ylabel("Procrustes Distance")
    plt.xlabel("Group")
    plt.tight_layout()
    plt.show()


def plot_centroid_comparison(distances, labels):
    """
    Plots a swarm plot comparing within-group and between-group centroid distances.
    """
    data = pd.DataFrame({'Distance': distances, 'Category': labels})

    plt.figure(figsize=(10, 6))
    sns.swarmplot(x='Category', y='Distance',
                  data=data, palette="Set2", size=5)
    plt.title("Centroid Distance Comparison Between Within-Group and Between-Group")
    plt.ylabel("Centroid Distance (Euclidean)")
    plt.xlabel("Group")
    plt.tight_layout()
    plt.show()


def plot_aggregated_weekly_data(df, var1="z_score", var2="pulse_mean_fr", animal_id="Aggregated", use_median=False):
    """
    Plots aggregated longitudinal data of two selected variables (e.g., Z-Scores and Pulse Mean FR) across weeks
    with error bars to represent either the standard error of the mean (SEM) or interquartile range (IQR).

    Args:
        df (DataFrame): The aggregated data for all channels and all sessions.
        var1 (str): The first variable to plot (default is 'z_score').
        var2 (str): The second variable to plot (default is 'pulse_mean_fr').
        animal_id (str): The identifier for the animal (default is 'Aggregated').
        use_median (bool): If True, plots median with IQR. If False, plots mean with SEM.
    """

    # Create a new column for 'weeks_relative' by dividing 'days_relative' by 7 and flooring it
    df['weeks_relative'] = (df['days_relative'] // 7).astype(int)

    # Absolute values of var1 and var2
    df['abs_var1'] = df[var1].abs()
    df['abs_var2'] = df[var2].abs()

    if use_median:
        # Compute aggregated data using median and IQR
        aggregated_data_var1 = df.groupby(['weeks_relative', 'stim_current']).agg(
            median=('abs_var1', 'median'),
            q1=('abs_var1', lambda x: x.quantile(0.25)),
            q3=('abs_var1', lambda x: x.quantile(0.75))
        ).reset_index()

        aggregated_data_var2 = df.groupby(['weeks_relative', 'stim_current']).agg(
            median=('abs_var2', 'median'),
            q1=('abs_var2', lambda x: x.quantile(0.25)),
            q3=('abs_var2', lambda x: x.quantile(0.75))
        ).reset_index()
    else:
        # Compute aggregated data using mean and SEM
        aggregated_data_var1 = df.groupby(['weeks_relative', 'stim_current'])[
            'abs_var1'].agg(['mean', 'sem']).reset_index()
        aggregated_data_var2 = df.groupby(['weeks_relative', 'stim_current'])[
            'abs_var2'].agg(['mean', 'sem']).reset_index()

    # Get unique stimulation currents
    stim_currents = sorted(df['stim_current'].unique())

    # Create a figure with 2 subplots: one for var1 and one for var2
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    # Plot for var1 with bars (error bars)
    ax1 = axes[0]
    for stim_current in stim_currents:
        current_data_var1 = aggregated_data_var1[aggregated_data_var1['stim_current'] == stim_current]
        if use_median:
            # Plot with median and IQR as error bars
            ax1.errorbar(
                current_data_var1['weeks_relative'],
                current_data_var1['median'],
                yerr=[current_data_var1['median'] - current_data_var1['q1'],
                      current_data_var1['q3'] - current_data_var1['median']],
                fmt='-o',
                label=f"{stim_current} µA"
            )
        else:
            # Plot with mean and SEM as error bars
            ax1.errorbar(
                current_data_var1['weeks_relative'],
                current_data_var1['mean'],
                yerr=current_data_var1['sem'],
                fmt='-o',
                label=f"{stim_current} µA"
            )

    # Replace underscores with spaces for better formatting
    var1_label = var1.replace('_', ' ').capitalize()
    ax1.set_title(f"Aggregated {var1_label.capitalize()} Across Weeks")
    ax1.set_ylabel(f"{'Median' if use_median else 'Mean'} {var1_label}")
    ax1.set_xlabel("Weeks Relative to First Session")
    ax1.set_xticks(sorted(df['weeks_relative'].unique()))

    # Plot for var2 with bars (error bars)
    ax2 = axes[1]
    for stim_current in stim_currents:
        current_data_var2 = aggregated_data_var2[aggregated_data_var2['stim_current'] == stim_current]
        if use_median:
            # Plot with median and IQR as error bars
            ax2.errorbar(
                current_data_var2['weeks_relative'],
                current_data_var2['median'],
                yerr=[current_data_var2['median'] - current_data_var2['q1'],
                      current_data_var2['q3'] - current_data_var2['median']],
                fmt='-o',
                label=f"{stim_current} µA"
            )
        else:
            # Plot with mean and SEM as error bars
            ax2.errorbar(
                current_data_var2['weeks_relative'],
                current_data_var2['mean'],
                yerr=current_data_var2['sem'],
                fmt='-o',
                label=f"{stim_current} µA"
            )

    # Replace underscores with spaces for better formatting
    var2_label = var2.replace('_', ' ').capitalize()

    ax2.set_title(f"Aggregated Pulse Window FR Across Weeks" if var2 ==
                  "pulse_mean_fr" else f"Aggregated {var2_label} Across Weeks")
    ax2.set_ylabel(f"{'Median' if use_median else 'Mean'} {var2_label}")
    ax2.set_xlabel("Weeks Relative to First Session")
    ax2.set_xticks(sorted(df['weeks_relative'].unique()))

    # Create a single legend for both subplots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right",
               title="Stimulation Current")

    # Adjust the layout and show the plot
    plt.suptitle(f"{animal_id} Longitudinal Data: {
                 var1_label} and {var2_label}")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


def filter_data(df):
    # Filter data using the correct comparison for boolean and numerical conditions
    for ch in df['stim_channel'].unique():
        ch_df = df[df['stim_channel'] == ch]
        if len(ch_df['days_relative'].unique()) < 5:
            df = df[df['stim_channel'] != ch]

    df = df[
        (df["baseline_too_slow"] == False) &
        (df['significantly_modulated'] == True) &
        (df["z_score"] < 80) &
        (
            # If z_score > 0, must be pulse-locked
            ((df["z_score"] > 0) & (df["is_pulse_locked"] == True)) |
            # Allow negative or zero z_scores without pulse-locked condition
            (df["z_score"] <= 0)
        ) &
        (df['num_spikes'] > 50) &
        (df['stim_current'] < 7) &
        (df['stim_current'] > 2)]

    return df


def filter_data(df):
    # Filter data using the correct comparison for boolean and numerical conditions
    for ch in df['stim_channel'].unique():
        ch_df = df[df['stim_channel'] == ch]
        if len(ch_df['days_relative'].unique()) < 5:
            df = df[df['stim_channel'] != ch]

    df = df[
        (df["baseline_too_slow"] == False) &
        (df['modulated'] == True) &
        # (df["t_val"] > 0) &
        (df["t_val"] < 60) &
        # (
        #     # If z_score > 0, must be pulse-locked
        #     ((df["t_val"] > 0) & (df["is_pulse_locked"] == True)) |
        #     # Allow negative or zero z_scores without pulse-locked condition
        #     (df["t_val"] <= 0)
        # ) &
        # ((df["t_val"] > 0) & (df['num_spikes'] > 150)) &

        # (
        #     # If z_score > 0, must be pulse-locked
        #     ((df["t_val"] > 0) & (df["num_spikes"] > 100)) |
        #     # Allow negative or zero z_scores without pulse-locked condition
        #     (df["t_val"] <= 0)
        # )
        (df['num_spikes'] > 100) &
        (df['stim_current'] < 7) &
        (df['stim_current'] > 2)]

    return df


# %%
animal_ids = ["ICMS92", "ICMS93", "ICMS98",
              "ICMS100", "ICMS101"]  # Example animal IDs
base_path = Path("C:/data/")  # Replace with your actual base path

# Load session data for each animal
dfs = [load_session_data(animal_id, base_path) for animal_id in animal_ids]

# Concatenate the DataFrames for all animals
df = pd.concat(dfs, ignore_index=True)
df = filter_data(df)
plot_aggregated_weekly_data(df, var1='z_score', use_median=False)


# %%
dfs = [load_session_data(animal_id, base_path) for animal_id in animal_ids]

# Concatenate the DataFrames for all animals
# df = pd.concat(dfs, ignore_index=True)
# df = filter_data(df)
# plot_aggregated_weekly_data(df, var1='t_val')

# sns.stripplot(x=df['weeks_relative'], y=df['t_val'], hue=df['stim_current'], palette='deep')
x_column = 'weeks_relative'
y_column = 't_val'
group_column = 'stim_current'
sns.stripplot(x=df[x_column], y=df[y_column],
              hue=df[group_column], dodge=True, jitter=True, palette='deep')

# sns.violinplot(x=df[x_column], y=df[y_column],
#                hue=df[group_column], dodge=True, palette='deep')

# %%
if __name__ == "__main__":
    # animal_id = "ICMS101"  # Replace with your animal ID
    # base_path = Path("C:/data/")  # Replace with your base path
    # df = load_session_data(animal_id, base_path)
    # df = filter_data(df)

    # distances, labels = compute_within_between_procrustes_distances(df)
    # plot_similarity_comparison(distances, labels)

    # centroid_distances, centroid_labels = compute_centroid_distances(df)
    # plot_centroid_comparison(centroid_distances, centroid_labels)

    # plot_aggregated_weekly_data(
    #     df, var1="latency", var2="jitter")

    # plot_EI_ratio(df)
    plot_longitudinal_data(df, var1="z_score", var2="time_to_peak")
    # plot_longitudinal_all_animals()

    # plot_prestim_vs_stim_firing_rates_for_cell_type(df, animal_id)

    # first_half_avg_fr, second_half_avg_fr
    plot_prestim_vs_stim_firing_rates_for_cell_type_all_animals()
    # plot_prestim_vs_stim_firing_rates_for_cell_type_all_animals()

    # plot_cell_type_feature_hist(df, feature="time_to_peak")
    # plt.suptitle("Pulse window mean firing rate")

    # plot_waveforms_by_cell_type_and_week(df)

    # plot_prestim_vs_stim_firing_rates_by_unit_location(df, animal_id)
    # plot_prestim_vs_stim_firing_rates_by_unit_location_all_animals()
    # plt.suptitle("Location")

    # plot_prestim_vs_stim_firing_rates_for_cell_type_all_animals()
    # plot_prestim_vs_stim_firing_rates_for_cell_type_and_other_var()
    # plot_prestim_vs_stim_firing_rates_for_cell_type_and_var_all_animals(
    #     var="time_to_peak")
    # plt.suptitle("Detection threshold and cell type")

    # plot_prestim_vs_stim_firing_rates_for_stim_current_and_threshold(
    #     df, animal_id)
    # plot_prestim_vs_stim_firing_rates_by_stim_current_and_threshold_all_animals()
    # plt.suptitle("Stim current and threshold")

# %%


# %%
animal_ids = ["ICMS92", "ICMS93", "ICMS98",
              "ICMS100", "ICMS101"]  # Example animal IDs
base_path = Path("C:/data/")  # Replace with your actual base path

# Load session data for each animal
dfs = [load_session_data(animal_id, base_path) for animal_id in animal_ids]

# Concatenate the DataFrames for all animals
df = pd.concat(dfs, ignore_index=True)
plot_aggregated_weekly_data(df)
