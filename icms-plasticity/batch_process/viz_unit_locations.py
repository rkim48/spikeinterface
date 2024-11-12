import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
from datetime import datetime
import pandas as pd

import spikeinterface.full as si

import batch_process.util.file_util as file_util
import merge.merge_util as merge_util
import merge.classify_cell_type as cell_classifier
import batch_process.util.misc as misc
import batch_process.util.file_util as file_util

# %%


def main(data_folder=None, output_batch_file="batch_data.pkl"):

    job_kwargs = dict(n_jobs=5, chunk_duration="1s", progress_bar=True)
    batch_data = {}
    if data_folder:
        data_folders = [data_folder]
    else:
        starting_dir = "C:\\data"
        data_folders = file_util.file_dialog(starting_dir=starting_dir)

    fig, axes = plt.subplots(nrows=1, ncols=len(data_folders), figsize=(1.5 * len(data_folders), 6))

    if len(data_folders) == 1:
        axes = [axes]

    sorted_data_folders = file_util.sort_data_folders(data_folders)
    for i, data_folder in enumerate(sorted_data_folders):
        print("\n###########################################")
        print(f"{data_folder}: {i+1}/{len(data_folders)}")
        print("###########################################")

        save_folder = Path(data_folder) / "batch_sort/unit_viz"
        file_util.create_folder(save_folder)
        # Load horizontal merge analyzer
        analyzer = si.load_sorting_analyzer(folder=Path(data_folder) / "batch_sort/merge/hmerge_analyzer.zarr")

        df_path = Path(data_folder) / "batch_sort/stim_condition_results.csv"
        df = pd.read_csv(df_path)

        # Identify unit_ids to exclude (those with all include == False)
        units_to_exclude = df.groupby("unit_id")["include"].apply(lambda x: all(x == False)).reset_index()
        units_to_exclude = units_to_exclude[units_to_exclude["include"] == True]["unit_id"]

        extensions_to_compute = [
            "random_spikes",
            "waveforms",
            "templates",
            "template_similarity",
            "correlograms",
            "unit_locations",
        ]

        extension_params = {
            "unit_locations": {"method": "center_of_mass"},
            "correlograms": {"window_ms": 100, "bin_ms": 0.5},
        }

        for extension in extensions_to_compute:
            if not analyzer.has_extension(extension):
                analyzer.compute(extensions_to_compute, extension_params=extension_params, **job_kwargs)
                continue

        cell_type_df = cell_classifier.classify_units_into_cell_types(analyzer)["df"]
        cell_type_df = cell_type_df.reset_index()
        # Filter out the units in cell_type_df based on the exclusion list
        filtered_cell_type_df = cell_type_df[~cell_type_df["unit_id"].isin(units_to_exclude)]

        markers = {"Pyramidal Cell": "^", "Narrow Interneuron": "s", "Wide Interneuron": "o"}

        # get unit locations
        ul_ext = analyzer.get_extension("unit_locations")
        unit_locations = ul_ext.get_data()

        date_str = file_util.get_date_str(str(analyzer.folder))
        animal_id = file_util.get_animal_id(str(analyzer.folder))

        # Store data in batch_data dictionary
        batch_data[date_str] = {
            "animal_id": animal_id,
            "cell_type_df": filtered_cell_type_df,
            "unit_locations": unit_locations,
        }

        for cell_type, marker in markers.items():
            # Select rows corresponding to the current cell type
            subset = filtered_cell_type_df[filtered_cell_type_df["cell_type"] == cell_type]

            # Get the unit locations for this subset
            unit_ids = subset.index
            subset_unit_locations = unit_locations[unit_ids, :]

            # Plot the subset with the appropriate marker
            axes[i].scatter(
                subset_unit_locations[:, 0], subset_unit_locations[:, 1], label=cell_type, marker=marker, s=14
            )

        axes[i].set_title(f"{date_str}")
        axes[i].set_ylim([-50, 1400])
        axes[i].set_xticks([])
        axes[i].set_xlabel("")

        if i == 0:
            axes[i].set_ylabel("Position along shank (um)")
        else:
            axes[i].set_yticks([])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.6), title="Cell Type")

    plt.suptitle(animal_id)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    with open(output_batch_file, "wb") as f:
        pickle.dump(batch_data, f)
    print(f"\nBatch data saved to {output_batch_file}")


# %% Load from batch file


def load_and_plot_unit_loc_from_batch(batch_file):
    # Load batch data from file
    with open(batch_file, "rb") as f:
        batch_data = pickle.load(f)

    sorted_data = sort_data_folders_by_date(batch_data)
    relative_days = [day + 1 for day in calculate_relative_days(sorted_data)]
    # Set up plotting
    n_dates = len(batch_data)
    fig, axes = plt.subplots(nrows=1, ncols=n_dates, figsize=(1.4 * n_dates, 5))

    if n_dates == 1:
        axes = [axes]

    for i, ((date_str, data), rel_day) in enumerate(zip(sorted_data, relative_days)):
        ax = axes[i]

        cell_type_df = data["cell_type_df"]
        unit_locations = data["unit_locations"]
        animal_id = data["animal_id"]

        markers = {"Pyramidal Cell": "^", "Narrow Interneuron": "s", "Wide Interneuron": "o"}
        colors = {"Pyramidal Cell": "C0", "Narrow Interneuron": "C1", "Wide Interneuron": "C2"}

        for cell_type, marker in markers.items():
            subset = cell_type_df[cell_type_df["cell_type"] == cell_type]
            if subset.empty:
                continue

            unit_ids = subset.index.to_numpy()
            subset_ul = unit_locations[unit_ids, :]

            ax.scatter(
                subset_ul[:, 0],
                subset_ul[:, 1],
                label=cell_type,
                marker=marker,
                s=20,
                color=colors[cell_type],
                alpha=0.7,
            )

        ax.set_title(f"Day {rel_day}")
        ax.set_ylim([-50, 1700])

        if i == 0:
            ax.set_ylabel("Position along shank (µm)")
        else:
            ax.set_yticks([])

        ax.set_xticks([])
        ax.set_xlabel("")

    # Add a single legend outside the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.6), title="Cell Type")

    plt.suptitle(animal_id)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)


def sort_data_folders_by_date(batch_data):
    def parse_date(date_str):
        return datetime.strptime(date_str, "%d-%b-%Y")

    return sorted(batch_data.items(), key=lambda x: parse_date(x[0]))


def calculate_relative_days(sorted_data):
    first_date = datetime.strptime(sorted_data[0][0], "%d-%b-%Y")
    relative_days = [(datetime.strptime(date_str, "%d-%b-%Y") - first_date).days for date_str, _ in sorted_data]
    return relative_days


def load_and_plot_cell_types_from_batch(batch_file):
    # Load batch data from file
    with open(batch_file, "rb") as f:
        batch_data = pickle.load(f)

    # Set up plotting
    n_dates = len(batch_data)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 5))

    # Prepare data for plotting
    dates = []
    pyramidal_counts = []
    narrow_counts = []
    wide_counts = []

    for date_str, data in batch_data.items():
        cell_type_df = data["cell_type_df"]
        animal_id = data["animal_id"]

        # Count the number of each cell type
        pyramidal_count = len(cell_type_df[cell_type_df["cell_type"] == "Pyramidal Cell"])
        narrow_count = len(cell_type_df[cell_type_df["cell_type"] == "Narrow Interneuron"])
        wide_count = len(cell_type_df[cell_type_df["cell_type"] == "Wide Interneuron"])

        # Append to lists
        dates.append(date_str)
        pyramidal_counts.append(pyramidal_count)
        narrow_counts.append(narrow_count)
        wide_counts.append(wide_count)

    # Convert lists to numpy arrays for stacking
    pyramidal_counts = np.array(pyramidal_counts)
    narrow_counts = np.array(narrow_counts)
    wide_counts = np.array(wide_counts)

    # Plotting the stacked bar chart
    bar_width = 0.35
    indices = np.arange(n_dates)

    ax.bar(indices, pyramidal_counts, bar_width, label="Pyramidal Cell", color="C0")
    ax.bar(indices, narrow_counts, bar_width, bottom=pyramidal_counts, label="Narrow Interneuron", color="C1")
    ax.bar(
        indices, wide_counts, bar_width, bottom=pyramidal_counts + narrow_counts, label="Wide Interneuron", color="C2"
    )

    # Set labels and title
    rel_days = [days + 1 for days in file_util.convert_dates_to_relative_days(dates)]
    ax.set_xlabel("Day")
    ax.set_ylabel("Number of Cells")
    ax.set_title("Number of Cell Types per Day")
    ax.set_xticks(indices)
    ax.set_xticklabels(rel_days)

    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1, 0.6), title="Cell Type")

    plt.suptitle(animal_id)
    plt.tight_layout()
    plt.subplots_adjust(right=0.82)


# %%

if __name__ == "__main__":
    plt.style.use("default")
    plt.style.use("seaborn-v0_8-dark")
    output_batch_file = "batch_data_ICMS100.pkl"
    main(data_folder=None, output_batch_file=output_batch_file)

    load_and_plot_unit_loc_from_batch(batch_file=output_batch_file)
    load_and_plot_cell_types_from_batch(batch_file=output_batch_file)
