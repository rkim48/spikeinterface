import numpy as np
from datetime import datetime
import shutil
import os
from batch_process.util.ppt_image_inserter import PPTImageInserter
from collections import defaultdict
import re
import batch_process.util.file_util as file_util
import batch_process.postprocessing.responses.response_plotting_util as rpu
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import pandas as pd


# Plot particular units across time for animal ID
# Specify stim channel and current
# Need dictionary date and unit id list
animal_date_dict = {
    "ICMS92": {},
    "ICMS93": {},
    "ICMS98": {},
    "ICMS100": {},
    "ICMS101": {}
}

animalID = "ICMS92"
tracked_unit_id = 'A'
date_unit_id_dict = {
    "30-Aug-2023": [11, 12],
    "01-Sep-2023": [5, 8],
    "06-Sep-2023": [9],
    "08-Sep-2023": [8],
    "12-Sep-2023": [7, 12],
    "14-Sep-2023": [8, 9],
    "19-Sep-2023": [10, 11],
    "21-Sep-2023": [10, 11],
    "25-Sep-2023": [8],
    "27-Sep-2023": [11]
}
animal_date_dict[animalID][tracked_unit_id] = date_unit_id_dict


tracked_unit_id = 'B'
date_unit_id_dict = {
    "30-Aug-2023": [13, 14],
    "01-Sep-2023": [6],
    "06-Sep-2023": [6],
    "08-Sep-2023": [999],
    "12-Sep-2023": [9],
    "14-Sep-2023": [10, 11],
    "19-Sep-2023": [12, 13],
    "21-Sep-2023": [12, 14],
    "25-Sep-2023": [10],
    "27-Sep-2023": [12, 13]
}
animal_date_dict[animalID][tracked_unit_id] = date_unit_id_dict


tracked_unit_id = 'C'
date_unit_id_dict = {
    "30-Aug-2023": [2],
    "01-Sep-2023": [1],
    "06-Sep-2023": [999],
    "08-Sep-2023": [1, 11],
    "12-Sep-2023": [2],
    "14-Sep-2023": [2],
    "19-Sep-2023": [2],
    "21-Sep-2023": [5],
    "25-Sep-2023": [3],
    "27-Sep-2023": [3, 4]
}
animal_date_dict[animalID][tracked_unit_id] = date_unit_id_dict


animalID = "ICMS93"
tracked_unit_id = 'A'
date_unit_id_dict = {
    "30-Aug-2023": [21],
    "06-Sep-2023": [10],
    "14-Sep-2023": [10],
    "20-Sep-2023": [20],
    "22-Sep-2023": [19],
    "26-Sep-2023": [12],
    "29-Sep-2023": [23],
    "04-Oct-2023": [22],
    "06-Oct-2023": [20],
}
animal_date_dict[animalID][tracked_unit_id] = date_unit_id_dict


tracked_unit_id = 'B'
date_unit_id_dict = {
    "30-Aug-2023": [18],
    "06-Sep-2023": [8],
    "14-Sep-2023": [9],
    "20-Sep-2023": [10],
    "22-Sep-2023": [12],
    "26-Sep-2023": [11],
    "29-Sep-2023": [14],
    "04-Oct-2023": [9],
    "06-Oct-2023": [17],
}
animal_date_dict[animalID][tracked_unit_id] = date_unit_id_dict

animalID = "ICMS98"
tracked_unit_id = 'A'
date_unit_id_dict = {
    "20-Oct-2023": [6],
    "24-Oct-2023": [6],
    "26-Oct-2023": [9],
    "31-Oct-2023": [5],
    "02-Nov-2023": [6],
    "07-Nov-2023": [6],
    "14-Nov-2023": [999],
    "17-Nov-2023": [7],
    "20-Nov-2023": [5],
    "22-Nov-2023": [6],
}

animal_date_dict[animalID][tracked_unit_id] = date_unit_id_dict

tracked_unit_id = 'B'
date_unit_id_dict = {
    "20-Oct-2023": [13, 16],
    "24-Oct-2023": [14],
    "26-Oct-2023": [16],
    "31-Oct-2023": [12],
    "02-Nov-2023": [15],
    "07-Nov-2023": [11],
    "14-Nov-2023": [999],
    "17-Nov-2023": [16],
    "20-Nov-2023": [16],
    "22-Nov-2023": [15],
}

animal_date_dict[animalID][tracked_unit_id] = date_unit_id_dict


tracked_unit_id = 'C'
date_unit_id_dict = {
    "20-Oct-2023": [11],
    "24-Oct-2023": [11],
    "26-Oct-2023": [14],
    "31-Oct-2023": [9],
    "02-Nov-2023": [11],
    "07-Nov-2023": [10],
    "14-Nov-2023": [10],
    "17-Nov-2023": [13],
    "20-Nov-2023": [10],
    "22-Nov-2023": [13],
}

animal_date_dict[animalID][tracked_unit_id] = date_unit_id_dict


tracked_unit_id = 'D'
date_unit_id_dict = {
    "20-Oct-2023": [1],
    "24-Oct-2023": [1],
    "26-Oct-2023": [4],
    "31-Oct-2023": [2],
    "02-Nov-2023": [2],
    "07-Nov-2023": [2],
    "14-Nov-2023": [2],
    "17-Nov-2023": [1],
    "20-Nov-2023": [999],
    "22-Nov-2023": [999],
}

animal_date_dict[animalID][tracked_unit_id] = date_unit_id_dict
# %% Plot

plot_flag = True
# Create the directory to save the figures
save_dir = Path("C:/data/") / animalID / "Figures"
save_dir.mkdir(parents=True, exist_ok=True)

base_path = Path("C:/data/") / animalID / "Behavior"
if not os.path.exists(base_path):
    base_path = Path("C:/data/") / animalID

# assume channels used for all sessions is in third session
pkl_path = base_path / list(date_unit_id_dict.keys()
                            )[2] / "batch_sort/session_responses.pkl"
with open(pkl_path, "rb") as file:
    session_responses = pickle.load(file)
stim_conds = session_responses.unit_responses[0].show_stim_conditions()
stim_channels = np.unique([stim_cond[0] for stim_cond in stim_conds])

# TODO get stim channels
for stim_channel in stim_channels:
    for stim_current in [2, 3, 4, 5]:
        if plot_flag:
            fig, axes = plt.subplots(
                3, len(date_unit_id_dict), figsize=(16, 6))

        for date_index, (data_folder, unit_ids) in enumerate(date_unit_id_dict.items()):
            print(data_folder)
            pkl_path = base_path / data_folder / "batch_sort/session_responses.pkl"

            try:
                with open(pkl_path, "rb") as file:
                    session_responses = pickle.load(file)
            except FileNotFoundError:
                # If the file is not found, turn off the axes for this column
                if plot_flag:
                    for i in range(3):
                        axes[i, date_index].axis('off')
                continue

            valid_units = False  # Flag to track if any valid unit is found

            for unit_id in unit_ids:
                if unit_id not in session_responses.unit_ids:
                    continue  # Skip if the unit doesn't exist in this session
                if plot_flag:
                    axes[0, date_index].set_title(data_folder)

                ur = session_responses.get_unit_response(unit_id)

                all_conditions = ur.show_stim_conditions()
                if (stim_channel, stim_current) not in all_conditions:
                    continue  # Skip if the condition doesn't exist

                scr = ur.get_stim_response(stim_channel, stim_current)
                tr = scr.train_response
                pr = scr.pulse_response

                # Plot the responses
                if plot_flag:
                    rpu.plot_pulse_firing_rate(
                        axes[0, date_index], pr, 'C0')
                    rpu.plot_train_firing_rate(
                        axes[1, date_index], tr, 'C0')
                    rpu.plot_template(scr, unit_id, y_negative=-
                                      200, ax=axes[2, date_index])

                    valid_units = True  # Mark as valid if we plot something

            # If no valid units or conditions were found, turn off the axes for this date
            if plot_flag:
                if not valid_units:
                    for i in range(3):
                        axes[i, date_index].axis('off')

                # Set y-limits for all valid axes
                axes[0, date_index].set_ylim([0, 200])
                axes[1, date_index].set_ylim([0, 100])

                if date_index == 0:
                    axes[0, 0].set_ylabel("Firing Rate (Hz)")
                    axes[0, 0].set_xlabel("Time (ms)")
                    axes[1, 0].set_ylabel("Firing Rate (Hz)")
                    axes[1, 0].set_xlabel("Time (ms)")

        # Add a title and adjust layout
        if plot_flag:
            plt.suptitle(f"{animalID}: Unit {tracked_unit_id} ch.{
                         stim_channel} at {stim_current} uA")
            plt.tight_layout()

            # Save the figure with a filename based on stim_channel and stim_current
            save_path = save_dir / \
                f"{animalID}_Unit_{tracked_unit_id}_Ch{
                    stim_channel}_Curr{stim_current}.png"
            plt.savefig(save_path, dpi=300)
            plt.show()
            plt.close()

# %%
def extract_unit_name(image_path):
    """
    Extracts the unit name ('A' or 'B') from the filename.
    Assumes filenames are in the format 'Unit_<unit_name>_*.png'.
    """
    # Use a regular expression to extract the unit name ('A' or 'B') from the filename
    match = re.search(r"Unit_([A|B|C|D])_", image_path.name)
    if match:
        return match.group(1)
    return None


# Collect all image paths from the directory
# Adjust the pattern if needed
image_paths = list(save_dir.glob("*.png"))

# Group images by unit name ('A' or 'B')
images_by_unit = defaultdict(list)
for image_path in image_paths:
    unit_name = extract_unit_name(image_path)
    if unit_name is not None:
        images_by_unit[unit_name].append(image_path)

# Initialize the PPTImageInserter with grid dimensions and title font size
ppt_inserter = PPTImageInserter(grid_dims=(
    2, 1), spacing=(0.05, 0.05), title_font_size=16)

# Add images to the PPT, grouping them by unit name ('A' or 'B')
for unit_name, image_paths in images_by_unit.items():
    # Create a title for each unit (A or B)
    slide_title = f"Unit {unit_name}"
    ppt_inserter.add_slide(slide_title)

    # Add the images to the slide
    for image_path in image_paths:
        ppt_inserter.add_image(str(image_path), slide_title=slide_title)

# Save the PowerPoint file with the session path stem in the filename
ppt_inserter.save(save_dir / f"units_over_time.pptx")

# %%


animal_id = "ICMS98"
source_path = f"C:/data/{animal_id}"
copy_path = f"//10.129.151.108/xieluanlabs/xl_stimulation/Robin/Behavioral/Analysis/Ephys stim responses/{
    animal_id}"

# Ensure the copy path exists
os.makedirs(copy_path, exist_ok=True)

# Loop through each date folder in the source path
for date_folder in os.listdir(source_path):
    date_folder_path = os.path.join(source_path, date_folder)

    # Check if it's a directory and contains 'batch_sort/figures/stim_responses.pptx'
    ppt_file = os.path.join(date_folder_path, "batch_sort",
                            "figures", "stim_responses.pptx")

    if os.path.isdir(date_folder_path) and os.path.exists(ppt_file):
        try:
            # Format the destination file name
            formatted_date = datetime.strptime(
                date_folder, "%d-%b-%Y").strftime("%d-%b-%Y")
            dest_file_name = f"stim_responses_{formatted_date}.pptx"
            dest_file_path = os.path.join(copy_path, dest_file_name)

            # Copy the PowerPoint file to the destination path with the new name
            shutil.copyfile(ppt_file, dest_file_path)
            print(f"Copied: {ppt_file} to {dest_file_path}")

        except ValueError as e:
            print(f"Skipping folder '{date_folder}': {e}")
    else:
        print(f"No ppt file found in {date_folder_path}")




# %% Dataframe
rows = []
stim_currents = [3,4,5,6]

# Create the directory to save the figures
for animalID in animal_date_dict.keys():
    save_dir = Path("C:/data/") / animalID / "Figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path("C:/data/") / animalID / "Behavior"
    if not os.path.exists(base_path):
        base_path = Path("C:/data/") / animalID

    # assume channels used for all sessions is in third session
    if animalID == "ICMS100":
        continue
    if animalID == "ICMS101":
        continue

    date_unit_id_dict = animal_date_dict[animalID]['A']
    pkl_path = base_path / list(date_unit_id_dict.keys()
                                )[2] / "batch_sort/session_responses.pkl"
    with open(pkl_path, "rb") as file:
        session_responses = pickle.load(file)
    stim_conds = session_responses.unit_responses[0].show_stim_conditions()
    stim_channels = np.unique([stim_cond[0] for stim_cond in stim_conds])

    for stim_channel in stim_channels[0:1]:
        for stim_current in stim_currents:
            for tracked_unit_label, date_unit_id_dict in animal_date_dict[animalID].items():

                for date_index, (data_folder, unit_ids) in enumerate(date_unit_id_dict.items()):
                    print(data_folder)
                    pkl_path = base_path / data_folder / "batch_sort/session_responses.pkl"

                    try:
                        with open(pkl_path, "rb") as file:
                            session_responses = pickle.load(file)
                    except FileNotFoundError:
                        # If the file is not found, turn off the axes for this column
                        continue

                    for unit_id in unit_ids:
                        if unit_id not in session_responses.unit_ids:
                            continue  # Skip if the unit doesn't exist in this session

                        ur = session_responses.get_unit_response(unit_id)

                        all_conditions = ur.show_stim_conditions()
                        if (stim_channel, stim_current) not in all_conditions:
                            continue  # Skip if the condition doesn't exist

                        scr = ur.get_stim_response(stim_channel, stim_current)
                        tr = scr.train_response
                        pr = scr.pulse_response

                        train_indices = np.where(
                            (tr.fr_times > 0) & (tr.fr_times < 700))[0]
                        train_peak_fr = np.max(tr.firing_rate[train_indices])
                        train_avg_fr = np.mean(tr.firing_rate[train_indices])

                        pulse_indices = np.where(
                            (pr.fr_times > 1.5) & (pr.fr_times < 9.5))[0]

                        if len(pr.firing_rate[pulse_indices]) == 0:
                            pulse_peak_fr = 0
                            pulse_avg_fr = 0
                            pulse_latency = np.nan
                        else:
                            pulse_peak_fr = np.max(pr.firing_rate[pulse_indices])
                            pulse_avg_fr = np.mean(pr.firing_rate[pulse_indices])
                            pulse_latency = pr.fr_times[np.argmax(pr.firing_rate)]

                        # Append to metrics_df
                        rows.append({
                            "AnimalID": animalID,
                            "Session": data_folder,
                            "SessionIndex": date_index,
                            "TrackedUnitLabel": tracked_unit_label,
                            "TrainPeakFR": train_peak_fr,
                            "TrainAvgFR": train_avg_fr,
                            "PulsePeakFR": pulse_peak_fr,
                            "PulseAvgFR": pulse_avg_fr,
                            "PulsePeakLatency": pulse_latency,
                        })

metrics_df = pd.DataFrame(rows)


#%% Plot metrics
import matplotlib.pyplot as plt

animalID = "ICMS98"
metrics = ["TrainPeakFR", "TrainAvgFR", "PulsePeakFR", "PulseAvgFR", "PulsePeakLatency"]
animal_data = metrics_df[metrics_df["AnimalID"] == animalID]
num_units = len(animal_data['TrackedUnitLabel'].unique())
# Create a 2x3 subplot layout for the animal
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle(f"Metrics {animalID} with {num_units} tracked units", fontsize=12)

# Loop over each metric and plot it in the appropriate subplot
for i, metric in enumerate(metrics):
    row, col = divmod(i, 3)  # Calculate row and column index for 2x3 layout
    ax = axes[row, col]
    ax.scatter(animal_data["SessionIndex"], animal_data[metric], marker='o', s=10)
    session_medians = animal_data.groupby("SessionIndex")[metric].median()
    ax.plot(session_medians.index, session_medians.values, color='k', linestyle='-', marker='o', label="Session Median")
    ax.plot()
    ax.set_xlabel("SessionIndex")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric}")

# Remove any unused subplot (bottom-right corner in this 2x3 grid)
if len(metrics) < 6:
    fig.delaxes(axes[1, 2])

# Adjust layout to fit the main title
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
