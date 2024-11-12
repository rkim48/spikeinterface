import tkinter as tk
from tkfilebrowser import askopendirnames
from util.is_valid_folder import is_valid_folder
import os
import re
import json
from datetime import datetime


# %%
def file_dialog(starting_dir="E:\\robin"):
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    selected_folders = askopendirnames(initialdir=starting_dir)

    if selected_folders:
        valid_folders = [folder for folder in selected_folders if is_valid_folder(folder)]
        print("\nValid folders:")
        for folder in valid_folders:
            print(f"- {folder}")
    else:
        print("No folders selected")

    root.destroy()
    return valid_folders


def newest_subfolder(directory):
    try:
        subfolders = [f.path for f in os.scandir(directory) if f.is_dir() and "Processed" in f.name]
        newest_folder = max(subfolders, key=os.path.getmtime)
        return newest_folder
    except ValueError:
        print("Directory is empty or does not contain any subfolders with 'Processed' in the name")
        return None


def load_timing_params(path="timing_params.json"):
    with open(path, "r") as json_file:
        timing_params = json.load(json_file)
    return timing_params


def get_animal_id(folder):
    # Regular expression pattern for animal ID (e.g., ICMS followed by numbers)
    pattern = r"ICMS\d+"

    # Search for the pattern in the folder path
    match = re.search(pattern, folder)
    if match:
        return match.group()
    else:
        raise ValueError("Animal ID not found in the folder path.")


def get_date_str(folder):
    # Regular expression pattern for date string (e.g., 02-Nov-2023)
    pattern = r"\d{2}-[A-Za-z]{3}-\d{4}"

    # Search for the pattern in the folder path
    match = re.search(pattern, folder)
    if match:
        return match.group()
    else:
        raise ValueError("Date string not found in the folder path.")


def sort_data_folders(data_folders):
    # returns sorted data folders by date
    def parse_date_from_path(path):
        date_str = path.split("\\")[-1]  # Extract the date part of the string
        return datetime.strptime(date_str, "%d-%b-%Y")  # Parse the date

    return sorted(data_folders, key=parse_date_from_path)


def convert_dates_to_relative_days(date_string_arr):
    sorted_dates = sort_data_folders(date_string_arr)
    first_date = datetime.strptime(sorted_dates[0].split("\\")[-1], "%d-%b-%Y")
    relative_days = []
    for folder in sorted_dates:
        current_date = datetime.strptime(folder.split("\\")[-1], "%d-%b-%Y")
        days_diff = (current_date - first_date).days
        relative_days.append(days_diff)

    return relative_days


# if __name__ == "__main__":
# valid_folders = file_dialog()
# a = load_timing_params()
