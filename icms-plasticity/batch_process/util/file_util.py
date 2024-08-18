import shutil
from pathlib import Path
import tkinter as tk
from tkfilebrowser import askopendirnames
from util.is_valid_folder import is_valid_folder
import os
import re
import json


def create_folder(folder_path):
    folder = Path(folder_path)

    if folder.exists() and folder.is_dir():
        shutil.rmtree(folder)

    folder.mkdir(parents=True, exist_ok=True)


def file_dialog(starting_dir="E:\\robin"):
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    selected_folders = askopendirnames(initialdir=starting_dir)

    if selected_folders:
        valid_folders = [
            folder for folder in selected_folders if is_valid_folder(folder)]
        print("\nValid folders:")
        for folder in valid_folders:
            print(f"- {folder}")
    else:
        print("No folders selected")

    root.destroy()
    return valid_folders


def newest_subfolder(directory):
    try:
        subfolders = [f.path for f in os.scandir(directory) if f.is_dir() and
                      "Processed" in f.name]
        newest_folder = max(subfolders, key=os.path.getmtime)
        return newest_folder
    except ValueError:
        print("Directory is empty or does not contain any subfolders with 'Processed' in the name")
        return None


def load_timing_params(path='timing_params.json'):
    with open(path, 'r') as json_file:
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