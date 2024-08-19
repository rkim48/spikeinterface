import shutil
from pathlib import Path
import tkinter as tk
from tkfilebrowser import askopendirnames
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


def is_valid_folder(folder_path, required_subfolder=None, file_patterns=None):
    if required_subfolder is None:
        required_subfolder = "ChannelVolumetric"
    if file_patterns is None:
        file_patterns = [f"block{i}_H.*" for i in range(1, 5)]
    if not os.path.isdir(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return False

    required_subfolder_path = os.path.join(folder_path, required_subfolder)
    if not os.path.isdir(required_subfolder_path):
        print(
            f"The required subfolder {required_subfolder} is missing from {folder_path}.")
        return False

    subfolder_contents = os.listdir(required_subfolder_path)
    for pattern in file_patterns:
        mat_pattern = re.compile(f"{pattern}.mat")
        ns5_pattern = re.compile(f"{pattern}.ns5")

        mat_files = [f for f in subfolder_contents if mat_pattern.match(f)]
        ns5_files = [f for f in subfolder_contents if ns5_pattern.match(f)]

        if not mat_files or not ns5_files:
            print(
                f"Missing files for pattern {pattern} in {required_subfolder_path}.")
            return False
        if len(mat_files) > 1 or len(ns5_files) > 1:
            print(
                f"Multiple files found for pattern {pattern} in {required_subfolder_path}.")
            return False

    return True
