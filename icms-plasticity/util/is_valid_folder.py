import os
import re


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


if __name__ == "__main__":
    valid_folder_path = "C:\\data\\ICMS92\\Behavior\\08-Sep-2023"
    invalid_folder_path = "C:\\data\\ICMS92\\Behavior\\11-Sep-2023"
    required_subfolder = "ChannelVolumetric"
    file_patterns = [f"block{i}_H.*" for i in range(1, 5)]

    if is_valid_folder(valid_folder_path, required_subfolder, file_patterns):
        print("The folder is valid.")
    else:
        print("The folder is not valid.")

    if is_valid_folder(invalid_folder_path, required_subfolder, file_patterns):
        print("The folder is valid.")
    else:
        print("The folder is not valid.")
