from dataloader import DataLoader

from pathlib import Path
import os


def load_data(data_folder, server_mount_drive="S:", get_recording=True, make_folder=True, load_sorting=False, save_folder_name=None, first_N_files=4):
    dataloader = DataLoader(
        data_folder, make_folder=make_folder, save_folder_name=save_folder_name, server_mount_drive=server_mount_drive)
    dataloader.get_save_folder()
    dataloader.get_base_server_folder()
    dataloader.get_ns5_and_mat_files(first_N_files=first_N_files)
    dataloader.get_key_files()
    if get_recording:
        dataloader.append_recordings()
        dataloader.get_concatenated_recording()
        dataloader.attach_probe()
        dataloader.remove_high_impedance_ch()
        dataloader.get_trial_dataframe()
    if load_sorting:
        dataloader.load_sorting()

    return dataloader


def get_dataframe(data_folder, make_folder=False):
    dataloader = DataLoader(data_folder, make_folder=make_folder)
    dataloader.get_save_folder()
    dataloader.get_base_server_folder()
    dataloader.get_ns5_and_mat_files()
    dataloader.get_key_files()
    dataloader.append_recordings()
    dataloader.get_concatenated_recording()
    df = dataloader.get_trial_dataframe()
    return df


# def load_data2(data_folder, get_recording=True, make_folder=True):
#     dataloader = DataLoader2(data_folder, make_folder=make_folder)
#     dataloader.get_ns5_and_mat_files()
#     dataloader.get_key_files()
#     if get_recording:
#         dataloader.append_recordings()
#         dataloader.get_concatenated_recording()
#         dataloader.get_trial_dataframe()

#     return dataloader


def save_block_detection_thresholds(data_folder):

    if not os.path.exists(os.path.join(data_folder, 'block_thresholds.pkl')):
        dataloader = DataLoader(data_folder, make_folder=False)
        dataloader.get_save_folder()
        dataloader.get_base_server_folder()
        dataloader.get_ns5_and_mat_files()
        dataloader.get_key_files()
        dataloader.append_recordings()
        dataloader.get_concatenated_recording()
        dataloader.get_trial_dataframe()
        dataloader.get_block_detection_threshold()
        dataloader.save_block_thresholds()


if __name__ == "__main__":
    data_folder = Path('C:\\data\\ICMS101\\27-Oct-2023')
    # dataloader = load_data(Path('C:\\data\\ICMS101\\27-Oct-2023'))
    df = get_dataframe(data_folder)
