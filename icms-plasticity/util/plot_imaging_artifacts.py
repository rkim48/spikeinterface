from util.load_data import load_data
import util.file_util as file_util
import spikeinterface.full as si

data_folders = file_util.file_dialog("C://data")
sorted_data_folders = file_util.sort_data_folders(data_folders)
animalID = file_util.get_animal_id(sorted_data_folders[0])

for session_num, data_folder in enumerate(sorted_data_folders):
    animalID = file_util.get_animal_id(data_folder)

    dataloader = load_data(
        data_folder, make_folder=True, save_folder_name="batch_sort", first_N_files=2, server_mount_drive="S:"
    )
    rec = dataloader.recording
    rec_cr = si.common_reference(rec, operator="median", reference="global")
    # %%
    time_range = [103, 103.5]
    time_range = [46, 46.5]
    si.plot_traces(rec_cr, backend="matplotlib", time_range=time_range, order_channel_by_depth=True)
