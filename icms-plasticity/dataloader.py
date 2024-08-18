import os
import re
import scipy.io as sio
import numpy as np
from spikeinterface import full as si, extractors as se
import datetime
from pathlib import Path
from util.impedance_analyzer import ImpedanceAnalyzer
import pandas as pd
# import psignifit as ps
import pickle
from probeinterface.io import read_probeinterface


class DataLoader:
    def __init__(self, data_folder: str, make_folder: bool, save_folder_name: str = None, server_mount_drive: str = "S:"):
        self.server_mount_drive = server_mount_drive
        self.data_folder = Path(data_folder)
        self.sub_folder = self.data_folder / "ChannelVolumetric"
        self.get_animal_id()
        self.make_folder = make_folder
        self.save_folder_name = save_folder_name
        print(f"Data folder set to {self.data_folder}")

    def get_animal_id(self):
        match = re.search(r'ICMS\d+', str(self.data_folder))
        if match:
            self.animalID = match.group()
        else:
            raise ValueError(
                f"ICMS* pattern not found in the path: {self.data_folder}")

    def get_save_folder(self):
        if self.save_folder_name:
            # A specific save folder name is provided
            self.save_folder = self.data_folder / self.save_folder_name
            if self.make_folder:
                # If folder creation is requested and the folder does not exist, create it
                os.makedirs(self.save_folder, exist_ok=True)
                print(
                    f"Save folder created or confirmed at: {self.save_folder}")
            elif not self.save_folder.exists():
                # If folder creation is not requested but the folder does not exist, raise an error
                raise ValueError(
                    f"Specified save folder does not exist and make_folder is False: {self.save_folder}")
            print(f"Using specified save folder: {self.save_folder}")
        else:
            if self.make_folder:
                # No specific save folder name provided, create a new folder with timestamp
                current_date_string = datetime.datetime.now().strftime('%d-%b-%Y_%H%M%S')
                self.save_folder = self.data_folder / \
                    f"Processed_{current_date_string}"
                os.makedirs(self.save_folder, exist_ok=True)
                print(f"Save folder created at: {self.save_folder}")
            else:
                # No specific save folder name provided and not making a new folder, find the newest 'Processed_' folder
                self.save_folder = self.find_newest_processed_folder()
                print(f"Using existing save folder: {self.save_folder}")

    def find_newest_processed_folder(self):
        processed_folders = [f for f in os.listdir(
            self.data_folder) if f.startswith("Processed_")]
        if processed_folders:
            def parse_folder_date(folder_name):
                try:
                    return datetime.datetime.strptime(folder_name.split("_")[1], '%d-%b-%Y_%H%M%S')
                except ValueError:
                    return datetime.datetime.strptime(folder_name.split("_")[1], '%d-%b-%Y')

            newest_folder = max(processed_folders,
                                key=lambda x: parse_folder_date(x))
            return self.data_folder / newest_folder
        else:
            raise ValueError("No existing 'Processed_' folders found.")

    def load_sorting(self, curated_flag=False):
        if curated_flag:
            sorting_folder_name = 'sorting_curated'
        else:
            sorting_folder_name = 'sorting'

        sorting_folder_path = Path(self.save_folder) / sorting_folder_name

        try:
            self.sorting = si.read_sorter_folder(sorting_folder_path)
            print(f"Loaded sorting data from {sorting_folder_path}")
            return self.sorting
        except Exception as e:
            print(f"Error loading sorting data: {e}")
            return None

    def remove_high_impedance_ch(self):
        impedance_analyzer = ImpedanceAnalyzer()
        impedance_analyzer.get_intan_impedances(animal_id=self.animalID,
                                                server_mount_drive=self.server_mount_drive)
        filtered_impedances, good_channels = impedance_analyzer.get_good_impedances(
            threshold=2e6)
        good_ripple_channels = impedance_analyzer.intan_to_ripple(
            good_channels)
        good_indices = sorted(good_ripple_channels - 1)
        channel_ids = self.recording.channel_ids
        self.recording = self.recording.channel_slice(
            channel_ids=channel_ids[good_indices])

    def attach_probe(self):
        pi = read_probeinterface(
            'util\\net32Ch.json')
        probe = pi.probes[0]
        self.recording.set_probe(probe, in_place=True)

    def get_base_server_folder(self):
        """Determine the base server folder from the data folder."""
        # hard-coded!!!
        # print(f"Using server mount drive: '{self.server_mount_drive}'")
        parts = os.path.normpath(self.data_folder).split(os.sep)
        icms_id = self._get_part_matching(parts, "ICMS")
        date = self._get_part_matching(parts, r'\d{2}-\w{3}-\d{4}')
        self.date = date
        subfolder = "ChannelVolumetric"
        self.base_server_folder = os.path.join(self.server_mount_drive,
                                               icms_id, "Keys", date, subfolder)
        assert (os.path.isdir(self.base_server_folder))

    def _get_part_matching(self, parts, pattern):
        """Helper to get a part of the path matching a given pattern."""
        return next(part for part in parts if re.match(pattern, part))

    def _get_ns5_file_matching_base(self, base_name, all_files):
        """Helper to get the NS5 file that matches a given base name."""
        return next((f for f in all_files if f.startswith(base_name) and f.endswith('.ns5')), None)

    def get_ns5_and_mat_files(self, first_N_files=4):
        """Extract NS5 and MAT files from the data folder."""
        all_files = os.listdir(self.sub_folder)
        n_files = min(len(all_files), first_N_files)
        self.mat_files = [f for f in all_files if re.match(
            r'block\d+_H\d+M\d+_behavior.mat', f)]
        self.mat_files = self.mat_files[:n_files]
        base_names = [re.match(
            r'(block\d+_H\d+M\d+)_behavior.mat', f).group(1) for f in self.mat_files]
        self.ns5_files = [self._get_ns5_file_matching_base(
            base_name, all_files) for base_name in base_names]

    def get_key_files(self):
        """Extract key information from server folder."""
        key_paths = [os.path.join(self.base_server_folder,
                                  f.replace('_behavior', '_key')) for f in self.mat_files]
        self.all_currents, self.all_depths, self.all_stim_timestamps, \
            self.all_stim_start_stop, self.all_image_start_stop, self.all_behave_responses, \
            self.all_response_times, self.all_frame_timestamps = [], [], [], [], [], [], [], []
        self.all_modified_stim_timestamps = []
        self.all_modified_frame_timestamps = []
        self.all_modified_img_start_stop = []
        for i, key_path in enumerate(key_paths):
            mat_data = sio.loadmat(key_path, simplify_cells=True)
            self._extract_key_data_from_mat(mat_data, i)
        flattened_list = [
            item for sublist in self.all_stim_timestamps for item in sublist]
        assert (flattened_list == sorted(flattened_list))
        flattened_list = [
            item for sublist in self.all_stim_start_stop for item in sublist]
        assert (flattened_list == sorted(flattened_list))
        flattened_list = [
            item for sublist in self.all_image_start_stop for item in sublist]
        assert (flattened_list == sorted(flattened_list))

        self._correct_depths()

    def check_premature_recording_stop(self, stim_start_stop):
        # in case ephys recording stopped early
        for index, sublist in enumerate(stim_start_stop):
            if type(sublist) == float:
                sublist = [sublist, sublist + 1]
                print(sublist)
                stim_start_stop[index] = sublist
            if len(sublist) < 2:
                break
        else:
            # No premature stop found, all trials are complete
            return -1
        last_trial = index - 1
        print(
            f'Prematurely stopped ephys recording found: using trials 1-{last_trial+1}')
        return last_trial

    def fill_with_nans(self, last_trial, input_list):
        modified_input_list = input_list.copy()
        for i in range(last_trial + 1, len(modified_input_list)):
            modified_input_list[i] = [np.nan]
        return modified_input_list

    def _extract_key_data_from_mat(self, mat_data, index):
        """Helper to extract specific key data from the MAT file."""
        self.get_fs()
        self.get_start_times()

        trial_array = mat_data['trialDataStruct']
        currents = [trial['Current'] for trial in trial_array]
        depths = [trial['Depth'] for trial in trial_array]
        stim_timestamps = [(np.array(trial['StimTimestamps']) * self.fs
                            + self.start_times[index] * self.fs).tolist() for trial in trial_array]
        stim_start_stop = [(np.array(trial['StimStartStop']) * self.fs
                            + self.start_times[index] * self.fs).tolist() for trial in trial_array]
        img_start_stop = [(np.array(trial['ImageStartStop']) * self.fs
                           + self.start_times[index] * self.fs).tolist() for trial in trial_array]
        frame_timestamps = [(np.array(trial['FrameTimestamps']) * self.fs
                             + self.start_times[index] * self.fs).tolist() for trial in trial_array]
        behave_responses = [trial['Response'] for trial in trial_array]
        response_times = [trial['ResponseTime'] for trial in trial_array]

        # check recording end prematurely
        last_trial = self.check_premature_recording_stop(stim_start_stop)
        if last_trial > -1:
            # Adjust the lists to include data up to and including the last complete trial
            self.modified_stim_timestamps = self.fill_with_nans(
                last_trial, stim_timestamps)
            self.modified_frame_timestamps = self.fill_with_nans(
                last_trial, frame_timestamps)
            self.modified_img_start_stop = self.fill_with_nans(
                last_trial, img_start_stop)

            self.all_modified_stim_timestamps.extend(
                self.modified_stim_timestamps)
            self.all_modified_frame_timestamps.extend(
                self.modified_frame_timestamps)
            self.all_modified_img_start_stop.extend(
                self.modified_img_start_stop)
            stim_timestamps = stim_timestamps[:last_trial + 1]
            stim_start_stop = stim_start_stop[:last_trial + 1]

            img_start_stop = img_start_stop[:last_trial + 1]
            frame_timestamps = frame_timestamps[:last_trial + 1]

        if type(img_start_stop[-1]) != list:
            img_start_stop[-1] = [img_start_stop[-1],
                                  img_start_stop[-1] + 30000 * 5]

        self.all_currents.extend(currents)
        self.all_depths.extend(depths)
        self.all_stim_timestamps.extend(stim_timestamps)
        self.all_stim_start_stop.extend(stim_start_stop)
        self.all_image_start_stop.extend(img_start_stop)
        self.all_frame_timestamps.extend(frame_timestamps)
        self.all_behave_responses.extend(behave_responses)
        self.all_response_times.extend(response_times)

    def get_fs(self):
        file_path = os.path.join(self.sub_folder, self.ns5_files[0])
        rec_1 = se.read_blackrock(
            file_path=file_path, stream_id='5')
        # rec_1 = se.read_blackrock(
        #     file_path=file_path, stream_id='5', block_index=1)
        self.fs = rec_1.get_sampling_frequency()

    def get_start_times(self):
        num_frames_arr = []
        self.start_times = []
        for i, ns5_file in enumerate(self.ns5_files):
            file_path = os.path.join(self.sub_folder, ns5_file)
            rec_i = se.read_blackrock(
                file_path=file_path, stream_id='5')
            # rec_i = se.read_blackrock(
            #     file_path=file_path, stream_id='5', block_index=1)
            num_frames_arr.append(rec_i.get_num_frames() / self.fs)
        self.start_times = [0] + np.cumsum(num_frames_arr[:-1]).tolist()

    def append_recordings(self):
        """Append all the sub-recordings."""
        self.recordings_list = []
        self.time_vectors = []
        for i, ns5_file in enumerate(self.ns5_files):
            file_path = os.path.join(self.sub_folder, ns5_file)
            rec_i = se.read_blackrock(
                file_path=file_path, stream_id='5')
            self.recordings_list.append(rec_i)
        durations = [rec_i.get_num_frames() / self.fs for rec_i in
                     self.recordings_list]
        self.start_times = [0] + np.cumsum(durations[:-1]).tolist()

        self.time_vectors = [self.get_time_vector(rec_i, start_time)
                             for rec_i, start_time in
                             zip(self.recordings_list, self.start_times)]
        self.recording = si.append_recordings(self.recordings_list)

    def get_time_vector(self, subrecording, start_time):
        """Return a time vector for the given recording, starting from the start_time."""
        num_samples = subrecording.get_num_frames()
        return (np.arange(0, num_samples) / self.fs) + start_time

    def _correct_depths(self):
        # older sessions don't use depth but ripple_id
        ia = ImpedanceAnalyzer()
        if any(depth > 20 for depth in self.all_depths):
            # assume intan channels and convert to depth
            ripple_ids = np.unique(self.all_depths)
            ripple_ids = [
                ripple_id for ripple_id in ripple_ids if ripple_id != 0]
            depth_idx = ia.ripple_to_depth2(ripple_ids)
            mapping = dict(zip(ripple_ids + [0], depth_idx + [0]))
            self.all_depths = [mapping.get(val, val)
                               for val in self.all_depths]

    def get_concatenated_recording(self):
        """Concatenate all the recordings and slice."""
        recording = si.concatenate_recordings(self.recordings_list)
        self.channel_ids = recording.get_channel_ids()
        self.recording = recording.channel_slice(self.channel_ids[0:32])

    def get_trial_dataframe(self):
        num_trials = len(self.all_currents)
        if len(self.all_stim_timestamps) < num_trials:
            df_stim_ts = self.all_modified_stim_timestamps
            df_frame_ts = self.all_modified_frame_timestamps
            df_img_start_stop = self.all_modified_img_start_stop
        else:
            df_stim_ts = self.all_stim_timestamps
            df_frame_ts = self.all_frame_timestamps
            df_img_start_stop = self.all_image_start_stop
        trial_df = pd.DataFrame({
            'trial': range(1, num_trials + 1),
            'response': self.all_behave_responses,
            'response_time': self.all_response_times,
            'current': self.all_currents,
            'channel': self.all_depths,
            'stim_timestamps': df_stim_ts,
            'frame_timestamps': df_frame_ts,
            'image_start': [sublist[0] if isinstance(sublist, list) and sublist != [np.nan] else np.nan for sublist in df_img_start_stop],
            'image_stop': [sublist[1] if isinstance(sublist, list) and sublist != [np.nan] else np.nan for sublist in df_img_start_stop]
        })
        self.trial_df = trial_df
        return trial_df

    def get_block_detection_threshold(self):
        options = {'sigmoidName': 'norm',
                   'expType': 'YesNo', 'plotThresh': 'false'}
        num_blocks = len(self.ns5_files)
        trials_per_block = int(self.trial_df.shape[0] / num_blocks)
        block_thresholds = {}
        for block in range(num_blocks):
            start_idx = trials_per_block * block
            end_idx = start_idx + trials_per_block
            sub_df = self.trial_df.iloc[start_idx:end_idx]
            catch_trials = sub_df[sub_df['channel'] == 0]
            false_positives = catch_trials['response'].sum()
            catch_trial_row = np.array([0, false_positives, len(catch_trials)])

            unique_channels = sub_df[sub_df['channel']
                                     != 0]['channel'].unique()

            thresholds = {}

            for channel in unique_channels:
                channel_data = sub_df[sub_df['channel'] == channel]
                grouped_data = channel_data.groupby('current').agg(
                    responses=('response', 'sum'),
                    total_trials=('trial', 'count')
                ).reset_index()

                psignifit_data = np.column_stack([
                    grouped_data['current'],
                    grouped_data['responses'],
                    grouped_data['total_trials']
                ])
                psignifit_data = np.row_stack(
                    [catch_trial_row, psignifit_data])

                result = ps.psignifit(psignifit_data, options)
                threshold = ps.getThreshold(result, 0.5)[0]
                thresholds[channel] = np.round(threshold, 2)
                # ps.psigniplot.plotPsych(result)

            block_thresholds[block] = thresholds

        self.block_thresholds = block_thresholds

    def save_block_thresholds(self):
        with open(Path(self.data_folder) / 'block_thresholds.pkl', 'wb') as file:
            pickle.dump(self.block_thresholds, file)


if __name__ == "__main__":
    # data_folders = get_multiple_data_folders()
    # data_folder = data_folders[0]
    data_folder = 'C:\\data\\ICMS98\\02-Nov-2023'
    dataloader = DataLoader(data_folder, make_folder=False)
    dataloader.get_save_folder()
    dataloader.get_base_server_folder()
    dataloader.get_ns5_and_mat_files()
    dataloader.get_key_files()
    dataloader.append_recordings()
    dataloader.get_concatenated_recording()
    dataloader.get_trial_dataframe()
    dataloader.get_block_detection_threshold()
