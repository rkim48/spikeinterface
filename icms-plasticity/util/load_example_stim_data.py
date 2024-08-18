import numpy as np
# from spikeinterface.preprocessing import TrendSubtractedRecording
import spikeinterface.preprocessing as sp
import matplotlib.pyplot as plt
from util.impedance_analyzer import ImpedanceAnalyzer
from util.load_data import load_data
from probeinterface.io import read_probeinterface


def load_example_stim_data(data_folder='C:\\data\\ICMS98\\02-Nov-2023'):
    # re = load_response_extractor(data_folder)
    # unit_ids = re.unit_ids
    dataloader = load_data(data_folder, make_folder=False)
    fs = dataloader.fs
    all_stim_timestamps = dataloader.all_stim_timestamps
    # we = re.we

    recording = dataloader.recording
    pi = read_probeinterface('..//util//net32Ch.json')
    probe = pi.probes[0]
    rec_prb = recording.set_probe(probe, in_place=True)
    probe_rec = rec_prb.get_probe()
    probe_rec.to_dataframe(complete=True).loc[:, [
        "contact_ids", "device_channel_indices"]]

    rec_prb = recording.set_probe(probe, in_place=True)
    probe_rec = rec_prb.get_probe()
    probe_rec.to_dataframe(complete=True).loc[:, [
        "contact_ids", "device_channel_indices"]]
    # Keep channels under impedance threshold
    server_mount_drive = dataloader.server_mount_drive
    animalID = dataloader.animalID
    channel_ids = dataloader.channel_ids
    all_stim_timestamps = dataloader.all_stim_timestamps
    save_folder = dataloader.save_folder
    fs = dataloader.fs
    trial_df = dataloader.trial_df

    impedance_analyzer = ImpedanceAnalyzer()
    impedance_analyzer.get_intan_impedances(animal_id=animalID,
                                            server_mount_drive=server_mount_drive)
    filtered_impedances, good_channels = impedance_analyzer.get_good_impedances(
        threshold=3e6)
    good_ripple_channels = impedance_analyzer.intan_to_ripple(good_channels)
    good_indices = sorted(good_ripple_channels - 1)

    rec = recording.channel_slice(channel_ids=channel_ids[good_indices])
    return rec, all_stim_timestamps, trial_df, save_folder


def preprocess(rec, all_stim_timestamps):
    pre_stim_blank_ms = 0.5
    post_stim_blank_ms = 2.5
    stim_ts = np.concatenate(all_stim_timestamps)
    channel_ids = rec.get_channel_ids()
    locations = rec.get_channel_locations()
    sorted_channel_ids = [channel for _, channel in sorted(
        zip(locations, channel_ids), key=lambda x: x[0][1], reverse=True)]
    rec_reordered = rec.channel_slice(sorted_channel_ids)
    rec_cr = sp.common_reference(
        rec_reordered, operator="median", reference="global")

    rec_art1 = sp.remove_artifacts(
        rec_reordered, stim_ts, ms_before=pre_stim_blank_ms, ms_after=post_stim_blank_ms, mode='cubic')
    rec_filt = sp.bandpass_filter(
        rec_art1, freq_min=300, freq_max=6000, dtype='int32')
    rec_cr = sp.common_reference(
        rec_filt, operator="median", reference="global")
    rec_art2 = sp.remove_artifacts(
        rec_cr, stim_ts, ms_before=pre_stim_blank_ms, ms_after=post_stim_blank_ms, mode='linear')
    rec_art3 = sp.remove_artifacts(
        rec_cr, stim_ts, ms_before=pre_stim_blank_ms, ms_after=post_stim_blank_ms)
    traces = rec_cr.get_traces(
        start_frame=time_range[0], end_frame=time_range[1], return_scaled=True)
    return traces


def plot_traces(rec, all_stim_timestamps, time_range, ch_idx: list[int], trial_df):

    traces = preprocess(rec, all_stim_timestamps)
    stim_ts = np.concatenate(all_stim_timestamps)
    num_plot_ch = traces.shape[1]

    filtered_df = trial_df[(time_range[0] < trial_df['image_start']) & (
        trial_df['image_start'] < time_range[1])]
    frame_timestamps = filtered_df['frame_timestamps'].explode()  # samples
    response_time = filtered_df['stim_timestamps'].apply(
        lambda x: x[0]) + filtered_df['response_time']*30000.0
    stim_timestamps = filtered_df['stim_timestamps'].explode()  # samples

    # Plot the results
    offset = 100
    for i in range(num_plot_ch - 1, -1, -1):
        plt.plot(traces[:, i] + (num_plot_ch - 1 - i) *
                 offset, 'k', alpha=0.5, linewidth=1)
    plot_stim_ts = [
        ts for ts in stim_timestamps if time_range[0] <= ts <= time_range[1]]
    plot_stim_ts = np.array(plot_stim_ts) - time_range[0]
    for ts in plot_stim_ts:
        start_shade = ts
        end_shade = ts + 1 * 30
        plt.axvspan(start_shade, end_shade, color='gray', alpha=0.2)
    for rt in response_time:
        plt.axvline(rt - time_range[0], color='red',
                    linestyle='--', linewidth=1)
    for ft in frame_timestamps:
        plt.axvline(ft - time_range[0], color='blue',
                    linestyle='-', linewidth=0.5)


if __name__ == "__main__":
    # rec, all_stim_timestamps, trial_df, save_folder = load_example_stim_data(
    #     data_folder='C:\\data\\ICMS92\\Behavior\\08-Sep-2023')
    num_ch = len(rec.get_channel_ids())
    ch_idx = list(range(num_ch))
    time_range = [400*30000, 410*30000]
    plot_traces(rec, all_stim_timestamps,
                time_range, ch_idx, trial_df, save_folder)
