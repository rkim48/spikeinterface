from pathlib import Path
import time
import matplotlib.pyplot as plt

from util.file_util import file_dialog
from util.load_data import load_data
from spikeinterface import full as si
import spikeinterface.preprocessing as sp
from preprocessing.preprocessing_pipelines import new_pipeline3
from batch_process.util.plotting import plot_units_in_batches
from batch_process.util.file_util import create_folder
from batch_process.util.curate_util import *
from batch_process.util.misc import *


# %% Load example data

data_folder = 'C:\\data\\ICMS93\\behavior\\30-Aug-2023'
start_time = time.time()
dataloader = load_data(
    data_folder,
    make_folder=True,
    save_folder_name="batch_sort",
    first_N_files=1,
)
rec = dataloader.recording
fs = dataloader.fs
all_stim_timestamps = dataloader.all_stim_timestamps
ch_ids = rec.channel_ids

# %%
# from spikeinterface.preprocessing.mean_artifact_subtract import MeanArtifactSubtractedRecording
flattened_stim = [ts for l in all_stim_timestamps for ts in l]
offset_stim_ts = [[int(item + 0) for item in sublist]
                  for sublist in all_stim_timestamps]
art_remove_pre = 0.5
art_remove_post1 = 1.4  # 1.4
art_remove_post2 = 1.5  # 1.5
trend_remove_post_pulse_start = 1.4  # 1.4
trend_remove_post_pulse_end = 10

# %%
rec1 = sp.remove_artifacts(
    rec, flattened_stim,  ms_before=art_remove_pre, ms_after=art_remove_post1, mode='zeros')
rec2 = sp.mean_artifact_subtract(
    rec1, list_triggers=offset_stim_ts, post_stim_window_ms=10, mode='median')
rec3 = sp.trend_subtract(
    rec2, all_stim_timestamps, trend_remove_post_pulse_start, trend_remove_post_pulse_end, mode='poly', poly_order=3)
rec4 = sp.common_reference(rec3, operator="median", reference="global")
rec5 = sp.remove_artifacts(
    rec4, flattened_stim, ms_before=art_remove_pre, ms_after=art_remove_post1, mode='cubic')
rec6 = sp.bandpass_filter(rec5, freq_min=300, freq_max=5000)
rec7 = sp.remove_artifacts(
    rec6, flattened_stim, ms_before=art_remove_pre, ms_after=art_remove_post2, mode='zeros')
rec8 = sp.whiten(si.scale(rec7, dtype='float'), dtype='float32')

# %%
trace1 = rec1.get_traces(return_scaled=True)
trace2 = rec2.get_traces(return_scaled=True)
trace3 = rec3.get_traces(return_scaled=True)

trace4 = rec4.get_traces(return_scaled=True)
trace5 = rec5.get_traces(return_scaled=True)
trace6 = rec6.get_traces(return_scaled=True)

trace7 = rec7.get_traces(return_scaled=True)
trace8 = rec8.get_traces(return_scaled=False)

# %% Plot
ch_to_plot = np.arange(15, 20)
start_frame = int(fs * 59)
end_frame = int(fs * 62)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.plot(trace1[start_frame:end_frame, ch_to_plot], color=colors[0])
plt.plot(trace2[start_frame:end_frame, ch_to_plot], color=colors[1])
plt.plot(trace3[start_frame:end_frame, ch_to_plot], color=colors[2])
plt.plot(trace4[start_frame:end_frame, ch_to_plot], color=colors[3])
plt.plot(trace5[start_frame:end_frame, ch_to_plot], color=colors[4])
plt.plot(trace6[start_frame:end_frame, ch_to_plot], color=colors[5])
plt.plot(trace7[start_frame:end_frame, ch_to_plot], color=colors[6])
plt.plot(trace8[start_frame:end_frame, ch_to_plot], color=colors[7])

plt.plot([], [], color=colors[0], label='Trace 1')
plt.plot([], [], color=colors[1], label='Trace 2')
plt.plot([], [], color=colors[2], label='Trace 3')
plt.plot([], [], color=colors[3], label='Trace 4')
plt.plot([], [], color=colors[4], label='Trace 5')
plt.plot([], [], color=colors[5], label='Trace 6')
plt.plot([], [], color=colors[6], label='Trace 7')
plt.plot([], [], color=colors[7], label='Trace 8')

plt.legend(loc='upper right')
plt.show()
