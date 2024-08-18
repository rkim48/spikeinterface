import spikeinterface.preprocessing as sp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def preprocess_data(rec, stim_ts, time_range, plot_ids, plot_flag=True):

    palette = sns.color_palette("deep")
    flattened_stim = [ts for l in stim_ts for ts in l]
    offset_stim_ts = [[item + 0 for item in sublist] for sublist in stim_ts]

    # offset_stim_ts = [[item + 30 for item in sublist] for sublist in all_stim_timestamps]

    art_remove_pre = 0
    art_remove_post1 = 1  # 1.4
    art_remove_post2 = 1.5  # 1.5
    trend_remove_start = 1  # 1.4
    trend_remove_end = 10

    art_remove_pre = 0
    art_remove_post1 = 1.4  # 1.4
    art_remove_post2 = 1.5  # 1.5
    trend_remove_start = 1.4  # 1.4
    trend_remove_end = 10

    rec1 = sp.remove_artifacts(
        rec, flattened_stim,  ms_before=art_remove_pre, ms_after=art_remove_post1, mode='zeros')
    rec2 = sp.mean_artifact_subtract(
        rec1, list_triggers=offset_stim_ts, post_stim_window_ms=10, mode='median')
    # mean artifact subtract introduces artifact at start of window
    # if you don't mean artifact subtract, need to add interpolation between -0.5 to 1.5 ms
    rec3 = sp.trend_subtract(
        rec2, stim_ts, trend_remove_start, trend_remove_end, mode='poly', poly_order=3)
    rec4 = sp.common_reference(
        rec3, operator="median", reference="global")
    rec5 = sp.remove_artifacts(
        rec4, flattened_stim,  ms_before=art_remove_pre, ms_after=art_remove_post1, mode='cubic')
    rec6 = sp.bandpass_filter(rec5, freq_min=300, freq_max=5000)
    rec7 = sp.remove_artifacts(
        rec6, flattened_stim,  ms_before=art_remove_pre, ms_after=art_remove_post2, mode='zeros')
    rec8 = sp.whiten(rec2, dtype='float32')

    preprocessed_traces = rec7.get_traces(
        start_frame=time_range[0], end_frame=time_range[1], return_scaled=True, channel_ids=plot_ids)
    raw_traces = rec.get_traces(
        start_frame=time_range[0], end_frame=time_range[1], return_scaled=True, channel_ids=plot_ids)

    if plot_flag:
        plt.plot(raw_traces, color=palette[0], alpha=1, label='raw')
        plt.plot(preprocessed_traces, color=palette[1], alpha=1, label='raw')
        plot_stim_ts = [
            ts for ts in flattened_stim if time_range[0] <= ts <= time_range[1]]
        plot_stim_ts = np.array(plot_stim_ts) - time_range[0]
        plt.vlines(plot_stim_ts,
                   ymin=-5000, ymax=1000, color='k')
        plt.vlines(plot_stim_ts + 30 * 1.5,
                   ymin=-5000, ymax=1000, color='k')

        plt.hlines(-100, xmin=0, xmax=len(raw_traces))

        plt.text(4000, -200, '-100 uV', fontsize=10,
                 rotation=0, rotation_mode='anchor')

        plt.text(plot_stim_ts[0]+2, 200, 'Stim onset', fontsize=10,
                 rotation=270, rotation_mode='anchor')
        plt.text(plot_stim_ts[0]+30 * 1.5+2, 200, '1.5 ms', fontsize=10,
                 rotation=270, rotation_mode='anchor')
        plt.xlabel("Samples")

    return rec7
