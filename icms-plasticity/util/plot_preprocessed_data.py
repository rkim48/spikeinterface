import matplotlib.pyplot as plt
import numpy as np


def preprocess_and_plot(rec_preprocessed, all_stim_timestamps, time_range, plot_flag):

    traces = rec_preprocessed.get_traces(
        start_frame=time_range[0], end_frame=time_range[1], return_scaled=False)

    # Plot the results
    if plot_flag:
        plt.plot(traces[:, 0:20], 'k', alpha=0.2)
        flattened_stim = [ts for l in all_stim_timestamps for ts in l]
        plot_stim_ts = [
            ts for ts in flattened_stim if time_range[0] <= ts <= time_range[1]]
        plot_stim_ts = np.array(plot_stim_ts) - time_range[0]
        plt.vlines(plot_stim_ts,
                   ymin=-5, ymax=5, color='r')

    return traces


if __name__ == "__main__":
    fs = 30000
    time_range = np.array([500, 510]) * int(fs)
    traces = preprocess_and_plot(
        rec_preprocessed, all_stim_timestamps, time_range, plot_flag=True)
