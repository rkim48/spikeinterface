import spikeinterface.full as si
import matplotlib.pyplot as plt
from batch_process.postprocessing.responses.session_responses import SessionResponses
from batch_process.postprocessing.responses.unit_response import UnitResponse
import dill as pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import batch_process.postprocessing.stim_response_util as stim_response_util
import batch_process.postprocessing.responses.pulse_locked_response_metrics as plrm

# %%
data_folder = "C:\\data\\ICMS92\\Behavior\\30-Aug-2023"
pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"
with open(pkl_path, "rb") as file:
    session_responses = pickle.load(file)

# %%
unit_id = 18
stim_channel = 11
stim_current = 5

unit_response = session_responses.get_unit_response(unit_id)
stim_response = unit_response.get_stim_response(stim_channel, stim_current)

spike_timestamps = stim_response.spike_timestamps
stim_timestamps = stim_response.stim_timestamps
timing_params = unit_response.timing_params
# %%

rel_stim_spike_times, stim_trial_indices, _ = stim_response_util.get_relative_spike_data(
    spike_timestamps, stim_timestamps, timing_params
)
stim_raster_array = stim_response_util.get_pulse_interval_raster_array(rel_stim_spike_times, stim_trial_indices)

baseline_interval_start_times = plrm.get_baseline_interval_start_times(stim_timestamps, timing_params)
rel_baseline_spike_times, baseline_trial_indices, _ = stim_response_util.get_relative_spike_data(
    spike_timestamps, baseline_interval_start_times, timing_params
)
baseline_raster_array = stim_response_util.get_pulse_interval_raster_array(
    rel_baseline_spike_times, baseline_trial_indices
)

baseline_bin_centers, baseline_prob_spike, _ = plrm.get_spike_prob_distribution(
    baseline_raster_array, timing_params, subsample_fraction=0.2
)
stim_bin_centers, stim_prob_spike, _ = plrm.get_spike_prob_distribution(
    stim_raster_array, timing_params, subsample_fraction=0.2
)

bin_centers, baseline_mean_prob_spike, baseline_std_prob_spike, baseline_mean_latency, baseline_std_latency = (
    plrm.get_bootstrapped_metrics(baseline_raster_array, timing_params, subsample_fraction=0.2)
)
bin_centers, stim_mean_prob_spike, stim_std_prob_spike, stim_mean_latency, stim_std_latency = (
    plrm.get_bootstrapped_metrics(stim_raster_array, timing_params, subsample_fraction=0.2)
)

plt.subplot(2, 2, 1)
plt.eventplot(
    baseline_raster_array,
    orientation="horizontal",
    linewidths=2,
    linelengths=5,
)
plt.xlim([0, 10])

plt.subplot(2, 2, 2)
plt.eventplot(
    stim_raster_array,
    orientation="horizontal",
    linewidths=2,
    linelengths=5,
)
plt.xlim([0, 10])


plt.subplot(2, 2, 3)

plt.plot(baseline_bin_centers, baseline_mean_prob_spike, color="k")
plt.fill_between(
    baseline_bin_centers,
    baseline_mean_prob_spike - baseline_std_prob_spike,
    baseline_mean_prob_spike + baseline_std_prob_spike,
    color="k",
    alpha=0.3,
)

plt.xlim([0, 10])
plt.ylim([0, 0.2])


plt.subplot(2, 2, 4)

plt.plot(stim_bin_centers, stim_mean_prob_spike, color="C0")
plt.fill_between(
    stim_bin_centers,
    stim_mean_prob_spike - stim_std_prob_spike,
    stim_mean_prob_spike + stim_std_prob_spike,
    color="C0",
    alpha=0.3,
)
plt.xlim([0, 10])
plt.ylim([0, 0.2])

plt.tight_layout()

# %% PLI measurement

stim_pli = plrm.calculate_pulse_locked_index(bin_centers, stim_mean_prob_spike, timing_params)
baseline_pli = plrm.calculate_pulse_locked_index(bin_centers, baseline_mean_prob_spike, timing_params)

# is pli_stim > 99% of shuffled stim trials?
null_dist = plrm.get_null_distribution(stim_raster_array, timing_params, plot_flag=False)

# plt.axvline(pli_stim)

if plrm.is_pulse_locked(stim_pli, null_dist):
    print("Phase locked!")


# %%
conditions = ["Baseline", "Stimulation"]
mean_latencies = [5.43225, 3.947]
std_latencies = [2.0777926839557406, 0.24533854161138235]

# Create the bar plot
plt.figure(figsize=(8, 6))
bars = plt.bar(conditions, mean_latencies, yerr=std_latencies, capsize=10, color=["blue", "red"], alpha=0.7)

# Customize the plot
plt.xlabel("Condition")
plt.ylabel("Mean Latency (ms)")
plt.title("Mean Latency with Standard Deviation")
plt.ylim(0, max(mean_latencies) + max(std_latencies) + 1)  # Adjust y-axis limits for better visualization
plt.grid(True, linestyle="--", alpha=0.6)

# Show the plot
plt.show()
