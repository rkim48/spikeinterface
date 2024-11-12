import spikeinterface.full as si
from batch_process.postprocessing.responses.session_responses import SessionResponses
from batch_process.postprocessing.responses.unit_response import UnitResponse
import dill as pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# %%
# rec = si.generate_recording()
# sorting = si.generate_sorting()

# test_analyzer_path = "batch_process/postprocessing/responses/tests/test_analyzer.zarr"

# analyzer = si.create_sorting_analyzer(
#     sorting=sorting, recording=rec, format="zarr", folder=test_analyzer_path, overwrite=True
# )

# analyzer.compute("random_spikes")
# analyzer.compute("waveforms")
# analyzer.compute("templates")

analyzer = si.load_sorting_analyzer("C:/data/ICMS93/behavior/30-Aug-2023/batch_sort/stage3/stage3_analyzer.zarr")

timing_params = {
    "pulse_win_ms": 10,
    "post_stim_blank_ms": 1.5,
    "pre_stim_blank_ms": 0,
    "window_size_ms": 1400,
    "stim_duration_ms": 700,
    "post_stim_duration_ms": 700,
    "windows_per_sample": 50,
    "bootstrap_samples": 500,
    "fs": 30000,
    "bin_width": 0.5,
    "single_pulse_pre": 0,
    "single_pulse_post": 300,
}

# %%
sr1 = SessionResponses(
    session_path="test2", sorting_analyzer=analyzer, all_stim_timestamps=[1, 2, 3, 4, 5], timing_params=timing_params
)

# initialize unit responses
# initializing unit response will automatically update sr attributes
unit0_response = UnitResponse(unit_id=0, session=sr1)
unit1_response = UnitResponse(unit_id=1, session=sr1)


# %% Test with real session responses pkl

data_folder = "C:\\data\\ICMS92\\Behavior\\30-Aug-2023"
pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"
with open(pkl_path, "rb") as file:
    session_responses = pickle.load(file)

# %%
for idx, unit_id in enumerate(session_responses.unit_ids):
    unit_response = session_responses.get_unit_response(unit_id)
    print(unit_response)
    for stim_response in unit_response.stim_responses:
        stim_channel = stim_response[0]
        stim_current = stim_response[1]
        print(f"\t{unit_response.get_stim_response(stim_channel=stim_channel, stim_current=stim_current)}")


# %% Get specific stim condition response for unit
unit_id = 18
stim_channel = 11
stim_current = 5

unit_response = session_responses.get_unit_response(unit_id)
stim_response = unit_response.get_stim_response(stim_channel, stim_current)

pulse_response = stim_response.pulse_response
train_response = stim_response.train_response

#  Plot pulse response raster array
linewidth_factor = 20
line_offset = 0

fig, axes = plt.subplots(nrows=1, ncols=2)

tparams = pulse_response.timing_params
raster_array = pulse_response.raster_array
good_linelength = np.ceil(len(raster_array) / linewidth_factor)
lineoffsets = line_offset + np.arange(len(raster_array))
axes[0].eventplot(raster_array, linelengths=good_linelength, linewidths=1, lineoffsets=lineoffsets)

axes[0].axvspan(0, tparams["post_stim_blank_ms"], color="lightgray", alpha=0.5)
axes[0].axvspan(
    tparams["pulse_win_ms"] - tparams["pre_stim_blank_ms"], tparams["pulse_win_ms"], color="lightgray", alpha=0.5
)

axes[0].set_xlim([0, tparams["pulse_win_ms"]])
axes[0].set_ylim([0, len(raster_array)])
axes[0].set_ylabel("Pulses")


# Plot train response raster array
raster_array = train_response.raster_array
good_linelength = np.ceil(len(raster_array) / linewidth_factor)
lineoffsets = line_offset + np.arange(len(raster_array))
axes[1].eventplot(
    raster_array, orientation="horizontal", linelengths=good_linelength, linewidths=1, lineoffsets=lineoffsets
)

axes[1].set_xlim([-700, 1400])
axes[1].axvline(x=0, color="gray", linestyle="--", linewidth=1)
axes[1].axvline(x=700, color="gray", linestyle="--", linewidth=1)

# Update the y-axis limits to accommodate the new raster
axes[1].set_ylim([0, line_offset + len(raster_array)])
axes[1].set_ylabel("Trials")

plt.suptitle(f"Ch.{stim_channel} at {stim_current} uA")
plt.tight_layout()

# %% Test pulse locked metrics

baseline_metrics = pulse_response.baseline_metrics
stim_metrics = pulse_response.stim_metrics

plt.plot(baseline_metrics["bin_centers"], baseline_metrics["mean_prob_spike"], color="k")
plt.fill_between(
    baseline_metrics["bin_centers"],
    baseline_metrics["mean_prob_spike"] - baseline_metrics["std_prob_spike"],
    baseline_metrics["mean_prob_spike"] + baseline_metrics["std_prob_spike"],
    color="k",
    alpha=0.3,
)

plt.plot(stim_metrics["bin_centers"], stim_metrics["mean_prob_spike"], color="C0")
plt.fill_between(
    stim_metrics["bin_centers"],
    stim_metrics["mean_prob_spike"] - stim_metrics["std_prob_spike"],
    stim_metrics["mean_prob_spike"] + stim_metrics["std_prob_spike"],
    color="C0",
    alpha=0.3,
)

plt.xlim([0, 10])
plt.ylim([0, 0.2])
