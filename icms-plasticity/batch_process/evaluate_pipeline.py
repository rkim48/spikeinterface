import os
import batch_process.util.template_util as template_util
import spikeinterface.full as si

#%%
data_folder = "C:\\data\\ICMS92\Behavior\\01-Sep-2023"
merge_path = os.path.join(
    data_folder, "batch_sort\\merge\\hmerge_analyzer_curated.zarr")

pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"
with open(pkl_path, "rb") as file:
    session_responses = pickle.load(file)

analyzer = si.load_sorting_analyzer(merge_path)

#%%
dataloader = load_data(
    data_folder, make_folder=True, save_folder_name="batch_sort", first_N_files=4, server_mount_drive="S:"
)
rec_raw = dataloader.recording

raw_analyzer = si.create_sorting_analyzer(
    sorting=analyzer.sorting,
    recording=rec_raw,
    format="memory",
    sparse=False,
)

raw_analyzer.compute('random_spikes', method='all')
raw_analyzer.compute('waveforms')
raw_analyzer.compute('templates', operators=['average', 'median'])
raw_analyzer.compute('unit_locations')

# %%
template_util.save_sparse_analyzer(analyzer)
template_dict = template_util.get_template_ch(analyzer)
raw_template_ext = raw_analyzer.get_extension('templates')
templates = raw_template_ext.get_data(operator='average')

unit_ids = raw_analyzer.unit_ids
for idx in range(0, 50):
    plt.figure()
    unit_id = unit_ids[idx]

    # plt.plot(templates[idx, :, :])

    ch_template = np.mean(templates[idx, :, :], axis=1)

    primary_ch_idx_sparse = template_dict[unit_id]['primary_ch_idx_sparse']
    primary_ch_idx_dense = template_dict[unit_id]['primary_ch_idx_dense']

    primary_ch_template = templates[idx, :, primary_ch_idx_dense]

    plt.plot(templates[idx, :, :], 'k', linewidth=1)

    plt.plot(primary_ch_template, 'r', linewidth=2)
# %%


# %% Look at spikes assigned to units for trace

fig = plt.figure(figsize=(6, 2.5))

# Define GridSpec with 2 rows, 2 columns
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[3, 1])
axs = {}
# Define subplots within the grid
axs["train_raster"] = fig.add_subplot(
    gs[0, 0:2])  # Raster plot covering top row
axs["a"] = fig.add_subplot(gs[1, 0:2])  # Trace plot covering bottom row

for id in session_responses.unit_ids:

    if id == 0:
        ur = session_responses.get_unit_response(id)

        # Draw vertical lines at intervals of 10 ms for raster plot
        for i in np.arange(0, 700, 10):
            axs["train_raster"].axvline(i, color='r', linewidth=0.5)

        train_line_offset = 0
        for idx, current in enumerate(np.arange(3, 6)):
            scr = ur.get_stim_response(9, current)
            tr = scr.train_response

            rpu.plot_train_raster_with_scatter(
                axs["train_raster"], tr, 'k', train_line_offset)

            if idx > -1:
                axs["train_raster"].axhline(
                    y=train_line_offset - 0.5, color='k', linewidth=0.5)

            midpoint = train_line_offset + len(tr.raster_array) / 2

            # Add current label text
            axs["train_raster"].text(
                x=-120,  # Outside the axis on the left
                y=midpoint,
                s=f"{current} µA",  # Label text
                va='center',
                ha='right',
                fontsize=10,
            )
            train_line_offset += len(tr.raster_array)

        # Configure axis labels and limits for train_raster subplot
        axs["train_raster"].set_ylabel('')
        axs["train_raster"].set_yticks([])
        axs["train_raster"].set_ylim([0 - 0.5, train_line_offset + 0.5])
        axs["train_raster"].set_xlim([-100, 800])
        axs["train_raster"].set_xlabel("Time (ms)")

rec = session_responses.sorting_analyzer.recording
ur = session_responses.get_unit_response(0)  # Assuming 0 is a valid unit ID
scr = ur.get_stim_response(9, 3)
stim_ts = scr.stim_timestamps
trial = 3  # Define the trial number
t1 = trial * 70
t2 = trial * 70 + 10
traces = rec.get_traces(
    start_frame=stim_ts[t1], end_frame=stim_ts[t2], return_scaled=True)

# %%
ch_ids = rec.get_channel_ids()
ch_locs = rec.get_channel_locations()
sorted_indices = np.argsort(ch_locs[:, 1])
sorted_ch_ids = ch_ids[sorted_indices]

# Plot the trace once for selected channels in the trace plot subplot
fig, ax = plt.subplots(1, 1)
for i, ch_id in enumerate(sorted_indices[0:32]):
    ax.vlines(x=(stim_ts[t1:t2] - stim_ts[t1]) /
              30, ymin=-200, ymax=200, colors='r')
    ax.plot((np.arange(traces.shape[0]) / 30),
            traces[:, ch_id] + i * 500, 'k')

# Loop through all unit IDs to plot markers without replotting the trace
for id in session_responses.unit_ids:
    ur = session_responses.get_unit_response(id)

    unit_ch_index = template_util.get_dense_primary_ch_index(analyzer, id)
    ordered_index = np.where(sorted_ch_ids == ch_ids[unit_ch_index])[0][0]
    marker_y = (ordered_index * 500) - 300

    scr = ur.get_stim_response(9, 3)
    stim_spikes = scr.stim_spikes[(scr.stim_spikes >= stim_ts[t1]) & (
        scr.stim_spikes < stim_ts[t2])]

    # Plot spike markers as a scatter plot
    ax.scatter(
        x=(stim_spikes - stim_ts[t1]) / 30,
        y=np.full_like(stim_spikes, marker_y),
        color=f'C3',
        marker='^',
        s=20,
        # Only add label for first unit to avoid duplicates
        label='Spikes' if id == 0 else ""
    )

# Set labels and layout adjustments
ax.set_xlabel("Time (ms)")
plt.tight_layout()  # Adjust layout to prevent overlapping

# %% Read from stage 2
import pickle
from pathlib import Path

def load_stage_data(data_folder, stage_name):
    # read curated if exists
    analyzer_path = os.path.join(
        data_folder, f"batch_sort\\{stage_name}\\original\\{stage_name}_analyzer_curated.zarr")
    if not os.path.exists(analyzer_path):
        analyzer_path = os.path.join(
            data_folder, f"batch_sort\\{stage_name}\\sd5\\{stage_name}_analyzer.zarr")

    analyzer = si.load_sorting_analyzer(analyzer_path)
    rec = analyzer.recording

    return analyzer, rec

def load_merge_stage_data(data_folder):
    analyzer_path = os.path.join(
        data_folder, "batch_sort\\merge\\hmerge_analyzer_curated.zarr")
    pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"
    with open(pkl_path, "rb") as file:
        session_responses = pickle.load(file)
    analyzer = si.load_sorting_analyzer(analyzer_path)
    rec = session_responses.sorting_analyzer.recording

    return analyzer, rec

def plot_traces_with_unit_spikes(rec, analyzer, session_responses):
    ur = session_responses.get_unit_response(0)  # Assuming 0 is a valid unit ID
    scr = ur.get_stim_response(9, 3)
    stim_ts = scr.stim_timestamps
    trial = 3  # Define the trial number
    t1 = trial * 70
    t2 = trial * 70 + 10
    traces = rec.get_traces(
        start_frame=stim_ts[t1], end_frame=stim_ts[t2], return_scaled=True)

    ch_ids = rec.get_channel_ids()
    ch_locs = rec.get_channel_locations()
    sorted_indices = np.argsort(ch_locs[:, 1])
    sorted_ch_ids = ch_ids[sorted_indices]

    # Plot the trace once for selected channels in the trace plot subplot
    fig, ax = plt.subplots(1, 1)
    for i, ch_id in enumerate(sorted_indices[0:32]):
        ax.vlines(x=(stim_ts[t1:t2] - stim_ts[t1]) /
                  30, ymin=-200, ymax=200, colors='r')
        ax.plot((np.arange(traces.shape[0]) / 30),
                traces[:, ch_id] + i * 500, 'k')

    # Loop through all unit IDs to plot markers without replotting the trace
    extremum_ch_dict = si.get_template_extremum_channel(analyzer)
    unit_ids = analyzer.unit_ids
    for unit_id in unit_ids:
        spike_train = analyzer.sorting.get_unit_spike_train(unit_id=unit_id)
        extremum_ch_id = extremum_ch_dict[unit_id]
        ordered_index = np.where(sorted_ch_ids == extremum_ch_id)[0][0]
        marker_y = (ordered_index * 500) - 300

        scr = ur.get_stim_response(9, 3)
        stim_spikes = spike_train[(spike_train >= stim_ts[t1]) & (
            spike_train < stim_ts[t2])]

        # Plot spike markers as a scatter plot
        ax.scatter(
            x=(stim_spikes - stim_ts[t1]) / 30,
            y=np.full_like(stim_spikes, marker_y),
            color=f'C3',
            marker='^',
            s=20,
            # Only add label for first unit to avoid duplicates
            label='Spikes' if id == 0 else ""
        )

    # Set labels and layout adjustments
    ax.set_xlabel("Time (ms)")
    plt.tight_layout()  # Adjust layout to prevent overlapping


#%%
data_folder = "C:\\data\\ICMS92\Behavior\\01-Sep-2023"
pkl_path = Path(data_folder) / "batch_sort/session_responses.pkl"
with open(pkl_path, "rb") as file:
    session_responses = pickle.load(file)


stage1_analyzer, stage1_rec = load_stage_data(data_folder, stage_name="stage1")
stage2_analyzer, stage2_rec = load_stage_data(data_folder, stage_name="stage2")

merge_analyzer, merge_rec = load_merge_stage_data(data_folder)

# Look at spikes at same segment
#%%
plot_traces_with_unit_spikes(stage1_rec, stage1_analyzer, session_responses)

plt.title("Stage 1")
plt.tight_layout()

plot_traces_with_unit_spikes(stage2_rec, stage2_analyzer, session_responses)
plt.title("Stage 2")
plt.tight_layout()

plot_traces_with_unit_spikes(merge_rec, merge_analyzer, session_responses)
plt.title("Stage 3")
plt.tight_layout()
