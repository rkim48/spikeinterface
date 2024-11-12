import h5py
import matplotlib.pyplot as plt
import numpy as np


def plot_template(template, channel_locations):
    fs = 30000  # sampling rate
    t = np.arange(90) / fs * 1000  # ms
    # plot depth-ordered template
    for idx, wvf in enumerate(template.T):
        plt.plot(t, wvf + channel_locations[idx, 1], "k")


def plot_acg(time_bins, acg):
    time_bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
    plt.bar(time_bin_centers, acg, width=np.diff(time_bins), align="center")


hdf5_file_path = "ICMS92.h5"

f = h5py.File(hdf5_file_path, "r")
list(f.keys())

session1 = f["session_1"]

print(f"Session attributes: {list(session1.attrs.keys())}")
print(f"Session date: {session1.attrs["session_date"]}")

session1_keys = list(session1.keys())
print(f"Session keys: {session1_keys}")

channel_locations = session1["channel_locations"]
time_bins = session1["acg_time_bins"]

print(f"channel_locations[0]: {channel_locations[0]}")
print(f"channel_locations[1]: {channel_locations[1]}")

units_group = session1["units"]
unit_ids = list(units_group.keys())

# accepted unit example
for unit_id in unit_ids[40:41]:
    unit_group = units_group[unit_id]

    is_accepted = unit_group.attrs["is_accepted"]

    # Access datasets like template, timestamps, and amplitudes
    template = unit_group["template"][:]
    timestamps = unit_group["timestamps"][:]
    amplitudes = unit_group["amplitudes"][:]
    acg = unit_group["acg"][:]

    # Print or use the loaded data
    print(f"{unit_id} accepted: {is_accepted}")
    print(f"Template shape: {template.shape}")
    print(f"Timestamps: {timestamps[:5]}")  # Print first 5 timestamps
    print(f"Amplitudes: {amplitudes[:5]}")  # Print first 5 amplitudes

    plt.subplot(1, 2, 1)
    plot_template(template, channel_locations)
    plt.subplot(1, 2, 2)
    plot_acg(time_bins, acg)
    plt.suptitle(f"{unit_id}: accepted")

# rejected unit example
for unit_id in unit_ids[14:15]:
    unit_group = units_group[unit_id]

    is_accepted = unit_group.attrs["is_accepted"]

    # Access datasets like template, timestamps, and amplitudes
    template = unit_group["template"][:]
    timestamps = unit_group["timestamps"][:]
    amplitudes = unit_group["amplitudes"][:]
    acg = unit_group["acg"][:]

    # Print or use the loaded data
    print(f"{unit_id} accepted: {is_accepted}")
    print(f"Template shape: {template.shape}")
    print(f"Timestamps: {timestamps[:5]}")
    print(f"Amplitudes: {amplitudes[:5]}")

    plt.figure()
    plt.subplot(1, 2, 1)
    plot_template(template, channel_locations)
    plt.subplot(1, 2, 2)
    plot_acg(time_bins, acg)
    plt.suptitle(f"{unit_id}: rejected")
