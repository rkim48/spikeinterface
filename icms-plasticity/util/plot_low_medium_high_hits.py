

for block in block_dict.items():
    ch_hit_dict = block[1]
    lines = list(zip(*ch_hit_dict.values()))
    for i, line in enumerate(lines):
        plt.plot(ch_hit_dict.keys(), line)

# %%
# Define labels for low, medium, and high currents
current_labels = ['Low Current', 'Medium Current', 'High Current']

# Plot low, medium, high current hits as three separate curves, take mean and std across channels for each block
for i, (date, block_dict) in enumerate(hits_dict.items()):
    offset = i
    for i in range(3):
        first_values = {block: [channel_data[i] for channel_data in channels.values(
        )] for block, channels in block_dict.items()}

        # Calculate mean and standard deviation across channels for each block
        means = [np.mean(values) for values in first_values.values()]
        stds = [np.std(values) for values in first_values.values()]

        # Plot the mean with standard deviation error bars
        plt.errorbar(first_values.keys() + offset, means,
                     yerr=stds, fmt='-o', label=current_labels[i])

# Add labels and legend
# plt.xticks([0,1,2,3], [1,2,3,4])
plt.xticklabels([1, 2, 3, 4])
plt.xlabel('Block')
plt.ylabel('Hits')
plt.legend()
plt.show()


# %%
def get_hits_per_ch(df):
    assert len(df) == 400
    trials_per_block = 100  # Each block has 100 trials
    hits_per_ch = {}
    for block in range(4):
        start_idx = trials_per_block * block
        end_idx = start_idx + trials_per_block
        sub_df = df.iloc[start_idx:end_idx]
        unique_channels = sorted(
            sub_df[sub_df['channel'] != 0]['channel'].unique())

        channel_hits = {}
        for channel in unique_channels:
            channel_data = sub_df[sub_df['channel'] == channel]
            channel_currents = np.sort(channel_data['current'].unique())
            current_hits = []
            for current in channel_currents:
                current_data = channel_data[channel_data['current'] == current]
                current_hits.append(current_data['response'].sum())
            channel_hits[channel] = current_hits

        hits_per_ch[block] = channel_hits
    return hits_per_ch


all_session_thresholds = read_thresholds([data_folder])

get_hits_dict([data_folder])
df = get_dataframe(data_folder)
