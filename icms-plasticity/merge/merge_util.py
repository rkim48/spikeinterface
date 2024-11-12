import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import spikeinterface.full as si

import batch_process.util.template_util as template_util

# %%


def find_similar_groups(
    similarity_matrix, distance_matrix, cell_types, distance_um_threshold=25, similarity_threshold=0.9
):
    n_templates = similarity_matrix.shape[0]
    groups = []
    visited = np.zeros(n_templates, dtype=bool)

    def dfs(i, group):
        stack = [i]
        while stack:
            node = stack.pop()
            if not visited[node]:
                visited[node] = True
                group.append(node)
                for neighbor in range(n_templates):
                    if (
                        similarity_matrix[node, neighbor] > similarity_threshold
                        and distance_matrix[node, neighbor] < distance_um_threshold
                        and not visited[neighbor]
                        and cell_types[node] == cell_types[neighbor]
                    ):
                        stack.append(neighbor)

    for i in range(n_templates):
        if not visited[i]:
            group = []
            dfs(i, group)
            groups.append(group)

    return groups


def shift_waveform(waveform, shift):
    original_length = len(waveform)
    if shift > 0:
        shifted = np.pad(waveform, (shift, 0), "constant")[:-shift]
    elif shift < 0:
        shifted = np.pad(waveform, (0, -shift), "constant")[-shift:]
    else:
        shifted = waveform
    # Ensure the length is the same as the original
    return shifted[:original_length]


# def get_unit_primary_ch_template_dict(analyzer, unit_ids, template_ch_dict):
#     templates_ext = analyzer.get_extension("templates")

#     unit_primary_ch_wvfs_dict = {}
#     for unit_id in unit_ids:
#         primary_ch_idx = template_ch_dict[unit_id]["primary_ch_idx_dense"]
#         primary_ch_wvf = templates_ext.get_unit_template(unit_id)[:, primary_ch_idx]
#         unit_primary_ch_wvfs_dict[unit_id] = primary_ch_wvf
#     return unit_primary_ch_wvfs_dict


def get_template_cosine_similarity_matrix(analyzer, unit_ids, max_shift=5, samples_to_use=np.arange(20, 42)):

    unit_primary_ch_template_dict = template_util.get_unit_primary_ch_templates(analyzer)
    templates = [unit_primary_ch_template_dict[unit_id] for unit_id in unit_ids]

    n_templates = len(templates)
    similarity_matrix = np.zeros((n_templates, n_templates))

    for i in range(n_templates):
        for j in range(i, n_templates):
            max_similarity = -1
            for shift in range(-max_shift, max_shift + 1):
                shifted_template_i = shift_waveform(templates[i], shift)
                # Select the specified range of samples
                shifted_template_i = shifted_template_i[samples_to_use]
                template_j = templates[j][samples_to_use]

                similarity = cosine_similarity(shifted_template_i.reshape(1, -1), template_j.reshape(1, -1))[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity

            similarity_matrix[i, j] = max_similarity
            similarity_matrix[j, i] = max_similarity
    return similarity_matrix


def get_unit_distance_matrix(analyzer, unit_ids):

    n_units = len(unit_ids)
    distance_matrix = np.zeros((n_units, n_units))
    ul_ext = analyzer.get_extension("unit_locations")
    ul_ext.get_data(outputs="by_unit")
    unit_locations = ul_ext.get_data(outputs="by_unit")

    for i in range(n_units):
        unit_loc_i = unit_locations[unit_ids[i]]
        for j in range(i, n_units):
            unit_loc_j = unit_locations[unit_ids[j]]
            distance = np.linalg.norm(unit_loc_i - unit_loc_j)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix


def plot_template_cosine_similarity_matrix(similarity_matrix, merge_candidate_ids, ax=None):
    n_templates = similarity_matrix.shape[0]
    if ax is None:
        ax = plt.gca()

    # Use Seaborn heatmap to plot the similarity matrix
    sns.heatmap(
        similarity_matrix,
        annot=True,  # Annotate cells with the values
        fmt=".2f",  # Format of the annotations
        cmap="RdBu",  # Use the "crest" colormap
        vmin=0.6,
        vmax=1,  # Set the color range
        linewidths=0.5,  # Add grid lines with a specified width
        linecolor="black",  # Set grid line color
        cbar=False,  # Include a color bar
        xticklabels=merge_candidate_ids,  # Set x-axis labels
        yticklabels=merge_candidate_ids,  # Set y-axis labels
        ax=ax,  # Specify the axes to plot on
    )

    # Set the title of the plot
    ax.set_title("Template Cosine Similarity")


def plot_unit_primary_ch_waveforms(analyzer, unit_ids, ax=None):

    unit_primary_templates_dict = template_util.get_unit_primary_ch_templates(analyzer)
    wvfs_per_unit = analyzer.sorting.count_num_spikes_per_unit()

    for unit_id in unit_ids:
        num_wvfs = wvfs_per_unit[unit_id]
        wvf = unit_primary_templates_dict[unit_id]
        label = f"{unit_id}: ({num_wvfs} waveforms)"
        if ax is not None:
            ax.plot(wvf, label=label)
        else:
            plt.plot(wvf, label=label)
    if ax is not None:
        ax.legend()
        ax.set_title("Primary channel waveforms")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (uV)")
    else:
        plt.legend()
