import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from batch_process.util.curate_util import get_template_ch


def find_similar_groups(similarity_matrix, cell_types, threshold=0.9):
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
                        similarity_matrix[node, neighbor] > threshold
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


def get_template_ch_template(we, unit_ids):
    template_ch_dict = get_template_ch(we)
    templates = []
    for unit_id in unit_ids:
        num_wvfs = np.shape(we.get_waveforms(unit_id))[0]
        primary_ch_idx = template_ch_dict[unit_id]["primary_ch_idx_sparse"]
        template = we.get_template(unit_id)[:, primary_ch_idx]
        templates.append(template)
    return templates


def get_template_cosine_similarity_matrix(
    templates, max_shift=5, samples_to_use=np.arange(20, 42)
):
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

                similarity = cosine_similarity(
                    shifted_template_i.reshape(1, -1), template_j.reshape(1, -1)
                )[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity

            similarity_matrix[i, j] = max_similarity
            similarity_matrix[j, i] = max_similarity
    return similarity_matrix


def plot_template_cosine_similarity_matrix(similarity_matrix, ax=None):
    n_templates = similarity_matrix.shape[0]
    if ax is None:
        ax = plt.gca()

    # Plot the normalized similarity matrix with appropriate colormap range
    cax = ax.imshow(similarity_matrix, cmap="Blues", vmin=0.6, vmax=1)
    for i in range(n_templates):
        for j in range(n_templates):
            ax.text(
                j,
                i,
                f"{similarity_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
            )
    ax.set_title("Template cosine similarity")
