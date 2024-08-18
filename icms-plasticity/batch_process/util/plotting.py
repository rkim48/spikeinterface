from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from spikeinterface import full as si

from batch_process.util.ppt_image_inserter import PPTImageInserter


# %%


def plot_units_in_batches(analyzer, save_dir, ppt_name, unit_ids=None, unit_colors=None, batch_size=5):

    analyzer.sparsity = si.compute_sparsity(
        analyzer, method="radius", radius_um=60)
    analyzer.compute("template_similarity")

    ppt_inserter = PPTImageInserter(
        grid_dims=(2, 2), spacing=(0.05, 0.05), title_font_size=16
    )

    if unit_ids is None:
        num_units = len(analyzer.unit_ids)
        unit_ids = analyzer.unit_ids
    else:
        num_units = len(unit_ids)

    if unit_colors is None:
        unit_colors = {unit_id: "k" for unit_id in unit_ids}

    for start in range(0, num_units, batch_size):
        end = min(start + batch_size, num_units)
        batch_unit_ids = unit_ids[start:end]

        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(2, 1, height_ratios=[1, 1])

        # First subplot grid for unit templates
        gs1 = GridSpecFromSubplotSpec(
            1, len(batch_unit_ids), subplot_spec=gs[0])
        template_axes = [fig.add_subplot(gs1[i])
                         for i in range(len(batch_unit_ids))]

        # Plot unit templates on the new set of axes
        for ax, unit_id in zip(template_axes, batch_unit_ids):
            all_wvfs = analyzer.get_extension(extension_name="waveforms")
            unit_wvfs = all_wvfs.get_waveforms_one_unit(unit_id=unit_id)
            num_spikes = len(unit_wvfs)
            si.plot_unit_templates(
                analyzer,
                sparsity=analyzer.sparsity,
                unit_ids=[unit_id],
                same_axis=True,
                unit_colors=unit_colors,
                x_offset_units=True,
                templates_percentile_shading=None,
                plot_legend=False,
                ax=ax,
            )
            ax.xaxis.set_visible(False)
            ax.set_title(f"Unit {unit_id}\n# spikes: {num_spikes}")

        # Second subplot grid for autocorrelograms
        gs2 = GridSpecFromSubplotSpec(
            1, len(batch_unit_ids), subplot_spec=gs[1])
        autocorr_axes = [fig.add_subplot(gs2[i])
                         for i in range(len(batch_unit_ids))]

        # Plot autocorrelograms on the new set of axes
        si.plot_autocorrelograms(
            analyzer,
            unit_ids=batch_unit_ids,
            axes=autocorr_axes,
            unit_colors=unit_colors,
        )

        # Hide x-axis and y-axis labels for the autocorrelograms
        for i, ax in enumerate(autocorr_axes):
            # ax.yaxis.set_visible(False)
            if i > 0:
                ax.xaxis.set_visible(False)
            ax.title.set_visible(False)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.2, wspace=0.4)

        img_path = save_dir / "img.png"
        plt.savefig(str(img_path))
        ppt_inserter.add_image(str(img_path))
        plt.close()

    ppt_inserter.save(save_dir / f"{ppt_name}.pptx")


def plot_primary_ch_template(we, unit_ids, plot=True, ax=None):
    from batch_process.util.curate_util import get_template_ch

    template_ch_dict = get_template_ch(we)
    templates = []
    for unit_id in unit_ids:
        num_wvfs = np.shape(we.get_waveforms(unit_id))[0]
        primary_ch_idx = template_ch_dict[unit_id]["primary_ch_idx_sparse"]
        template = we.get_template(unit_id)[:, primary_ch_idx]
        templates.append(template)
        label = f"Unit {unit_id} ({num_wvfs} waveforms)"
        if plot:
            if ax is not None:
                ax.plot(template, label=label)
            else:
                plt.plot(template, label=label)
    if plot:
        if ax is not None:
            ax.legend()
        else:
            plt.legend()
    return np.array(templates)


def adjust_font_sizes(ax, fontsize=10):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fontsize)
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)
    ax.title.set_size(fontsize)
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontsize(fontsize)


def add_scale_bars(
    ax,
    h_pos,
    v_pos,
    h_length,
    v_length,
    line_width=1,
    h_label="",
    v_label="",
    font_size=6,
):
    """
    Add horizontal and vertical scale bars to a Matplotlib axes.

    :param ax: The axes to add scale bars to.
    :param h_pos: Tuple (x, y) for the starting position of the horizontal scale bar.
    :param v_pos: Tuple (x, y) for the starting position of the vertical scale bar.
    :param h_length: Length of the horizontal scale bar.
    :param v_length: Length of the vertical scale bar.
    :param line_width: Width of the scale bar lines.
    :param h_label: Label for the horizontal scale bar.
    :param v_label: Label for the vertical scale bar.
    """
    # Add horizontal scale bar
    ax.hlines(
        h_pos[1], h_pos[0], h_pos[0] + h_length, color="black", linewidth=line_width
    )
    # Position the horizontal label below the line
    ax.text(
        h_pos[0] + h_length / 2,
        h_pos[1] - 1.25,
        h_label,
        verticalalignment="top",
        horizontalalignment="center",
        fontsize=font_size,
    )

    # Add vertical scale bar
    ax.vlines(
        v_pos[0], v_pos[1], v_pos[1] + v_length, color="black", linewidth=line_width
    )
    # Rotate the vertical label by 90 degrees
    ax.text(
        v_pos[0] - 0.25,
        v_pos[1] + v_length - 2,
        v_label,
        verticalalignment="center",
        horizontalalignment="right",
        rotation=90,
        fontsize=font_size,
    )


def add_scale_bars_wvf(
    ax,
    h_pos,
    v_pos,
    h_length,
    v_length,
    line_width=1,
    h_label="",
    v_label="",
    font_size=10,
):
    """
    Add horizontal and vertical scale bars to a Matplotlib axes.

    :param ax: The axes to add scale bars to.
    :param h_pos: Tuple (x, y) for the starting position of the horizontal scale bar.
    :param v_pos: Tuple (x, y) for the starting position of the vertical scale bar.
    :param h_length: Length of the horizontal scale bar.
    :param v_length: Length of the vertical scale bar.
    :param line_width: Width of the scale bar lines.
    :param h_label: Label for the horizontal scale bar.
    :param v_label: Label for the vertical scale bar.
    """
    # Add horizontal scale bar
    ax.hlines(
        h_pos[1], h_pos[0], h_pos[0] + h_length, color="black", linewidth=line_width
    )
    # Position the horizontal label below the line
    ax.text(
        h_pos[0] + h_length / 2,
        h_pos[1] - 10,
        h_label,
        verticalalignment="top",
        horizontalalignment="center",
        fontsize=font_size,
    )

    # Add vertical scale bar
    ax.vlines(
        v_pos[0], v_pos[1], v_pos[1] + v_length, color="black", linewidth=line_width
    )
    # Rotate the vertical label by 90 degrees
    ax.text(
        v_pos[0] - 0.25,
        v_pos[1] + v_length - 50,
        v_label,
        verticalalignment="center",
        horizontalalignment="right",
        rotation=90,
        fontsize=font_size,
    )


def get_stim_colormap():
    palette = sns.color_palette("deep")

    color_map = {}
    # place less common currents last
    currents = [2, 3, 4, 5, 6, 7, 8, 9, 1, 10]
    for i, current in enumerate(currents):
        color_map[current] = palette[i]
    color_map[11] = palette[0]
    color_map[12] = palette[1]
    color_map[13] = palette[2]
    return color_map
