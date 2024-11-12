from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from spikeinterface import full as si
from spikeinterface.curation import CurationSorting
from spikeinterface.curation.auto_merge import get_potential_auto_merge
from spikeinterface.widgets import plot_unit_templates, plot_unit_locations

from batch_process.util.curate_util import *
from batch_process.util.subcluster_util import *
from batch_process.util.misc import get_unit_id_properties
from batch_process.util.ppt_image_inserter import PPTImageInserter
from batch_process.util.plotting import *
from merge.automerge_helper import (
    perform_grid_search,
    filter_merges_by_cell_type,
    build_components_from_pairs,
)
from merge.merge_util import *
from util.file_util import *
from merge.classify_cell_type import (
    classify_unit,
    plot_acg_with_fit,
)
import batch_process.util.template_util as template_util

# %%


def horizontal_merge_policy(
    similarity_matrix,
    distance_matrix,
    putative_cell_types,
    merge_candidate_ids,
    distance_um_threshold=25,
    similarity_threshold=0.9,
):
    """
    Determine groups of units to merge based on template similarity and cell type.

    This function identifies groups of units (from the same parent) that should be merged
    together based on a similarity matrix, their putative cell types, and a similarity threshold.

    Parameters
    ----------
    similarity_matrix : numpy.ndarray
        A 2D array where each element [i, j] represents the similarity between unit i and unit j.
    distance_matrix : numpy.ndarray
        A 2D array where each element [i, j] represents the distance in um between unit i and unit j.
    putative_cell_types : list of str
        A list of putative cell types corresponding to each unit in the similarity matrix.
    merge_candidate_ids : list of int
        A list of unit IDs that are candidates for merging.
    distance_um_threshold : float, optional
        The threshold distance between units below which units are considered close enough to merge.
    similarity_threshold : float, optional
        The template similarity threshold above which units are considered similar enough to merge.
    Returns
    -------
    merge_groups : list of list of int
        A list of groups, where each group is a list of unit IDs that should be merged together.
        Groups with only a single unit are not included in the output.
    """
    groups = find_similar_groups(
        similarity_matrix, distance_matrix, putative_cell_types, unit_distance, similarity_threshold
    )
    units_to_merge = [[merge_candidate_ids[i]
                       for i in group] for group in groups]

    # Filter out single-unit groups if no merging is needed
    merge_groups = [group for group in units_to_merge if len(group) > 1]

    return merge_groups


def plot_horizontal_merge(
    analyzer_accept, result, merge_candidate_ids, merge_groups, similarity_matrix, template_ch_dict
):
    N = len(merge_candidate_ids)
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, N + 1, height_ratios=[1, 1])

    ax_templates = fig.add_subplot(gs[0, :N])
    ax_primary_ch_template = fig.add_subplot(gs[0, N])  # Bottom right corner
    ax_cosine_similarity = fig.add_subplot(gs[1, N])  # Bottom right corner
    axes_acg = [fig.add_subplot(gs[1, i])
                for i in range(N)]  # Create axes for the ACGs

    colors = sns.color_palette("deep", len(merge_candidate_ids))
    unit_colors = {unit_id: colors[i]
                   for i, unit_id in enumerate(merge_candidate_ids)}

    sparsity = si.compute_sparsity(
        analyzer_accept, peak_sign="neg", method="radius", radius_um=150)

    si.plot_unit_templates(
        analyzer_accept,
        sparsity=sparsity,
        unit_ids=merge_candidate_ids,
        unit_colors=unit_colors,
        shade_templates=False,
        same_axis=True,
        x_offset_units=True,
        templates_percentile_shading=None,
        set_title=False,
        plot_legend=True,
        scalebar=True,
        ax=ax_templates,
    )
    ax_templates.set_title("Templates and unit locations")
    ax_templates.set_xticks([])
    ax_templates.set_ylabel("Probe position (um)")

    ul_ext = analyzer_accept.get_extension("unit_locations")
    ul_ext.get_data(outputs="by_unit")
    unit_locations = ul_ext.get_data(outputs="by_unit")

    for index, unit_id in enumerate(merge_candidate_ids):
        x = 28 * index
        y = unit_locations[unit_id][1]
        ax_templates.scatter(x, y)

    plot_unit_primary_ch_waveforms(
        analyzer, merge_candidate_ids, template_ch_dict, ax=ax_primary_ch_template)

    plot_template_cosine_similarity_matrix(
        similarity_matrix, merge_candidate_ids, ax=ax_cosine_similarity)

    for i, unit_id in enumerate(merge_candidate_ids):
        plot_acg_with_fit(result, unit_id, ax=axes_acg[i])

    # Adjust font sizes for all subplots
    for ax in fig.get_axes():
        adjust_font_sizes(ax, fontsize=10)

    # merge_groups
    if len(merge_groups) > 0:
        plt.suptitle(f"Merge {merge_groups[0]}")
    else:
        plt.suptitle("No merge")

    plt.tight_layout()


def horizontal_merge(data_folder, analyzer):
    save_folder = Path(data_folder) / "batch_sort"
    sorting = analyzer.sorting
    template_ch_dict = template_util.get_template_ch(analyzer)

    # Create directory to save horizontal merge viz
    img_save_dir = save_folder / "merge"
    create_folder(img_save_dir)

    # Get accepted unit ids using property key
    property_keys = sorting.get_property_keys()
    accept_mask = np.where(sorting.get_property("accept") == 1)[0]
    accepted_units = sorting.unit_ids[accept_mask]

    # Create new we and sorting objects containing only accepted units
    analyzer_accept = analyzer.select_units(accepted_units)
    sorting_accept = analyzer.sorting.select_units(accepted_units)

    analyzer_accept.compute("random_spikes")
    analyzer_accept.compute("waveforms")
    analyzer_accept.compute("templates")
    analyzer_accept.compute("template_similarity")
    analyzer_accept.compute("correlograms", window_ms=100, bin_ms=0.5)
    analyzer_accept.compute("spike_amplitudes")
    analyzer_accept.compute(
        "unit_locations", method="monopolar_triangulation", radius_um=200, feature="ptp")

    result = classify_child_units_into_cell_types(analyzer_accept)
    # Read dataframe from cell type classifier and set extractor sparsity for plotting
    df = result["df"]
    df.set_index("unit_id", inplace=True)

    # we_accept.sparsity = si.compute_sparsity(we_accept, method="radius", radius_um=60)

    # Initialize curation sorting object
    cs = CurationSorting(sorting=sorting_accept, make_graph=True)
    unit_ids = analyzer_accept.unit_ids
    unit_id_properties = get_unit_id_properties(sorting_accept)
    property_keys = sorting_accept.get_property_keys()
    parent_ids = [key for key in property_keys if "parent" in key]
    parent_ids = sorted(parent_ids, key=lambda x: int(x.split("parent_id")[1]))

    ppt_inserter = PPTImageInserter(grid_dims=(
        2, 1), spacing=(0.05, 0.05), title_font_size=16)

    for parent_id in parent_ids:
        parent_property = sorting_accept.get_property(parent_id)
        accept_property = sorting_accept.get_property("accept")

        merge_candidate_ids = unit_ids[parent_property == 1]

        if len(merge_candidate_ids) > 1:
            templates = get_unit_primary_ch_wvfs_dict(
                analyzer_accept, merge_candidate_ids, template_ch_dict)
            similarity_matrix = get_template_cosine_similarity_matrix(
                templates)
            putative_cell_types = df.loc[merge_candidate_ids,
                                         "cell_type"].tolist()
            merge_groups = horizontal_merge_policy(
                similarity_matrix, putative_cell_types, merge_candidate_ids)
            plot_horizontal_merge(
                analyzer_accept, result, merge_candidate_ids, merge_groups, similarity_matrix, template_ch_dict
            )

            img_path = img_save_dir / "horizontal.png"
            plt.savefig(str(img_path))
            ppt_inserter.add_image(str(img_path))
            plt.close()

            if len(merge_groups) > 0:
                for merge_group in merge_groups:
                    cs.merge(merge_group)

    ppt_inserter.save(img_save_dir / "horizontal_merge.pptx")

    cs.sorting.save_to_folder(save_folder / "sorting_h", overwrite=True)

    new_unit_ids = np.arange(len(cs.sorting.unit_ids))
    sorting_h = cs.sorting.rename_units(new_unit_ids)

    we_h = si.extract_waveforms(
        we.recording,
        sorting_h,
        folder=None,
        mode="memory",
        ms_before=1.0,
        ms_after=2.0,
        sparse=False,
        overwrite=True,
        max_spikes_per_unit=None,
        **job_kwargs,
    )

    return sorting_h, we_h


def automerge_func(we):
    steps_list = [
        "min_spikes",
        "remove_contaminated",
        "unit_positions",
        "correlogram",
        "template_similarity",
        "check_increase_score",
    ]

    merges, components, outs = get_potential_auto_merge(
        we,
        minimum_spikes=50,
        maximum_distance_um=30,
        template_diff_thresh=0.45,
        num_channels=3,
        corr_diff_thresh=0.45,
        contamination_threshold=0.2,
        firing_contamination_balance=0.5,
        censored_period_ms=2,
        refractory_period_ms=1,
        steps=steps_list,
        extra_outputs=True,
    )

    return merges, components, outs


def assign_colors_to_components(we, components):
    all_unit_ids = we.unit_ids
    num_components = len(components)
    cmap = plt.get_cmap("tab10")  # You can choose any colormap you like
    colors = cmap(np.linspace(0, 1, num_components))
    # Default color is black
    unit_colors = {unit_id: "k" for unit_id in all_unit_ids}

    for comp_idx, component in enumerate(components):
        for unit in component:
            unit_colors[unit] = colors[comp_idx]

    return unit_colors


def vertical_merge(sorting_h, we_h, save_folder):
    job_kwargs = dict(n_jobs=5, chunk_duration="1s", progress_bar=True)
    cs = CurationSorting(parent_sorting=sorting_h)
    si.compute_template_similarity(we_h)

    we_h.sparsity = None

    result = classify_child_units_into_cell_types(we_h)
    df = result["df"]
    df.set_index("unit_id", inplace=True)

    # perform_grid_search(we_h, save_dir=Path(save_folder) / "merge", df=df)
    merges, _, outs = automerge_func(we_h)

    # Filter merges by putative cell type
    putative_cell_types_dict = df["cell_type"].to_dict()

    filtered_merges = filter_merges_by_cell_type(
        merges, putative_cell_types_dict)

    # Build components from filtered pairs
    components = build_components_from_pairs(filtered_merges)
    unit_colors = assign_colors_to_components(we_h, components)

    if len(merges) > 0:
        for units_to_merge in components:
            cs.merge(units_to_merge)

        # plot unit templates and acgs with merges denoted by color
        plot_units_in_batches(
            we_h, save_folder, ppt_name="pre_merge", unit_colors=unit_colors)

        new_unit_ids = np.arange(len(cs.sorting.unit_ids))
        sorting_merge = cs.sorting.rename_units(new_unit_ids)
        we_merge = si.extract_waveforms(
            we_h.recording,
            sorting_merge,
            folder=save_folder / "waveforms_4",
            ms_before=1.0,
            ms_after=2.0,
            sparse=None,
            overwrite=True,
            max_spikes_per_unit=None,
            **job_kwargs,
        )

        si.compute_correlograms(we_merge)
        si.compute_template_similarity(we_merge)
        plot_units_in_batches(we_merge, save_folder, ppt_name="post_merge")

    else:
        we_merge = si.extract_waveforms(
            we_h.recording,
            sorting_h,
            folder=save_folder / "waveforms_4",
            ms_before=1.0,
            ms_after=2.0,
            sparse=None,
            overwrite=True,
            max_spikes_per_unit=None,
            **job_kwargs,
        )
    return we_merge


# %%
if __name__ == "__main__":
    # job_kwargs = dict(n_jobs=1, chunk_duration="1s", progress_bar=True)
    # si.set_global_job_kwargs(**job_kwargs)

    path_1 = "E:\\data\\ICMS93\\behavior\\30-Aug-2023"
    path_2 = "C:\\data\\ICMS92\\behavior\\30-Aug-2023"

    if os.path.exists(path_1):
        data_folder = path_1
    elif os.path.exists(path_2):
        data_folder = path_2
    else:
        data_folder = None  # or raise an error, or assign a default path
        print("Neither directory exists.")

    save_folder = Path(data_folder) / "batch_sort"
    analyzer = si.load_sorting_analyzer(
        folder=save_folder / "stage3/stage3_analyzer.zarr")

    analyzer.compute("random_spikes")
    analyzer.compute("waveforms")
    analyzer.compute("templates")
    analyzer.compute("template_similarity")
    analyzer.compute("correlograms", window_ms=100, bin_ms=0.5)
    analyzer.compute("spike_amplitudes")
    analyzer.compute("unit_locations")

    sorting_h, we_h = horizontal_merge(data_folder, analyzer)

    we_merge = vertical_merge(sorting_h, we_h, save_folder)

# %%
# analyzer_accept.compute("unit_locations", method="monopolar_triangulation", radius_um=75, feature="ptp")
# ul_ext = analyzer_accept.get_extension("unit_locations")
# ul_ext.get_data(outputs="by_unit")
# unit_locations = ul_ext.get_data(outputs="by_unit")


# %% Test sparsity

# analyzer = si.load_sorting_analyzer(folder=save_folder / "stage3/stage3_analyzer.zarr")

# analyzer.compute("random_spikes")
# analyzer.compute("waveforms")
# analyzer.compute("templates")
