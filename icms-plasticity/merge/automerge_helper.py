from spikeinterface import full as si
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.pyplot as plt
import itertools
from spikeinterface.widgets import plot_unit_templates
from spikeinterface.curation import get_potential_auto_merge

from batch_process.util.ppt_image_inserter import *

# %%


def perform_grid_search(analyzer, save_dir, df=None):
    # Define parameter lists

    # For testing
    contamination_thresh_list = [0.2]
    corr_diff_thresh_list = [0.3, 0.4, 0.6]
    template_diff_thresh_list = [0.2, 0.3, 0.6]
    fc_balance_list = [0.5]
    censored_period_list = [2]
    refract_period_list = [1]

    # For processing
    # maximum_distance_um_list = [60]
    # contamination_thresh_list = [0.2]
    # corr_diff_thresh_list = [0.45]
    # template_diff_thresh_list = [0.45]
    # num_channels_list = [3]
    # fc_balance_list = [0.5]
    # censored_period_list = [2]
    # refract_period_list = [1]

    step_lists = [
        "num_spikes",
        "remove_contaminated",
        "unit_locations",
        "template_similarity",
        "correlogram",
        "quality_score",
    ]

    unit_colors_template = {unit_id: "k" for unit_id in we.unit_ids}
    colormap = plt.cm.viridis

    # TODO
    we.sparsity = si.compute_sparsity(we, method="radius", radius_um=60)

    # Create a list of parameter names and values only for those with more than one value
    param_lists = {
        "contamination_thresh": contamination_thresh_list,
        "corr_diff_thresh": corr_diff_thresh_list,
        "template_diff_thresh": template_diff_thresh_list,
        "firing_contamination_balance": fc_balance_list,
        "censor_correlograms_ms": censored_period_list,
        "refractory_period_ms": refract_period_list,
    }

    varying_params = {name: values for name, values in param_lists.items() if len(values) > 1}
    fixed_params = {name: values[0] for name, values in param_lists.items() if len(values) == 1}

    # Create the parameter grid for varying parameters
    param_grid = list(itertools.product(*(values for values in varying_params.values())))

    total_iterations = len(param_grid)
    gradient_colors = [colormap(i / total_iterations) for i in range(total_iterations)]

    # Perform grid search
    ppt_inserter = PPTImageInserter(grid_dims=(3, 2), spacing=(0.05, 0.05), title_font_size=16)

    for iteration, param_values in enumerate(param_grid):
        ppt_inserter.add_slide(f"{param_values}")
        print(f"Iteration {iteration + 1} out of {total_iterations}...")
        color_for_iteration = gradient_colors[iteration]
        unit_colors_acg = {unit_id: color_for_iteration for unit_id in we.unit_ids}

        # Combine fixed and varying parameters
        params = {name: value for name, value in zip(varying_params.keys(), param_values)}
        params.update(fixed_params)

        merges, components = get_potential_auto_merge(
            we,
            minimum_spikes=50,
            maximum_distance_um=params["maximum_distance_um"],
            template_diff_thresh=params["template_diff_thresh"],
            num_channels=params["num_channels"],
            corr_diff_thresh=params["corr_diff_thresh"],
            contamination_threshold=params["contamination_thresh"],
            firing_contamination_balance=params["fc_balance"],
            censored_period_ms=params["censored_period"],
            refractory_period_ms=params["refract_period"],
            steps=step_lists,
            extra_outputs=True,
        )

        if df is not None:
            putative_cell_types_dict = df["cell_type"].to_dict()
            filtered_merges = filter_merges_by_cell_type(merges, putative_cell_types_dict)
            components = build_components_from_pairs(filtered_merges)

        # Plot and save figures for each merge
        for i, merge in enumerate(components):
            fig = plt.figure()
            gs = GridSpec(2, 1, height_ratios=[1, 1])

            # First subplot for unit templates
            ax1 = fig.add_subplot(gs[0])

            p1 = plot_unit_templates(
                we,
                unit_ids=merge,
                same_axis=True,
                unit_colors=unit_colors_template,
                x_offset_units=True,
                templates_percentile_shading=None,
                plot_legend=False,
                ax=ax1,
            )
            p1.ax.xaxis.set_visible(False)

            # Second subplot grid for autocorrelograms
            gs2 = GridSpecFromSubplotSpec(1, len(merge), subplot_spec=gs[1])
            autocorr_axes = [fig.add_subplot(gs2[j]) for j in range(len(merge))]

            # Plot autocorrelograms on the new set of axes
            p = si.plot_autocorrelograms(we, unit_ids=merge, axes=autocorr_axes, unit_colors=unit_colors_acg)

            # Hide x-axis and y-axis labels for the autocorrelograms
            for ax in autocorr_axes:
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.title.set_visible(False)
            plt.tight_layout()

            # Construct the image name with only varying parameters
            img_name = "merge_{}.png".format("_".join(f"{name}_{params[name]}" for name in varying_params.keys()))
            img_path = save_dir / img_name

            plt.savefig(str(img_path))

            ppt_inserter.add_image(str(img_path))

            plt.close()

    ppt_inserter.save(save_dir / "vertical_merge.pptx")


# # Example usage
# results = perform_grid_search(we, Path(save_folder) / 'test_automerge_2')


def filter_merges_by_cell_type(merges, putative_cell_types_dict):
    """
    Filters merges to ensure that only units with the same putative cell type are merged.

    Parameters:
    - merges: List of tuples, each containing a pair of unit IDs to be merged.
    - putative_cell_types: Dictionary mapping unit IDs to their putative cell types.

    Returns:
    - filtered_merges: List of tuples with merges approved by both automerge and putative cell type verification.
    """
    filtered_merges = []
    for unit1, unit2 in merges:
        if putative_cell_types_dict[unit1] == putative_cell_types_dict[unit2]:
            filtered_merges.append((unit1, unit2))
    return filtered_merges


def build_components_from_pairs(pairs):
    """
    Builds components from pairs of unit IDs.

    Parameters:
    - pairs: List of tuples, each containing a pair of unit IDs to be merged.

    Returns:
    - components: List of lists, each containing unit IDs to be merged into a final unit.
    """
    from collections import defaultdict

    parent = {}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            parent[rootY] = rootX

    # Initialize parent pointers
    for pair in pairs:
        for unit in pair:
            if unit not in parent:
                parent[unit] = unit

    # Union pairs
    for x, y in pairs:
        union(x, y)

    # Build components
    components = defaultdict(list)
    for unit in parent:
        root = find(unit)
        components[root].append(unit)

    return list(components.values())
