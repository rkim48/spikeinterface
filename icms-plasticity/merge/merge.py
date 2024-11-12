import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import spikeinterface.full as si
from spikeinterface.curation.auto_merge import get_potential_auto_merge

import batch_process.util.file_util as file_util
import merge.merge_util as merge_util
import merge.classify_cell_type as cell_type
import batch_process.util.misc as misc
from batch_process.util.ppt_image_inserter import PPTImageInserter
import batch_process.util.plotting as plotting


#
class Merger:
    def __init__(self, analyzer, save_folder):
        self.analyzer = analyzer
        self.sorting = self.analyzer.sorting
        self.save_folder = save_folder
        self.radius_um = None

        self.merge_analyzer_path = self.save_folder / "merge_analyzer.zarr"

        # if self.merge_analyzer_path.exists():
        #     print("Loading existing merge analyzer...")
        #     self.merge_analyzer = si.load_sorting_analyzer(folder=self.merge_analyzer_path)
        # else:
        #     print("Merge analyzer not found. Creating new one...")
        self.merge_analyzer = None

        self.curation_sorting_h = None
        self.cell_type_dict = None

        self.hmerge_analyzer = None

    def create_merge_analyzer(self):
        # if self.merge_analyzer_path.exists():
        #     print("Merge analyzer already exists. Skipping creation.")
        #     self.merge_analyzer = si.load_sorting_analyzer(folder=self.merge_analyzer_path)
        #     return

        property_keys = self.sorting.get_property_keys()
        accept_mask = np.where(self.sorting.get_property("accept") == 1)[0]
        accepted_units = self.sorting.unit_ids[accept_mask]
        merge_analyzer = self.analyzer.select_units(accepted_units)
        merge_sorting = self.sorting.select_units(accepted_units)

        self.merge_analyzer = si.create_sorting_analyzer(
            sorting=merge_sorting,
            recording=merge_analyzer.recording,
            format="zarr",
            folder=Path(self.save_folder) / "merge_analyzer.zarr",
            sparse=False,
            overwrite=True,
            max_spikes_per_unit=None,
        )

        self.merge_analyzer.compute("random_spikes")
        self.merge_analyzer.compute("waveforms")
        self.merge_analyzer.compute("templates")
        self.merge_analyzer.compute("template_similarity")
        self.merge_analyzer.compute("correlograms", window_ms=100, bin_ms=0.5)
        self.merge_analyzer.compute("spike_amplitudes")

    def compute_merge_analyzer_unit_locations(self, radius_um=120):
        self.radius_um = radius_um
        self.merge_analyzer.compute(
            "unit_locations", method="center_of_mass", radius_um=self.radius_um, feature="ptp")

    def check_analyzer_extensions(self, analyzer):

        extensions_needed = [
            "random_spikes",
            "waveforms",
            "templates",
            "template_similarity",
            "correlograms",
            "unit_locations",
            "spike_amplitudes",
        ]

        for extension in extensions_needed:
            if not analyzer.has_extension(extension):
                self.analyzer.compute(extension)

        print("All extensions computed.")

    def get_horizontal_merge_groups(
        self,
        merge_candidate_ids,
        cell_types,
        similarity_matrix,
        distance_matrix,
        similarity_threshold=0.9,
        distance_um_threshold=25,
    ):

        n_units = similarity_matrix.shape[0]
        groups = []
        visited = np.zeros(n_units, dtype=bool)

        def dfs(i, group):
            stack = [i]
            while stack:
                node = stack.pop()
                if not visited[node]:
                    visited[node] = True
                    group.append(node)
                    for neighbor in range(n_units):
                        if (
                            similarity_matrix[node,
                                              neighbor] > similarity_threshold
                            and distance_matrix[node, neighbor] < distance_um_threshold
                            and not visited[neighbor]
                            and cell_types[node] == cell_types[neighbor]
                        ):
                            stack.append(neighbor)

        for i in range(n_units):
            if not visited[i]:
                group = []
                dfs(i, group)
                groups.append(group)

        units_to_merge = [[merge_candidate_ids[i]
                           for i in group] for group in groups]
        merge_groups = [group for group in units_to_merge if len(group) > 1]
        return merge_groups

    def initialize_curation_sorting_h(self):
        self.curation_sorting_h = si.CurationSorting(
            sorting=self.merge_analyzer.sorting, make_graph=True)

    def load_data_for_vertical_merge(self):
        # vertical merge curation sorting should use complete horizontal merge curation sorting
        self.hmerge_analyzer = si.load_sorting_analyzer(
            folder=Path(data_folder) / "batch_sort/merge/hmerge_analyzer.zarr"
        )
        self.curation_sorting_v = si.CurationSorting(
            sorting=self.hmerge_analyzer.sorting, make_graph=True)
        self.check_analyzer_extensions(self.hmerge_analyzer)

    def classify_units_into_cell_types(self, analyzer):
        # pass merge_analyzer or h_merge_analyzer as initial analyzer
        self.cell_type_dict = cell_type.classify_units_into_cell_types(
            analyzer)

    def horizontal_merge(self, hmerge_plot_flag=False):
        print("Performing horizontal merge...")

        self.classify_units_into_cell_types(self.merge_analyzer)

        unit_ids = self.merge_analyzer.unit_ids
        unit_id_properties = misc.get_unit_id_properties(
            self.merge_analyzer.sorting)
        property_keys = self.merge_analyzer.sorting.get_property_keys()
        parent_ids = [key for key in property_keys if "parent" in key]
        parent_ids = sorted(
            parent_ids, key=lambda x: int(x.split("parent_id")[1]))

        if hmerge_plot_flag:
            ppt_inserter = PPTImageInserter(grid_dims=(
                2, 1), spacing=(0.05, 0.05), title_font_size=16)

        for parent_id in parent_ids:
            parent_property = self.merge_analyzer.sorting.get_property(
                parent_id)
            merge_candidate_ids = unit_ids[parent_property == 1]

            if len(merge_candidate_ids) > 1:
                cell_types = self.cell_type_dict["df"].loc[merge_candidate_ids, "cell_type"].tolist(
                )
                similarity_matrix = merge_util.get_template_cosine_similarity_matrix(
                    self.merge_analyzer, merge_candidate_ids
                )
                distance_matrix = merge_util.get_unit_distance_matrix(
                    self.merge_analyzer, merge_candidate_ids)

                merge_groups = self.get_horizontal_merge_groups(
                    merge_candidate_ids, cell_types, similarity_matrix, distance_matrix
                )

                if len(merge_groups) > 0:
                    for merge_group in merge_groups:
                        self.curation_sorting_h.merge(merge_group)

                if hmerge_plot_flag:
                    self.plot_horizontal_merge(
                        merge_candidate_ids, merge_groups, similarity_matrix)
                    img_path = self.save_folder / "horizontal.png"
                    plt.savefig(str(img_path))
                    ppt_inserter.add_image(str(img_path))
                    plt.close()

        new_unit_ids = np.arange(len(self.curation_sorting_h.sorting.unit_ids))
        sorting_h = self.curation_sorting_h.sorting.rename_units(new_unit_ids)

        self.hmerge_analyzer = si.create_sorting_analyzer(
            sorting=sorting_h,
            recording=self.merge_analyzer.recording,
            format="zarr",
            folder=Path(self.save_folder) / "hmerge_analyzer.zarr",
            sparse=False,
            overwrite=True,
        )

        if hmerge_plot_flag:
            ppt_inserter.save(self.save_folder / "horizontal_merge.pptx")

    def plot_horizontal_merge(self, merge_candidate_ids, merge_groups, similarity_matrix):
        N = len(merge_candidate_ids)
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, N + 1, height_ratios=[1, 1])

        ax_templates = fig.add_subplot(gs[0, :N])
        ax_primary_ch_template = fig.add_subplot(
            gs[0, N])  # Bottom right corner
        ax_cosine_similarity = fig.add_subplot(gs[1, N])  # Bottom right corner
        axes_acg = [fig.add_subplot(gs[1, i])
                    for i in range(N)]  # Create axes for the ACGs

        colors = sns.color_palette("deep", len(merge_candidate_ids))
        unit_colors = {unit_id: colors[i]
                       for i, unit_id in enumerate(merge_candidate_ids)}

        sparsity = si.compute_sparsity(
            self.merge_analyzer, peak_sign="neg", method="radius", radius_um=self.radius_um)

        si.plot_unit_templates(
            self.merge_analyzer,
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

        self.compute_merge_analyzer_unit_locations(self.radius_um)
        ul_ext = self.merge_analyzer.get_extension("unit_locations")
        ul_ext.get_data(outputs="by_unit")
        unit_locations = ul_ext.get_data(outputs="by_unit")

        for index, unit_id in enumerate(merge_candidate_ids):
            x = 28 * index
            y = unit_locations[unit_id][1]
            ax_templates.scatter(x, y)

        merge_util.plot_unit_primary_ch_waveforms(
            self.merge_analyzer, merge_candidate_ids, ax=ax_primary_ch_template)

        merge_util.plot_template_cosine_similarity_matrix(
            similarity_matrix, merge_candidate_ids, ax=ax_cosine_similarity
        )

        for i, unit_id in enumerate(merge_candidate_ids):
            cell_type.plot_acg_with_fit(
                self.cell_type_dict, unit_id, ax=axes_acg[i])

        # Adjust font sizes for all subplots
        for ax in fig.get_axes():
            plotting.adjust_font_sizes(ax, fontsize=10)

        # merge_groups
        if len(merge_groups) > 0:
            plt.suptitle(f"{merge_groups[0]}: merge")
        else:
            plt.suptitle(f"{merge_candidate_ids}: no merge")

        plt.tight_layout()

    def automerge_func(self, analyzer):

        potential_merges = si.get_potential_auto_merge(
            analyzer,
            min_spikes=50,
            max_distance_um=30,
            template_diff_thresh=0.45,
            corr_diff_thresh=0.45,
            contamination_thresh=0.2,
            firing_contamination_balance=0.5,
            censored_period_ms=2,
            refractory_period_ms=1,
            extra_outputs=False,
            resolve_graph=True,
        )

        return potential_merges

    def check_ccg_refractory_violation(self, analyzer):
        ccgs_ext = analyzer.get_extension("correlograms")

    def vertical_merge(self, vmerge_plot_flag=False):
        self.classify_units_into_cell_types(self.hmerge_analyzer)

        # merges = check_ccg_refractory_violation(self.hmerge_analyzer)

    def run(self, hmerge_plot_flag=False, vmerge_plot_flag=False, radius_um=120):
        if not self.merge_analyzer:
            self.create_merge_analyzer()
        self.compute_merge_analyzer_unit_locations(radius_um)
        self.initialize_curation_sorting_h()
        self.horizontal_merge(hmerge_plot_flag)

        # self.vertical_merge(vmerge_plot_flag)


def main(data_folder=None):
    job_kwargs = dict(n_jobs=5, chunk_duration="1s", progress_bar=True)
    # si.set_global_job_kwargs(**job_kwargs)

    if data_folder:
        data_folders = [data_folder]
    else:
        starting_dir = "C:\\data"
        data_folders = file_util.file_dialog(starting_dir=starting_dir)

    for i, data_folder in enumerate(data_folders):
        print("\n###########################################")
        print(f"{data_folder}: {i+1}/{len(data_folders)}")
        print("###########################################")

        save_folder = Path(data_folder) / "batch_sort/merge"
        file_util.create_folder(save_folder)
        # Load stage3 analyzer with Mountainsort units split into child units
        analyzer = si.load_sorting_analyzer(folder=Path(
            data_folder) / "batch_sort/stage3/stage3_analyzer.zarr")

        merger = Merger(analyzer, save_folder)
        merger.check_analyzer_extensions(analyzer)
        merger.run(hmerge_plot_flag=False, radius_um=120)


# %%
if __name__ == "__main__":
    # plt.style.use("default")
    # plt.style.use("seaborn-v0_8-darkgrid")

    path_1 = "E:\\data\\ICMS93\\behavior\\30-Aug-2023"
    path_2 = "C:\\data\\ICMS92\\behavior\\30-Aug-2023"

    if os.path.exists(path_1):
        data_folder = path_1
    elif os.path.exists(path_2):
        data_folder = path_2
    else:
        data_folder = None  # or raise an error, or assign a default path
        print("Neither directory exists.")

    main(data_folder)

# %%
# data_folder = "C://data//ICMS93//Behavior//30-Aug-2023"
# analyzer = si.load_sorting_analyzer(folder=Path(data_folder) / "batch_sort/merge/hmerge_analyzer.zarr")
# analyzer.sorting.to_spike_vector()[-1]
# %%
# merge_analyzer = merger.merge_analyzer
# distance_matrix = merge_util.get_unit_distance_matrix(merge_analyzer, [124, 127, 129])

# similarity_matrix = merge_util.get_template_cosine_similarity_matrix(merge_analyzer, [124, 127, 129])
