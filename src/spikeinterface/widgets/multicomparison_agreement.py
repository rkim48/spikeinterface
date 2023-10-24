import numpy as np

from .base import BaseWidget, to_attr


class MultiCompGlobalAgreementWidget(BaseWidget):
    """
    Plots multi comparison agreement as pie or bar plot.

    Parameters
    ----------
    multi_comparison: BaseMultiComparison
        The multi comparison object
    plot_type: str
        'pie' or 'bar'
    cmap: matplotlib colormap, default: 'YlOrRd'
        The colormap to be used for the nodes
    fontsize: int, default: 9
        The text fontsize
    show_legend: bool, default: True
        If True a legend is shown
    """

    def __init__(
        self,
        multi_comparison,
        plot_type="pie",
        cmap="YlOrRd",
        fontsize=9,
        show_legend=True,
        backend=None,
        **backend_kwargs,
    ):
        plot_data = dict(
            multi_comparison=multi_comparison,
            plot_type=plot_type,
            cmap=cmap,
            fontsize=fontsize,
            show_legend=show_legend,
        )
        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        mcmp = dp.multi_comparison
        cmap = plt.get_cmap(dp.cmap)
        colors = np.array([cmap(i) for i in np.linspace(0.1, 0.8, len(mcmp.name_list))])
        sg_names, sg_units = mcmp.compute_subgraphs()
        # fraction of units with agreement > threshold
        v, c = np.unique([len(np.unique(s)) for s in sg_names], return_counts=True)
        if dp.plot_type == "pie":
            p = self.ax.pie(c, colors=colors[v - 1], autopct=lambda pct: _getabs(pct, c), pctdistance=1.25)
            self.ax.legend(
                p[0],
                v,
                frameon=False,
                title="k=",
                handlelength=1,
                handletextpad=0.5,
                bbox_to_anchor=(1.0, 1.0),
                loc=2,
                borderaxespad=0.5,
                labelspacing=0.15,
                fontsize=dp.fontsize,
            )
        elif dp.plot_type == "bar":
            self.ax.bar(v, c, color=colors[v - 1])
            x_labels = [f"k={vi}" for vi in v]
            self.ax.spines["top"].set_visible(False)
            self.ax.spines["right"].set_visible(False)
            self.ax.set_xticks(v)
            self.ax.set_xticklabels(x_labels)
        else:
            raise AttributeError("Wrong plot_type. It can be 'pie' or 'bar'")
        self.ax.set_title("Units agreed upon\nby k sorters")


class MultiCompAgreementBySorterWidget(BaseWidget):
    """
    Plots multi comparison agreement as pie or bar plot.

    Parameters
    ----------
    multi_comparison: BaseMultiComparison
        The multi comparison object
    plot_type: str
        'pie' or 'bar'
    cmap: matplotlib colormap
        The colormap to be used for the nodes (default 'Reds')
    axes: list of matplotlib axes
        The axes to be used for the individual plots. If not given the required axes are created. If provided, the ax
        and figure parameters are ignored.
    show_legend: bool
        Show the legend in the last axes (default True).

    Returns
    -------
    W: MultiCompGraphWidget
        The output widget
    """

    def __init__(
        self,
        multi_comparison,
        plot_type="pie",
        cmap="YlOrRd",
        fontsize=9,
        show_legend=True,
        backend=None,
        **backend_kwargs,
    ):
        plot_data = dict(
            multi_comparison=multi_comparison,
            plot_type=plot_type,
            cmap=cmap,
            fontsize=fontsize,
            show_legend=show_legend,
        )
        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.colors as mpl_colors
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)
        mcmp = dp.multi_comparison
        name_list = mcmp.name_list

        backend_kwargs["num_axes"] = len(name_list)
        backend_kwargs["ncols"] = len(name_list)
        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        cmap = plt.get_cmap(dp.cmap)
        colors = np.array([cmap(i) for i in np.linspace(0.1, 0.8, len(mcmp.name_list))])
        sg_names, sg_units = mcmp.compute_subgraphs()
        # fraction of units with agreement > threshold
        for i, name in enumerate(name_list):
            ax = np.squeeze(self.axes)[i]
            v, c = np.unique([len(np.unique(sn)) for sn in sg_names if name in sn], return_counts=True)
            if dp.plot_type == "pie":
                p = ax.pie(
                    c,
                    colors=colors[v - 1],
                    textprops={"color": "k", "fontsize": dp.fontsize},
                    autopct=lambda pct: _getabs(pct, c),
                    pctdistance=1.18,
                )
                if (dp.show_legend) and (i == len(name_list) - 1):
                    plt.legend(
                        p[0],
                        v,
                        frameon=False,
                        title="k=",
                        handlelength=1,
                        handletextpad=0.5,
                        bbox_to_anchor=(1.15, 1.25),
                        loc=2,
                        borderaxespad=0.0,
                        labelspacing=0.15,
                    )
            elif dp.plot_type == "bar":
                ax.bar(v, c, color=colors[v - 1])
                x_labels = [f"k={vi}" for vi in v]
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_xticks(v)
                ax.set_xticklabels(x_labels)
            else:
                raise AttributeError("Wrong plot_type. It can be 'pie' or 'bar'")
            ax.set_title(name)

        if dp.plot_type == "bar":
            ylims = [np.max(ax_single.get_ylim()) for ax_single in self.axes]
            max_yval = np.max(ylims)
            for ax_single in self.axes:
                ax_single.set_ylim([0, max_yval])


def _getabs(pct, allvals):
    absolute = int(np.round(pct / 100.0 * np.sum(allvals)))
    return f"{absolute}"
