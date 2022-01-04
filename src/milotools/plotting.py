"""Plotting Utils."""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from functools import singledispatch


def map_series_palette(
    sr: pd.Series, palette: str = "hls"
) -> dict[str, tuple[float, float, float]]:
    base_type = type(sr.iloc[0])
    return sr.astype(base_type).map(
        dict(zip(sr.unique(), sns.color_palette(palette, len(sr.unique()))))
    )


def scale_col(col: np.ndarray) -> np.ndarray:
    return (col - np.mean(col)) / np.std(col)


def min_max(col: np.ndarray) -> np.ndarray:
    return (col - np.min(col)) / (np.max(col) - np.min(col))


def show_values_on_bars(axs):
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = "{:.2f}".format(p.get_height())
            ax.text(_x, _y, value, ha="center")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


# stolen from https://stackoverflow.com/a/47664533
class SeabornFig2Grid:
    """hack class that allows for adding complex seaborn plots
    that are technically already their own complex subplots
    to be added together.

    Example:

    fig = plt.figure(figsize =(12,6))
    gs = gridspec.GridSpec(1,2)
    g0 = sns.JointGrid(x=np.log1p(plot_df["CD45"]), y=np.log1p(plot_df["Ecad"]))
    g1 = sns.JointGrid(x=np.log1p(plot_df["CD45"]), y=np.log1p(plot_df["Ecad"]), hue=plot_df["cluster"])
    mg0 = SeabornFig2Grid(g0.plot(sns.histplot, sns.histplot, bins=50), fig, gs[0])
    mg1 = SeabornFig2Grid(g1.plot(sns.histplot, sns.histplot, bins=50), fig, gs[1])
    gs.tight_layout(fig)
    plt.show()
    """

    def __init__(self, seaborngrid, fig, subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or isinstance(
            self.sg, sns.axisgrid.PairGrid
        ):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """Move PairGrid or Facetgrid"""
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n, m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i, j], self.subgrid[i, j])

    def _movejointgrid(self):
        """Move Jointgrid"""
        h = self.sg.ax_joint.get_position().height
        h2 = self.sg.ax_marg_x.get_position().height
        r = int(np.round(h / h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(
            r + 1, r + 1, subplot_spec=self.subplot
        )

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        # https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure = self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


def above_below(x: float, lower: float, upper: float) -> float:
    if (x > lower) and (x < upper):
        return x
    elif x <= lower:
        return lower
    elif x >= upper:
        return upper


vec_above_below = np.vectorize(above_below, otypes=[float])


@singledispatch
def quantile_trim(arr, lower: float = 0.10, upper: float = 0.99):
    print(
        f"Attempting to trim an array of length {len(arr)}, lower bounds = {lower}, upper bounds = {upper}"
    )


@quantile_trim.register
def _(arr: np.ndarray, lower: float = 0.10, upper: float = 0.99) -> np.ndarray:
    lower_bounds = np.quantile(arr, lower)
    upper_bounds = np.quantile(arr, upper)
    return vec_above_below(arr, lower_bounds, upper_bounds)


@quantile_trim.register
def _(arr: pd.Series, lower: float = 0.10, upper: float = 0.99) -> pd.Series:
    lower_bounds = np.quantile(arr, lower)
    upper_bounds = np.quantile(arr, upper)
    return arr.apply(above_below, lower=lower_bounds, upper=upper_bounds)
