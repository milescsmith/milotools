"""Plotting Utils."""
from functools import singledispatch

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import CategoricalDtype
from typing import Any, Union

import mudata as mu
import anndata as ad


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


@singledispatch
def find_obs_column(data, group: str):
    pass


@find_obs_column.register(ad.AnnData)
def _(data: ad.AnnData, group: str) -> pd.Series:
    if group in data.obs.columns:
        group_sr = data.obs[group]
    else:
       raise KeyError(f"{group} not found in obs")
    return group_sr


@find_obs_column.register(mu.MuData)
def _(mudata: mu.MuData, group: str) -> pd.Series:
    if group in mudata.obs.columns:
        group_sr = mudata.obs[group]
    else:
        if 'rna' in mudata.mod.keys() and group in mudata['rna'].obs.columns:
            group_sr = mudata['rna'].obs[group]
        elif 'prot' in mudata.mod.keys() and group in mudata['prot'].obs.columns:
            group_sr = mudata['prot'].obs[group]
        elif 'atac' in mudata.mod.keys() and group in mudata['atac'].obs.columns:
            group_sr = mudata['atac'].obs[group]
        else:
            group_sr = pd.Series()
            for _ in mudata.mod.keys():
                if group in mudata[_].obs.columns:
                    group_sr = mudata[_].obs[group]
                    break
            if group_sr.empty:
                raise KeyError(f"{group} not found in the obs for any modality")
    return group_sr


@singledispatch
def find_embedding(data, basis: str):
    pass


@find_embedding.register(ad.AnnData)
def _(data: ad.AnnData, basis: str) -> np.ndarray:
    if basis in data.obsm:
        basis_arr = data.obsm[basis]
    else:
         raise KeyError(f"{basis} not found in obsm")
    return basis_arr


@find_embedding.register(mu.MuData)
def _(data: mu.MuData, basis: str) -> np.ndarray:
    if basis in data.obsm:
        basis_arr = data.obsm[basis]
    elif 'rna' in data.mod.keys() and basis in data['rna'].obsm:
        basis_arr = data['rna'].obsm[basis]
    elif 'prot' in data.mod.keys() and basis in data['prot'].obsm:
        basis_arr = data['prot'].obsm[basis]
    elif 'atac' in data.mod.keys() and basis in data['atac'].obsm:
        basis_arr = data['atac'].obsm[basis]
    else:
        basis_arr = np.empty((0,0))
        for _ in data.mod.keys():
            if basis in data[_].obsm:
                basis_arr = data[_].obsm[basis]
                break
        if np.size(basis_arr) == 0:
            raise KeyError(f"{basis} not found in the obsm for any modality")
    return basis_arr


def highlight_group_plot(
    data: Union[ad.AnnData, mu.MuData],
    group1: str,
    group1_val: Any,
    group2: str,
    group2_val: Any,
    group1_color: str="grey",
    group2_color: str="red",
    alpha: float=0.5,
    basis: str="umap",
    uninteresting_size: int=1,
    interesting_size: int=100,
    width: int=6,
    height: int=6,
    fontsize: int=9,
    style: str="white",
    *args,
    **kwargs
) -> None:
    group1_sr = find_obs_column(data, group1)
    group2_sr = find_obs_column(data, group2)
    
    interest_cat = CategoricalDtype(categories=[group1_val, group2_val], ordered=False)
    interest_sr = pd.Series(data=group1_val, name="of_interest", dtype=interest_cat, index=group1_sr.index)
    size_sr = pd.Series(data=uninteresting_size, name="interest_size", dtype=float, index=group1_sr.index)
    
    interest_df = pd.merge(
        left=group1_sr,
        right=group2_sr,
        left_index=True,
        right_index=True
    ).join(
        interest_sr
    ).join(
        size_sr
    )
    
    interest_df.loc[
        interest_df[(interest_df[group1] == group1_val) & (interest_df[group2] == group2_val)].index,
        "of_interest"
    ] = group2_val

    interest_df.loc[
        interest_df[(interest_df[group1] == group1_val) & (interest_df[group2] == group2_val)].index,
        "interest_size"
    ] = interesting_size
    
    try:
        basis_arr = find_embedding(data, basis)
    except KeyError:
        basis_arr = find_embedding(mudata, f"X_{basis}")
        basis = f"X_{basis}"
        
    basis_df = pd.DataFrame(
        data=basis_arr,
        index=interest_df.index
    )
    
    basis_df.columns = [f"{basis}{_+1}" for _ in basis_df.columns]
    
    interest_df = interest_df.join(basis_df)
    
    fig, ax = plt.subplots(figsize=(width,height))
    if style is None:
        sns.set_style("white")
    else:
        sns.set_style(style)

    sns.scatterplot(
        data=interest_df.sort_values("of_interest"),
        x=f"{basis}1",
        y=f"{basis}2",
        hue="of_interest",
        size="of_interest",
        sizes={group1_val: uninteresting_size, group2_val: interesting_size},
        palette={group1_val: group1_color, group2_val: group2_color},
        alpha=alpha,
        ax=ax,
        *args,
        **kwargs
    )
    plt.show()
