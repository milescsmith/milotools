"""Plotting Utils."""
import numpy as np
import pandas as pd
import seaborn as sns


def map_series_palette(sr: pd.Series, palette: str = "hls") -> pd.DataFrame:
    if str(sr.dtype) == "category":
        return pd.DataFrame(
            sr.astype(int).map(
                dict(zip(sr.unique(), sns.color_palette(palette, len(sr.unique()))))
            )
        )
    else:
        return pd.DataFrame(
            sr.map(dict(zip(sr.unique(), sns.color_palette(palette, len(sr.unique())))))
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
