"""Utility functions for working with imaging mass spec data"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from typing import Optional

def diff_expr(
    df: pd.DataFrame,
    grouping_col: str,
    value_col: str,
    expr_threshold: int = 0,
    logfc_threshold: float = 0.25,
) -> pd.DataFrame:
    """Calculate differential expression of one feature for one group versus
    all other groups. Modeled after `find_markers()` from the R package {Seurat}

    Parameters
    ----------
    df : pd.DataFrame
        A wide-form dataframe, in the form of
        `| grouping_col | value_col |`
        where grouping_col is something like a cluster assignment
        and value_col is the expression values for one feature
        (gene, protein, etc...). Other columns may be present, but will be
        ignored
    grouping_col : str
        Some sort of group identites
    value_col : str
        Expression values for one feature
    expr_threshold : int, optional
        Threshold below which a group will not be considered for comparison.
        Decrease this value to improve adjusted P value, by default 0
    logfc_threshold : float, optional
        Fold change threshold.  If the
        group/every_other_group fold change is not greater than this, ignore.
        Increasing this will improve the adjusted p value by decreasing
        the number of multiple comparisons, by default 0.25

    Returns
    -------
    pd.DataFrame
        DataFrame in the form of
        `| avg_log2FC | statistic | pvalue | pct_1 | pct_2 | marker | padj |`
    """    
    unique_groups = df[grouping_col].unique()
    deg_dict = dict()
    stats_dict = dict()
    pct_1 = dict()
    pct_2 = dict()
    for current_group in unique_groups:
        other_groups = np.setdiff1d(unique_groups, current_group)
        # for other_group in other_groups:
        current_group_vals = df[df[grouping_col] == current_group][value_col]
        other_group_vals = df[df[grouping_col].isin(other_groups)][value_col]

        deg_dict[current_group] = np.log2(
            np.mean(current_group_vals) / np.mean(other_group_vals)
        )
        stats_dict[current_group] = mannwhitneyu(
            current_group_vals, other_group_vals, alternative="two-sided"
        )

        pct_1[current_group] = sum(
            _ > expr_threshold for _ in current_group_vals
        ) / len(current_group_vals)
        pct_2[current_group] = sum(_ > expr_threshold for _ in other_group_vals) / len(
            other_group_vals
        )
    deg_df = (
        pd.DataFrame.from_dict(
            data=deg_dict,
            orient="index",
            columns=["avg_log2FC"],
        )
        # .rename(columns={0: "avg_log2FC"})
        .merge(
            right=(
                pd.DataFrame.from_dict(
                    data=stats_dict,
                    orient="index",
                )
            ),
            left_index=True,
            right_index=True,
        )
        .merge(
            right=(
                pd.DataFrame.from_dict(
                    data=pct_1,
                    orient="index",
                    columns=["pct_1"],
                )
            ),
            left_index=True,
            right_index=True,
        )
        .merge(
            right=(
                pd.DataFrame.from_dict(
                    data=pct_2,
                    orient="index",
                    columns=["pct_2"],
                )
            ),
            left_index=True,
            right_index=True,
        )
    )

    deg_df = deg_df[abs(deg_df["avg_log2FC"]) >= logfc_threshold]
    deg_df["marker"] = value_col
    reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(deg_df["pvalue"])
    deg_df["padj"] = pvals_corrected
    return deg_df


def diff_expr_all(
    df: pd.DataFrame,
    value_col: str,
    grouping_col: str,
    n_jobs: Optional[int]=None,
    *args
    ) -> pd.DataFrame:
    """A multicomparison version of `diff_expr()` that, for a given grouping variable,
    will perform all of the 1-vs-all comparisons for that variable.  Performs
    comparisons in parallel.

    Parameters
    ----------
    df : pd.DataFrame
        A wide-form dataframe, in the form of
        `| grouping_col | value_col |`
        where grouping_col is something like a cluster assignment
        and value_col is the expression values for one feature
        (gene, protein, etc...). Other columns may be present, but will be
        ignored
    value_cols : str
         Expression values for one feature
    grouping_col : str
        [description]
    n_jobs : Optional[int], optional
        [description], by default None
    args : additional arguments to pass along to `diff_expr()`

    Returns
    -------
    pd.DataFrame
        [description]
    """    
    
    if n_jobs is None:
        n_jobs=cpu_count()
    
    diff_res_dfs = Parallel(n_jobs=n_jobs)(
        delayed(diff_expr)(df, grouping_col, _, *args) for _ in value_col
        )
    
    return pd.concat(diff_res_dfs)