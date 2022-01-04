from functools import singledispatch
from typing import Callable, Any

import pandas as pd

from .funcs import add_method


@add_method(pd.DataFrame)
def filter_by(self, col: str, other: pd.Series):
    return self[self[col].isin(other)]


@add_method(pd.DataFrame)
@singledispatch
def nrows(obj) -> int:
    """stupid, but I'm lazy
    """
    pass

@nrows.register
def _(obj: pd.Series) -> int:
    return len(pd.Series)

@nrows.register
def _(obj: pd.DataFrame) -> int:
    return obj.shape[0]


@add_method(pd.DataFrame)
def mutate(self: pd.DataFrame, col: str, func: Callable, *args: Any) -> pd.DataFrame:
    self[col] = self[col].apply(func, args)
    return self