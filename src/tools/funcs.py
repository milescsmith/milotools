from functools import wraps, singledispatch

import pandas as pd


def add_method(cls):
    """#from https://stackoverflow.com/a/59089116
    can't say I completely understand this at the moment, but this allows us to bind
    a new function to an existing class definition
    so, for instance, I can add this `filter_by` function to pandas.DataFrame
    without creating a new class that inherits from DataFrame
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(cls, func.__name__, wrapper)
        return func
    return decorator


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