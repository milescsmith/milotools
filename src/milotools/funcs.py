from functools import wraps


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