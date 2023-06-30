import warnings

from functools import wraps
from typing import Callable


def deprecated(function: Callable) -> Callable:
    """This is a decorator which can be used to mark functions as deprecated.

    Adding this decorator will result in a deprecation warning being emitted when the function is used.
    """

    @wraps(function)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(f"Call to deprecated function {function.__name__}.",
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)
        return function(*args, **kwargs)

    return new_func
