import time

from functools import wraps
from typing import Callable


def benchmark(function: Callable) -> Callable:
    """This is a decorator function to measure performance of the given function.

    Adding this decorator will result in printing time taken by the function in seconds
    to the standard output stream.
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        start_sec = time.time()
        return_value = function(*args, **kwargs)
        finish_sec = time.time()
        elapsed_time_sec = finish_sec - start_sec
        print(f"** Finished {function.__name__} in {elapsed_time_sec} sec. **")
        return return_value

    return wrapper


def measure(function: Callable) -> float:
    """Measures time that the execution of the given function takes.

    Parameters
    ----------
    function: Callable
        The callback to measure. Should take no arguments and return no value,
        as the value will be ignored anyway.

    Returns
    -------
    float
        Time in seconds taken by the function execution.
    """

    start_sec = time.time()

    function()

    finish_sec = time.time()
    return finish_sec - start_sec


def convert_time_to_fps(elapsed_time: float) -> float:
    """Converts the time in seconds to frames per seconds.

    The result value is always rounded to the closest integer.
    """

    return round(1.0 / elapsed_time)
