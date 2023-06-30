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
