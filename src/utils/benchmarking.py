import time

from functools import wraps
from typing import Callable


def benchmark(function: Callable) -> Callable:
    """Measures a method performance.
    """

    instances_lookup = dict()

    @wraps(function)
    def wrapper(*args, **kwargs):
        start_sec = time.time()
        return_value = function(*args, **kwargs)
        finish_sec = time.time()
        elapsed_time_sec = finish_sec - start_sec
        print(f"** Finished {function.__name__} in {elapsed_time_sec} sec. **")
        return return_value

    return wrapper
