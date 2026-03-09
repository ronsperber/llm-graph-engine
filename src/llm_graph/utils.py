"""
module containing any utility functions
"""
from typing import Callable
import functools

def tool_call(input_key: str, output_key: str) -> Callable:
    """
    decorator to turn a function into a usable callable for a FunctionalNode
    Parameters
    ----------
    input_key: str
        key used to get the inputs to the function
    output_key: str
        key used to return the output
    """
    def decorator(func):

        @functools.wraps(func)
        def wrapper(state: dict):
            """
            wrapper function used with the original function
            """

            if input_key not in state:
                raise KeyError(f"{input_key} missing from state")
            # get the arguments to the function
            kwargs = state[input_key]
            # make sure it's a dict of arguments
            if not isinstance(kwargs, dict):
                raise TypeError(f"{input_key} must be dict of arguments")
            # apply the function and return it as a value in the dict
            result = func(**kwargs)

            return {output_key: result}

        return wrapper

    return decorator