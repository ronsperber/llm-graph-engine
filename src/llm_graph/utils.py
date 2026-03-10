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
            # create flag about whether tool call succeeded
            tool_success = True
            if input_key not in state:
                raise KeyError(f"{input_key} missing from state")
            # get the arguments to the function
            kwargs = state[input_key]
            # make sure it's a dict of arguments
            if not isinstance(kwargs, dict):
                raise TypeError(f"{input_key} must be dict of arguments")
            # apply the function and return it as a value in the dict
            # if the function fails do to an exception we return the exception as well
            try:
                result = func(**kwargs)
            except Exception as e:
                # if the function can't be applied, store the exception and set
                # success to False
                result = str(e)
                tool_success = False

            return {output_key: result, f"{output_key}_success": tool_success}

        return wrapper

    return decorator