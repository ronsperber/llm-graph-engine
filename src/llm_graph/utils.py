"""
module containing any utility functions
"""

from typing import Callable, Protocol, Any
from inspect import signature
from pydantic import BaseModel, ValidationError
from openai.types.chat import ChatCompletionMessageParam
import functools
import json


def json_parse(s: str) -> dict[str, Any]:
    """
    helper function to try to parse output into a dict
    Parameters
    ----------
    s : str
        raw text from LLM
    Returns
    -------
        parsed : dict
            when the string is in proper JSON format, this is the parsed dict from that
            when that was unable to be done, it returns a dict with 'raw_output' and 'parse_error'
            keys
    """
    try:
        parsed = json.loads(s)
    except Exception as e:
        return {
            "raw_output": s,
            "parse_error_message": f"{type(e).__name__}: {e}",
            "parse_error": True,
        }
    return {"parse_error": False} | parsed


class ResponseFn(Protocol):
    def __call__(
        self, messages: list[ChatCompletionMessageParam], **kwargs: Any
    ) -> dict[str, Any]: ...


def tool_call(
    input_key: str, output_key: str, model: BaseModel | None = None
) -> Callable:
    """
    decorator to turn a function into a usable callable for a FunctionalNode
    Parameters
    ----------
    input_key: str
        key used to get the inputs to the function
    output_key: str
        key used to return the output
    model: BaseModel | None
        model to validate the arguments against
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
            # validate the arguments against the model
            if model is not None:
                try:
                    validated = model.model_validate(kwargs)
                    kwargs = validated.model_dump()
                except ValidationError as e:
                    tool_success = False
                    result = str(e)
                    return {
                        output_key: result,
                        f"{output_key}_success": tool_success,
                        f"{output_key}_args": kwargs,
                    }
            # apply the function and return it as a value in the dict
            # if the function fails do to an exception we return the exception as well
            try:
                result = func(**kwargs)
            except Exception as e:
                # if the function can't be applied, store the exception and set
                # success to False
                result = str(e)
                tool_success = False

            return {
                output_key: result,
                f"{output_key}_success": tool_success,
                f"{output_key}_args": kwargs,
            }

        # store the tool metadata
        wrapper.tool_meta = {
            "input_key": input_key,
            "output_key": output_key,
            "schema_model": model,
            "tool_name": func.__name__,
            "tool_doc": func.__doc__,
            "tool_signature": str(signature(func)),
        }
        # store the original function
        wrapper.original_func = func
        return wrapper

    return decorator
