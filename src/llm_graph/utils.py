import functools

def tool_call(input_key: str, output_key: str):
    def decorator(func):

        @functools.wraps(func)
        def wrapper(state: dict):

            if input_key not in state:
                raise KeyError(f"{input_key} missing from state")

            args = state[input_key]

            if not isinstance(args, dict):
                raise TypeError(f"{input_key} must be dict of arguments")

            result = func(**args)

            return {output_key: result}

        return wrapper

    return decorator