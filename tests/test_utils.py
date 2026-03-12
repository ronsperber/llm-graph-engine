import pytest

from llm_graph.utils import tool_call


def test_tool_call_success():
    @tool_call("input", "output")
    def add(x, y):
        return x + y

    state = {"input": {"x": 1, "y": 2}}
    result = add(state)

    assert result["output"] == 3
    assert result["output_success"] is True


def test_tool_call_missing_input_key_raises_key_error():
    @tool_call("input", "output")
    def fn():
        return 1

    with pytest.raises(KeyError):
        fn({})


def test_tool_call_non_dict_input_raises_type_error():
    @tool_call("input", "output")
    def fn():
        return 1

    with pytest.raises(TypeError):
        fn({"input": "not-a-dict"})


def test_tool_call_exception_sets_flag_and_message():
    class CustomError(Exception):
        pass

    @tool_call("input", "output")
    def fn():
        raise CustomError("boom")

    result = fn({"input": {}})

    assert result["output_success"] is False
    assert "boom" in result["output"]


def test_tool_call_preserves_function_metadata():
    @tool_call("input", "output")
    def my_func():
        """Example docstring."""
        return 1

    assert my_func.__name__ == "my_func"
    assert "Example docstring." in (my_func.__doc__ or "")
