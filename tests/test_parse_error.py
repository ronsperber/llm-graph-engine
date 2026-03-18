from llm_graph.core.nodes import FunctionalNode, ConditionalNode
from llm_graph.llm.llm_call import LLMCall
from llm_graph.factories.tool import (
    create_conditional_parse_retry_node,
    create_conditional_tool_retry_node,
)
from llm_graph.factories.tool import (
    create_tool_node,
    create_retry_llm_prompt_func,
    create_retry_tool_prompt_func,
)
from llm_graph.factories.llm import create_llm_node
from llm_graph.core.graphrunner import GraphRunner
from llm_graph.utils import tool_call, json_parse


# simple tool to test with valid/invalid inputs
@tool_call(output_key="tool_output", input_key="tool_input")
def f(x):
    if x >= 0:
        return x**0.5
    raise ValueError(f"Expected non-negative input, got {x}")


# simple basic response function to test for parse errors/tool errors
def dummy_response_fn(history, **kwargs):
    if len(history) == 1:
        # in this case return a response that can't be parsed
        return {"raw_output": "This can't be parsed"}
    if len(history) < 3:
        # in this case return a reponse that can be parsed, but is invalid
        return {"raw_output": '{\n  "tool_input": {\n    "x": -1\n  }\n}'}
    # if the message history is longer return valid json with valid input
    return {"raw_output": '{\n  "tool_input": {\n    "x": 4\n  }\n}'}


base_prompt_template = """
Create a good JSON for the arguments for the tool using the user query : {user_query}
"""


def test_default_parse_error_prompt():
    history = [1]
    prompt_func = create_retry_llm_prompt_func(tool=f)
    output = dummy_response_fn(history)
    state = {"user_query": "query"} | output | json_parse(output["raw_output"])
    prompt = prompt_func(state).format(**state)
    assert state.get("parse_error", False), "There was no parse error"
    assert f.tool_meta["tool_name"] in prompt, "Function name missing from prompt"
    assert "parse_error_message" in state, "No parse error message"
    assert state["parse_error_message"] in prompt
    assert state["user_query"] in prompt
    assert state["raw_output"] in prompt


def test_custom_parse_error_prompt():
    history = [1]
    prompt_func = create_retry_llm_prompt_func(
        tool=f, prompt_template=base_prompt_template
    )
    output = dummy_response_fn(history)
    state = {"user_query": "query"} | output | json_parse(output["raw_output"])
    prompt = prompt_func(state).format(**state)
    assert state.get("parse_error", False), "There was no parse error"
    assert f.tool_meta["tool_name"] in prompt, "Function name missing from prompt"
    assert "parse_error_message" in state, "No parse error message"
    assert state["parse_error_message"] in prompt
    assert state["user_query"] in prompt
    assert state["raw_output"] in prompt
    assert "Create a good JSON" in prompt


def test_default_tool_error_prompt():
    history = [1, 2]
    prompt_func = create_retry_tool_prompt_func(tool=f)
    output = dummy_response_fn(history)
    parsed = json_parse(output["raw_output"])
    state = {"user_query": "query"} | output | parsed
    assert f.tool_meta["input_key"] in state
    tool_output = f(state)
    state = state | tool_output
    prompt = prompt_func(state)
    assert "tool_output_success" in state, "Missing tool output success"
    assert not state["tool_output_success"], "Tool success showed as True"
    assert f.tool_meta["tool_name"] in prompt, "Missing tool name"
    assert "query" in prompt, "Missing user query"
    assert state["tool_output"] in prompt, "Missing error message"
    assert "Expected non-negative input" in state["tool_output"], "Error message wrong"


def test_custom_tool_error_prompt():
    history = [1, 2]
    prompt_func = create_retry_tool_prompt_func(
        tool=f, prompt_template=base_prompt_template
    )
    output = dummy_response_fn(history)
    parsed = json_parse(output["raw_output"])
    state = {"user_query": "query"} | output | parsed
    assert f.tool_meta["input_key"] in state
    tool_output = f(state)
    state = state | tool_output
    prompt = prompt_func(state)
    assert "tool_output_success" in state, "Missing tool output success"
    assert not state["tool_output_success"], "Tool success showed as True"
    assert f.tool_meta["tool_name"] in prompt, "Missing tool name"
    assert "query" in prompt, "Missing user query"
    assert state["tool_output"] in prompt, "Missing error message"
    assert "Expected non-negative input" in state["tool_output"], "Error message wrong"
    assert "Create a good JSON" in prompt
