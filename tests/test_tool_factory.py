"""
Tests for llm_graph.factories.tool

Coverage targets beyond the existing test_parse_error.py suite:
  - get_tool_metadata (decorated and undecorated functions)
  - default_tool_prompt / wrap_tool_prompt / wrap_tool_output / default_tool_summary_prompt
  - create_tool_node / create_tool_llm_node / create_tool_llm_pair / create_tool_analysis_node
  - retry_llm_prompt / retry_tool_call_prompt (content assertions)
  - create_retry_llm_conditional / create_retry_tool_conditional
  - create_conditional_parse_retry_node / create_conditional_tool_retry_node
  - create_retry_parse_error_pair / create_retry_tool_error_pair
  - Integration: full graph execution with parse-error retry and tool-error retry
"""

import pytest
from inspect import signature
from unittest.mock import MagicMock

from llm_graph.core.nodes import FunctionalNode, ConditionalNode
from llm_graph.core.graphrunner import GraphRunner
from llm_graph.utils import tool_call, json_parse
from llm_graph.factories.tool import (
    get_tool_metadata,
    default_tool_prompt,
    wrap_tool_prompt,
    wrap_tool_output,
    default_tool_summary_prompt,
    create_tool_node,
    create_tool_llm_node,
    create_tool_llm_pair,
    create_tool_analysis_node,
    retry_llm_prompt,
    retry_tool_call_prompt,
    create_retry_llm_conditional,
    create_retry_tool_conditional,
    create_conditional_parse_retry_node,
    create_conditional_tool_retry_node,
    create_retry_parse_error_pair,
    create_retry_tool_error_pair,
    create_retry_llm_node,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@tool_call(output_key="sqrt_output", input_key="sqrt_input")
def sqrt_tool(x: float) -> float:
    """Return the square root of x. Raises on negative input."""
    if x < 0:
        raise ValueError(f"x must be non-negative, got {x}")
    return x ** 0.5


def undecorated_tool(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def no_doc_tool(x):
    pass


def make_response_fn(responses: list[str]):
    """
    Return a response_fn that cycles through a list of raw_output strings.
    Each call pops the next response.
    """
    remaining = list(responses)

    def response_fn(history, **kwargs):
        raw = remaining.pop(0) if remaining else "{}"
        return {"raw_output": raw}

    return response_fn


def mock_response_fn(history, **kwargs):
    """Generic mock that returns valid JSON for sqrt_tool."""
    return {"raw_output": '{"sqrt_input": {"x": 9}}'}


# ---------------------------------------------------------------------------
# get_tool_metadata
# ---------------------------------------------------------------------------

class TestGetToolMetadata:

    def test_decorated_function_uses_tool_meta(self):
        meta = get_tool_metadata(sqrt_tool)
        assert meta == sqrt_tool.tool_meta

    def test_decorated_function_has_correct_keys(self):
        meta = get_tool_metadata(sqrt_tool)
        assert meta["input_key"] == "sqrt_input"
        assert meta["output_key"] == "sqrt_output"
        assert meta["tool_name"] == "sqrt_tool"
        assert "square root" in meta["tool_doc"]

    def test_undecorated_function_reconstructs_metadata(self):
        meta = get_tool_metadata(undecorated_tool)
        assert meta["tool_name"] == "undecorated_tool"
        assert meta["input_key"] == "tool_undecorated_tool_args"
        assert meta["output_key"] == "tool_undecorated_tool_output"
        assert meta["schema_model"] is None
        assert meta["tool_doc"] == "Add two numbers."

    def test_undecorated_function_signature_is_string(self):
        meta = get_tool_metadata(undecorated_tool)
        assert isinstance(meta["tool_signature"], str)
        assert "a" in meta["tool_signature"]
        assert "b" in meta["tool_signature"]

    def test_function_without_docstring_gets_default_doc(self):
        meta = get_tool_metadata(no_doc_tool)
        assert meta["tool_doc"] == "No docstring provided."


# ---------------------------------------------------------------------------
# default_tool_prompt
# ---------------------------------------------------------------------------

class TestDefaultToolPrompt:

    def test_contains_tool_name(self):
        prompt = default_tool_prompt(sqrt_tool)
        assert "sqrt_tool" in prompt

    def test_contains_tool_doc(self):
        prompt = default_tool_prompt(sqrt_tool)
        assert "square root" in prompt

    def test_contains_input_key(self):
        prompt = default_tool_prompt(sqrt_tool)
        assert "sqrt_input" in prompt

    def test_contains_default_query_key_placeholder(self):
        prompt = default_tool_prompt(sqrt_tool)
        assert "{user_query}" in prompt

    def test_custom_query_key_in_prompt(self):
        prompt = default_tool_prompt(sqrt_tool, query_key="my_query")
        assert "{my_query}" in prompt
        assert "{user_query}" not in prompt

    def test_returns_string(self):
        assert isinstance(default_tool_prompt(sqrt_tool), str)

    def test_works_for_undecorated_tool(self):
        prompt = default_tool_prompt(undecorated_tool)
        assert "undecorated_tool" in prompt
        assert "tool_undecorated_tool_args" in prompt


# ---------------------------------------------------------------------------
# wrap_tool_prompt
# ---------------------------------------------------------------------------

class TestWrapToolPrompt:

    def test_adds_tool_info_when_input_key_absent(self):
        base = "Please generate arguments for the user query: {user_query}"
        wrapped = wrap_tool_prompt(sqrt_tool, base)
        assert "sqrt_tool" in wrapped
        assert "sqrt_input" in wrapped
        assert "square root" in wrapped
        assert base in wrapped

    def test_returns_original_when_input_key_already_present(self):
        # If the template already references the input_key placeholder, skip wrapping
        already_has_key = "Use {sqrt_input} for the tool."
        result = wrap_tool_prompt(sqrt_tool, already_has_key)
        assert result == already_has_key

    def test_wrapped_prompt_still_contains_original_template(self):
        base = "Some base instructions here."
        wrapped = wrap_tool_prompt(sqrt_tool, base)
        assert "Some base instructions here." in wrapped


# ---------------------------------------------------------------------------
# wrap_tool_output
# ---------------------------------------------------------------------------

class TestWrapToolOutput:

    def test_adds_output_key_when_absent(self):
        base = "Summarize the result for the user."
        wrapped = wrap_tool_output(sqrt_tool, base)
        assert "{sqrt_output}" in wrapped
        assert "sqrt_tool" in wrapped
        assert "square root" in wrapped

    def test_includes_original_template_content(self):
        base = "Summarize the result for the user."
        wrapped = wrap_tool_output(sqrt_tool, base)
        assert "Summarize the result for the user." in wrapped

    def test_returns_as_is_when_output_key_already_present(self):
        base = "The tool returned {sqrt_output}, analyze it."
        result = wrap_tool_output(sqrt_tool, base)
        assert result == base

    def test_returns_string(self):
        result = wrap_tool_output(sqrt_tool, "Some prompt.")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# default_tool_summary_prompt
# ---------------------------------------------------------------------------

class TestDefaultToolSummaryPrompt:

    def test_contains_tool_name(self):
        prompt = default_tool_summary_prompt(sqrt_tool)
        assert "sqrt_tool" in prompt

    def test_contains_output_key_placeholder(self):
        prompt = default_tool_summary_prompt(sqrt_tool)
        assert "{sqrt_output}" in prompt

    def test_contains_query_key_placeholder(self):
        prompt = default_tool_summary_prompt(sqrt_tool)
        assert "{user_query}" in prompt

    def test_custom_query_key(self):
        prompt = default_tool_summary_prompt(sqrt_tool, query_key="question")
        assert "{question}" in prompt
        assert "{user_query}" not in prompt

    def test_contains_tool_doc(self):
        prompt = default_tool_summary_prompt(sqrt_tool)
        assert "square root" in prompt


# ---------------------------------------------------------------------------
# create_tool_node
# ---------------------------------------------------------------------------

class TestCreateToolNode:

    def test_returns_functional_node(self):
        node = create_tool_node(sqrt_tool, name="my_tool")
        assert isinstance(node, FunctionalNode)

    def test_node_name_is_set(self):
        node = create_tool_node(sqrt_tool, name="my_tool")
        assert node.name == "my_tool"

    def test_next_node_name_is_set(self):
        node = create_tool_node(sqrt_tool, name="t", next_node_name="analysis")
        assert node.next_node_name == "analysis"

    def test_next_node_name_defaults_to_none(self):
        node = create_tool_node(sqrt_tool, name="t")
        assert node.next_node_name is None

    def test_node_func_is_tool(self):
        node = create_tool_node(sqrt_tool, name="t")
        assert node.func is sqrt_tool

    def test_node_executes_tool_correctly(self):
        node = create_tool_node(sqrt_tool, name="t")
        state = {"sqrt_input": {"x": 4}, "user_query": "what is sqrt(4)?"}
        result = node.execute(state)
        assert result["sqrt_output"] == pytest.approx(2.0)
        assert result["sqrt_output_success"] is True


# ---------------------------------------------------------------------------
# retry_llm_prompt
# ---------------------------------------------------------------------------

class TestRetryLlmPrompt:

    def _make_parse_error_state(self):
        raw = "this is not json"
        parsed = json_parse(raw)
        return {"user_query": "find sqrt of 9", "raw_output": raw} | parsed

    def test_contains_tool_name(self):
        state = self._make_parse_error_state()
        prompt = retry_llm_prompt(tool=sqrt_tool, state=state)
        assert "sqrt_tool" in prompt

    def test_contains_raw_output(self):
        state = self._make_parse_error_state()
        prompt = retry_llm_prompt(tool=sqrt_tool, state=state)
        assert "this is not json" in prompt

    def test_contains_parse_error_message(self):
        state = self._make_parse_error_state()
        prompt = retry_llm_prompt(tool=sqrt_tool, state=state)
        assert state["parse_error_message"] in prompt

    def test_default_base_prompt_included(self):
        state = self._make_parse_error_state()
        prompt = retry_llm_prompt(tool=sqrt_tool, state=state)
        # default_tool_prompt content should appear in base
        assert "sqrt_input" in prompt

    def test_custom_template_used_as_base(self):
        state = self._make_parse_error_state()
        custom = "Use {user_query} to figure out the arguments."
        prompt = retry_llm_prompt(tool=sqrt_tool, state=state, prompt_template=custom)
        assert "Use {user_query} to figure out the arguments." in prompt


# ---------------------------------------------------------------------------
# retry_tool_call_prompt
# ---------------------------------------------------------------------------

class TestRetryToolCallPrompt:

    def _make_tool_error_state(self):
        raw = '{"sqrt_input": {"x": -5}}'
        parsed = json_parse(raw)
        state = {"user_query": "find sqrt of -5"} | parsed
        tool_result = sqrt_tool(state)
        return state | tool_result

    def test_contains_tool_name(self):
        state = self._make_tool_error_state()
        prompt = retry_tool_call_prompt(tool=sqrt_tool, state=state)
        assert "sqrt_tool" in prompt

    def test_contains_tool_error_message(self):
        state = self._make_tool_error_state()
        prompt = retry_tool_call_prompt(tool=sqrt_tool, state=state)
        assert "x must be non-negative" in prompt

    def test_default_base_prompt_included(self):
        state = self._make_tool_error_state()
        prompt = retry_tool_call_prompt(tool=sqrt_tool, state=state)
        assert "sqrt_input" in prompt

    def test_custom_template_used_as_base(self):
        state = self._make_tool_error_state()
        custom = "Adjust args for {user_query}."
        prompt = retry_tool_call_prompt(
            tool=sqrt_tool, state=state, prompt_template=custom
        )
        assert "Adjust args for {user_query}." in prompt


# ---------------------------------------------------------------------------
# create_retry_llm_conditional
# ---------------------------------------------------------------------------

class TestCreateRetryLlmConditional:

    def setup_method(self):
        self.tool_node = FunctionalNode(func=sqrt_tool, name="tool_exec")
        self.retry_node = FunctionalNode(func=sqrt_tool, name="retry_llm")
        self.conditional_fn = create_retry_llm_conditional(
            tool_node=self.tool_node,
            retry_llm_node=self.retry_node,
        )

    def test_routes_to_retry_on_parse_error(self):
        state = {"parse_error": True}
        assert self.conditional_fn(state) == "retry_llm"

    def test_routes_to_tool_when_no_parse_error(self):
        state = {"parse_error": False}
        assert self.conditional_fn(state) == "tool_exec"

    def test_routes_to_tool_when_parse_error_key_absent(self):
        # Missing key should default to no error → go to tool
        state = {}
        assert self.conditional_fn(state) == "tool_exec"

    def test_is_callable(self):
        assert callable(self.conditional_fn)


# ---------------------------------------------------------------------------
# create_retry_tool_conditional
# ---------------------------------------------------------------------------

class TestCreateRetryToolConditional:

    def setup_method(self):
        self.analysis_node = FunctionalNode(func=lambda s: {}, name="analysis")
        self.retry_node = FunctionalNode(func=lambda s: {}, name="retry_tool")
        self.conditional_fn = create_retry_tool_conditional(
            tool_analysis_node=self.analysis_node,
            retry_tool_node=self.retry_node,
            output_key="sqrt_output",
        )

    def test_routes_to_retry_when_success_is_false(self):
        state = {"sqrt_output_success": False}
        assert self.conditional_fn(state) == "retry_tool"

    def test_routes_to_analysis_when_success_is_true(self):
        state = {"sqrt_output_success": True}
        assert self.conditional_fn(state) == "analysis"

    def test_routes_to_retry_when_success_key_absent(self):
        # Missing key → default False → retry
        state = {}
        assert self.conditional_fn(state) == "retry_tool"


# ---------------------------------------------------------------------------
# Node creation: structural tests
# ---------------------------------------------------------------------------

class TestCreateToolLlmNode:

    def test_returns_functional_node(self):
        node = create_tool_llm_node(
            tool=sqrt_tool,
            response_fn=mock_response_fn,
            name="llm_args",
        )
        assert isinstance(node, FunctionalNode)

    def test_correct_name(self):
        node = create_tool_llm_node(
            tool=sqrt_tool, response_fn=mock_response_fn, name="llm_args"
        )
        assert node.name == "llm_args"

    def test_next_node_name_forwarded(self):
        node = create_tool_llm_node(
            tool=sqrt_tool,
            response_fn=mock_response_fn,
            name="llm_args",
            next_node_name="exec_tool",
        )
        assert node.next_node_name == "exec_tool"

    def test_default_prompt_embedded_when_none_given(self):
        node = create_tool_llm_node(
            tool=sqrt_tool, response_fn=mock_response_fn, name="n"
        )
        # The LLMCall inside should have a prompt referencing the tool
        llm_call = node.func
        assert "sqrt_tool" in llm_call.prompt_template

    def test_custom_prompt_wrapped_into_node(self):
        custom = "Just give args for {user_query}."
        node = create_tool_llm_node(
            tool=sqrt_tool,
            response_fn=mock_response_fn,
            name="n",
            prompt_template=custom,
        )
        llm_call = node.func
        # wrapped prompt should contain the custom text and tool name
        assert "Just give args for {user_query}." in llm_call.prompt_template
        assert "sqrt_tool" in llm_call.prompt_template


class TestCreateToolLlmPair:

    def test_returns_two_functional_nodes(self):
        nodes = create_tool_llm_pair(
            tool=sqrt_tool,
            response_fn=mock_response_fn,
            llm_node_name="gen_args",
            tool_node_name="run_tool",
        )
        assert isinstance(nodes["gen_args"], FunctionalNode)
        assert isinstance(nodes["run_tool"], FunctionalNode)

    def test_llm_node_points_to_tool_node(self):
        nodes = create_tool_llm_pair(
            tool=sqrt_tool,
            response_fn=mock_response_fn,
            llm_node_name="gen_args",
            tool_node_name="run_tool",
        )
        assert nodes["gen_args"].next_node_name == "run_tool"

    def test_tool_node_points_to_given_next(self):
        nodes = create_tool_llm_pair(
            tool=sqrt_tool,
            response_fn=mock_response_fn,
            llm_node_name="gen_args",
            tool_node_name="run_tool",
            tool_node_next_node_name="analysis",
        )
        assert nodes["run_tool"].next_node_name == "analysis"

    def test_tool_node_name_is_correct(self):
        nodes = create_tool_llm_pair(
            tool=sqrt_tool,
            response_fn=mock_response_fn,
            llm_node_name="gen_args",
            tool_node_name="run_tool",
        )
        assert nodes["run_tool"].name == "run_tool"


class TestCreateToolAnalysisNode:

    def test_returns_functional_node(self):
        node = create_tool_analysis_node(
            tool=sqrt_tool, response_fn=mock_response_fn, name="analyze"
        )
        assert isinstance(node, FunctionalNode)

    def test_correct_name(self):
        node = create_tool_analysis_node(
            tool=sqrt_tool, response_fn=mock_response_fn, name="analyze"
        )
        assert node.name == "analyze"

    def test_default_prompt_references_output_key(self):
        node = create_tool_analysis_node(
            tool=sqrt_tool, response_fn=mock_response_fn, name="analyze"
        )
        assert "{sqrt_output}" in node.func.prompt_template

    def test_custom_prompt_wrapped_with_output_key(self):
        custom = "Summarize the findings for the user."
        node = create_tool_analysis_node(
            tool=sqrt_tool,
            response_fn=mock_response_fn,
            name="analyze",
            prompt_template=custom,
        )
        assert "{sqrt_output}" in node.func.prompt_template
        assert "Summarize the findings for the user." in node.func.prompt_template

    def test_next_node_name_forwarded(self):
        node = create_tool_analysis_node(
            tool=sqrt_tool,
            response_fn=mock_response_fn,
            name="analyze",
            next_node_name="done",
        )
        assert node.next_node_name == "done"


class TestCreateRetryLlmNode:

    def test_returns_functional_node(self):
        node = create_retry_llm_node(
            tool=sqrt_tool, response_fn=mock_response_fn, name="retry"
        )
        assert isinstance(node, FunctionalNode)

    def test_correct_name_and_next_node_name(self):
        node = create_retry_llm_node(
            tool=sqrt_tool,
            response_fn=mock_response_fn,
            name="retry",
            next_node_name="check_parse",
        )
        assert node.name == "retry"
        assert node.next_node_name == "check_parse"

    def test_prompt_is_callable(self):
        node = create_retry_llm_node(
            tool=sqrt_tool, response_fn=mock_response_fn, name="retry"
        )
        assert callable(node.func.prompt_template)


class TestCreateConditionalParseRetryNode:

    def test_returns_conditional_node(self):
        tool_node = FunctionalNode(func=sqrt_tool, name="exec_tool")
        retry_node = FunctionalNode(func=sqrt_tool, name="retry_llm")
        cond = create_conditional_parse_retry_node(
            tool_node=tool_node, retry_llm_node=retry_node, name="check_parse"
        )
        assert isinstance(cond, ConditionalNode)

    def test_correct_name(self):
        tool_node = FunctionalNode(func=sqrt_tool, name="exec_tool")
        retry_node = FunctionalNode(func=sqrt_tool, name="retry_llm")
        cond = create_conditional_parse_retry_node(
            tool_node=tool_node, retry_llm_node=retry_node, name="check_parse"
        )
        assert cond.name == "check_parse"

    def test_routes_to_retry_on_parse_error(self):
        tool_node = FunctionalNode(func=sqrt_tool, name="exec_tool")
        retry_node = FunctionalNode(func=sqrt_tool, name="retry_llm")
        cond = create_conditional_parse_retry_node(
            tool_node=tool_node, retry_llm_node=retry_node, name="check_parse"
        )
        cond.execute({"parse_error": True})
        assert cond.next_node_name == "retry_llm"

    def test_routes_to_tool_on_no_error(self):
        tool_node = FunctionalNode(func=sqrt_tool, name="exec_tool")
        retry_node = FunctionalNode(func=sqrt_tool, name="retry_llm")
        cond = create_conditional_parse_retry_node(
            tool_node=tool_node, retry_llm_node=retry_node, name="check_parse"
        )
        cond.execute({"parse_error": False})
        assert cond.next_node_name == "exec_tool"


class TestCreateRetryParseErrorPair:

    def setup_method(self):
        self.tool_node = create_tool_node(sqrt_tool, name="exec_tool")
        nodes = create_retry_parse_error_pair(
            tool=sqrt_tool,
            response_fn=mock_response_fn,
            retry_node_name="retry_parse",
            conditional_node_name="check_parse",
            tool_node=self.tool_node,
        )
        self.cond_node = nodes["check_parse"]
        self.retry_node = nodes["retry_parse"]

    def test_returns_conditional_node_first(self):
        assert isinstance(self.cond_node, ConditionalNode)

    def test_returns_functional_node_second(self):
        assert isinstance(self.retry_node, FunctionalNode)

    def test_conditional_node_name(self):
        assert self.cond_node.name == "check_parse"

    def test_retry_node_name(self):
        assert self.retry_node.name == "retry_parse"

    def test_retry_node_loops_back_to_conditional(self):
        assert self.retry_node.next_node_name == "check_parse"


class TestCreateConditionalToolRetryNode:

    def test_returns_conditional_node(self):
        analysis = FunctionalNode(func=lambda s: {}, name="analysis")
        retry = FunctionalNode(func=lambda s: {}, name="retry_tool")
        cond = create_conditional_tool_retry_node(
            tool=sqrt_tool,
            tool_analysis_node=analysis,
            retry_tool_node=retry,
            name="check_tool",
        )
        assert isinstance(cond, ConditionalNode)

    def test_correct_name(self):
        analysis = FunctionalNode(func=lambda s: {}, name="analysis")
        retry = FunctionalNode(func=lambda s: {}, name="retry_tool")
        cond = create_conditional_tool_retry_node(
            tool=sqrt_tool,
            tool_analysis_node=analysis,
            retry_tool_node=retry,
            name="check_tool",
        )
        assert cond.name == "check_tool"

    def test_routes_to_retry_on_tool_failure(self):
        analysis = FunctionalNode(func=lambda s: {}, name="analysis")
        retry = FunctionalNode(func=lambda s: {}, name="retry_tool")
        cond = create_conditional_tool_retry_node(
            tool=sqrt_tool,
            tool_analysis_node=analysis,
            retry_tool_node=retry,
            name="check_tool",
        )
        cond.execute({"sqrt_output_success": False})
        assert cond.next_node_name == "retry_tool"

    def test_routes_to_analysis_on_success(self):
        analysis = FunctionalNode(func=lambda s: {}, name="analysis")
        retry = FunctionalNode(func=lambda s: {}, name="retry_tool")
        cond = create_conditional_tool_retry_node(
            tool=sqrt_tool,
            tool_analysis_node=analysis,
            retry_tool_node=retry,
            name="check_tool",
        )
        cond.execute({"sqrt_output_success": True})
        assert cond.next_node_name == "analysis"


class TestCreateRetryToolErrorPair:

    def setup_method(self):
        self.analysis_node = FunctionalNode(func=lambda s: {}, name="analysis")
        nodes = create_retry_tool_error_pair(
            tool=sqrt_tool,
            response_fn=mock_response_fn,
            tool_analysis_node=self.analysis_node,
            retry_tool_name="retry_tool",
            check_tool_name="check_tool",
            check_parse_name="check_parse",
        )
        self.cond_node = nodes["check_tool"]
        self.retry_node = nodes["retry_tool"]

    def test_returns_conditional_node_first(self):
        assert isinstance(self.cond_node, ConditionalNode)

    def test_returns_functional_node_second(self):
        assert isinstance(self.retry_node, FunctionalNode)

    def test_conditional_node_name(self):
        assert self.cond_node.name == "check_tool"

    def test_retry_node_name(self):
        assert self.retry_node.name == "retry_tool"

    def test_retry_node_links_to_check_parse(self):
        assert self.retry_node.next_node_name == "check_parse"

    def test_routes_to_analysis_on_success(self):
        self.cond_node.execute({"sqrt_output_success": True})
        assert self.cond_node.next_node_name == "analysis"

    def test_routes_to_retry_on_failure(self):
        self.cond_node.execute({"sqrt_output_success": False})
        assert self.cond_node.next_node_name == "retry_tool"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegrationSuccessfulLlmToolPair:
    """
    Successful flow: llm_node generates valid args → tool_node executes → terminal.
    """

    def test_tool_output_correct(self):
        response_fn = make_response_fn(['{"sqrt_input": {"x": 16}}'])
        nodes = create_tool_llm_pair(
            tool=sqrt_tool,
            response_fn=response_fn,
            llm_node_name="gen_args",
            tool_node_name="exec_tool",
        )
        runner = GraphRunner(
            nodes=list(nodes.values()),
            start_node="gen_args",
        )
        result = runner.execute({"user_query": "what is sqrt(16)?"})
        state = result["state_dict"]
        assert state["sqrt_output"] == pytest.approx(4.0)
        assert state["sqrt_output_success"] is True

    def test_trace_visits_both_nodes(self):
        response_fn = make_response_fn(['{"sqrt_input": {"x": 9}}'])
        nodes = create_tool_llm_pair(
            tool=sqrt_tool,
            response_fn=response_fn,
            llm_node_name="gen_args",
            tool_node_name="exec_tool",
        )
        runner = GraphRunner(nodes=list(nodes.values()), start_node="gen_args")
        result = runner.execute({"user_query": "sqrt(9)"})
        node_names = [step["name"] for step in result["trace_log"]]
        assert "gen_args" in node_names
        assert "exec_tool" in node_names


class TestIntegrationParseErrorRetry:
    """
    Flow: llm produces bad JSON → conditional routes to retry → retry produces valid JSON
    → conditional routes to tool node.
    """

    def _build_graph(self, responses: list[str]):
        response_fn = make_response_fn(responses)
        tool_node = create_tool_node(sqrt_tool, name="exec_tool")
        parse_nodes = create_retry_parse_error_pair(
            tool=sqrt_tool,
            response_fn=response_fn,
            retry_node_name="retry_parse",
            conditional_node_name="check_parse",
            tool_node=tool_node,
        )
        llm_node = create_tool_llm_node(
            tool=sqrt_tool,
            response_fn=response_fn,
            name="gen_args",
            next_node_name="check_parse",
        )
        runner = GraphRunner(
            nodes=[llm_node, *parse_nodes.values(), tool_node],
            start_node="gen_args",
            max_node_visits=5,
        )
        return runner

    def test_recovers_from_one_parse_error(self):
        # First response is bad JSON; second is valid
        runner = self._build_graph([
            "not json at all",
            '{"sqrt_input": {"x": 25}}',
        ])
        result = runner.execute({"user_query": "sqrt of 25"})
        state = result["state_dict"]
        assert state["sqrt_output"] == pytest.approx(5.0)
        assert state["sqrt_output_success"] is True

    def test_conditional_visited_twice_on_one_parse_error(self):
        runner = self._build_graph([
            "bad",
            '{"sqrt_input": {"x": 4}}',
        ])
        result = runner.execute({"user_query": "sqrt of 4"})
        node_names = [step["name"] for step in result["trace_log"]]
        # check_parse should appear at least twice: once after llm_node, once after retry
        assert node_names.count("check_parse") >= 2

    def test_no_retry_needed_when_first_response_valid(self):
        runner = self._build_graph(['{"sqrt_input": {"x": 1}}'])
        result = runner.execute({"user_query": "sqrt(1)"})
        node_names = [step["name"] for step in result["trace_log"]]
        # retry_parse should not appear in trace
        assert "retry_parse" not in node_names
        assert result["state_dict"]["sqrt_output"] == pytest.approx(1.0)


class TestIntegrationToolErrorRetry:
    """
    Flow: LLM → tool(fails) → conditional routes to retry_tool → (new args) →
    check_parse → tool (succeeds).
    """

    def _build_graph(self, responses: list[str]):
        """
        Graph layout:
          gen_args → check_parse → exec_tool → check_tool
                        ↑                          ↓ fail
                    retry_tool  ←─────────────────┘
                    (parse-error guard before re-entering tool)
        """
        response_fn = make_response_fn(responses)

        # Terminal analysis node
        analysis_node = FunctionalNode(
            func=lambda s: {"answer": str(s.get("sqrt_output"))},
            name="analysis",
        )

        # Tool node → check_tool after execution
        tool_node = create_tool_node(sqrt_tool, name="exec_tool", next_node_name="check_tool")

        # Parse-error retry pair: guards both initial LLM output and retry_tool output
        parse_nodes = create_retry_parse_error_pair(
            tool=sqrt_tool,
            response_fn=response_fn,
            retry_node_name="retry_parse",
            conditional_node_name="check_parse",
            tool_node=tool_node,
        )

        # Tool-error conditional + retry node
        tool_nodes = create_retry_tool_error_pair(
            tool=sqrt_tool,
            response_fn=response_fn,
            tool_analysis_node=analysis_node,
            retry_tool_name="retry_tool",
            check_tool_name="check_tool",
            check_parse_name="check_parse",
        )

        # LLM node → check_parse
        llm_node = create_tool_llm_node(
            tool=sqrt_tool,
            response_fn=response_fn,
            name="gen_args",
            next_node_name="check_parse",
        )

        runner = GraphRunner(
            nodes=[llm_node, *parse_nodes.values(), tool_node,
                   *tool_nodes.values(), analysis_node],
            start_node="gen_args",
            max_node_visits=6,
        )
        return runner

    def test_recovers_from_tool_error(self):
        # First response: valid parse but invalid tool arg (x=-1)
        # Second response: valid parse and valid tool arg (x=9)
        runner = self._build_graph([
            '{"sqrt_input": {"x": -1}}',
            '{"sqrt_input": {"x": 9}}',
        ])
        result = runner.execute({"user_query": "sqrt of 9"})
        state = result["state_dict"]
        assert state["sqrt_output"] == pytest.approx(3.0)
        assert state["sqrt_output_success"] is True

    def test_check_tool_and_retry_visited_when_tool_fails(self):
        runner = self._build_graph([
            '{"sqrt_input": {"x": -4}}',
            '{"sqrt_input": {"x": 4}}',
        ])
        result = runner.execute({"user_query": "sqrt of 4"})
        node_names = [step["name"] for step in result["trace_log"]]
        assert "check_tool" in node_names
        assert "retry_tool" in node_names

    def test_no_retry_tool_when_first_call_succeeds(self):
        runner = self._build_graph(['{"sqrt_input": {"x": 100}}'])
        result = runner.execute({"user_query": "sqrt of 100"})
        node_names = [step["name"] for step in result["trace_log"]]
        assert "retry_tool" not in node_names
        assert result["state_dict"]["sqrt_output"] == pytest.approx(10.0)
