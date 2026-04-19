## Component: Tool Factory (`llm_graph.factories.tool`)

The tool factory builds complete LLM-driven tool-calling workflows with automatic retry handling. It addresses two classes of failure: JSON parse errors (the LLM returned malformed JSON) and tool execution errors (the tool raised an exception or returned a failure).

### Architecture Overview

A fully-featured tool workflow has the following structure:

```
llm_node (generates args)
    ↓
parse_check (ConditionalNode: parse_error?)
    ├── yes → retry_llm_node → parse_check (loop)
    └── no  → tool_node
                ↓
           tool_check (ConditionalNode: success?)
                ├── no  → retry_tool_node → parse_check (loop)
                └── yes → analysis_node (interprets result)
```

Each factory function creates one piece of this structure. The pair functions (`create_..._pair`) create two related pieces at once and return them as a single dict for use with `GraphRunner.build()`.

---

### Metadata Helper

**`get_tool_metadata(tool) -> dict`**

Reads metadata from a `@tool_call`-decorated function's `.tool_meta` attribute, or reconstructs it from `__name__`, `__doc__`, and `inspect.signature()` for undecorated functions.

Returns: `{input_key, output_key, schema_model, tool_name, tool_doc, tool_signature}`.

---

### Prompt Builders

**`default_tool_prompt(tool, query_key="user_query") -> str`**

Generates a prompt instructing the LLM to return JSON with `{input_key}` containing the arguments for `tool`. Includes the tool name, docstring, and signature.

**`default_tool_summary_prompt(tool, query_key="user_query") -> str`**

Generates a prompt for analyzing the tool's output and answering the original user query. Instructs the LLM to return `{"answer": "..."}`.

**`wrap_tool_prompt(tool, prompt_template) -> str`**

If `{input_key}` is already in `prompt_template`, returns it unchanged. Otherwise wraps the template with tool name/doc/signature info and JSON output instructions.

**`wrap_tool_output(tool, prompt_template) -> str`**

If `{output_key}` is already in `prompt_template`, returns it unchanged. Otherwise prepends a section describing the tool's output before the original instructions.

---

### Node Factories

**`create_tool_node(tool, name, next_node_name=None) -> FunctionalNode`**

Wraps a `@tool_call`-decorated function as a `FunctionalNode`.

**`create_tool_llm_node(tool, response_fn, name, next_node_name=None, prompt_template=None, query_key="user_query", max_history_pairs=10, temperature=0.1) -> FunctionalNode`**

Creates an LLM node that generates the argument dict for `tool`. Automatically uses `default_tool_prompt` or `wrap_tool_prompt` depending on whether a custom template is provided. Temperature defaults to `0.1` for more deterministic JSON output.

**`create_tool_analysis_node(tool, response_fn, name, next_node_name=None, query_key="user_query", prompt_template=None, max_history_pairs=10, temperature=None, max_tokens=None) -> FunctionalNode`**

Creates an LLM node that interprets the tool's output and answers the user's query.

---

### Pair Factories

All pair factories return `dict[str, FunctionalNode | ConditionalNode]`, compatible with `GraphRunner.build(node_dicts=[...])`.

**`create_tool_llm_pair(tool, response_fn, llm_node_name, tool_node_name, prompt_template=None, tool_node_next_node_name=None, query_key="user_query", max_history_pairs=10, temperature=0.1) -> dict`**

Creates an LLM arg-generator node and a tool execution node, wired LLM → tool. Returns `{llm_node_name: llm_node, tool_node_name: tool_node}`.

---

### Parse-Error Retry System

When the LLM returns malformed JSON, these nodes form a retry loop back into the LLM.

**`create_retry_parse_error_pair(tool, response_fn, retry_node_name, conditional_node_name, tool_node, prompt_template=None, query_key="user_query", max_history_pairs=10, temperature=0.1) -> dict`**

Creates two nodes:
1. A `ConditionalNode` (named `conditional_node_name`) that checks `state["parse_error"]` — routes to the retry LLM if `True`, or to `tool_node` if `False`.
2. A retry LLM node (named `retry_node_name`) that includes the raw bad output and the parse error message in its prompt, then loops back to the conditional.

Returns `{conditional_node.name: conditional_node, retry_node.name: retry_node}`.

---

### Tool-Error Retry System

When the tool execution fails (raises an exception or returns `{output_key}_success == False`), these nodes retry argument generation.

**`create_retry_tool_error_pair(tool, response_fn, tool_analysis_node, retry_tool_name, check_tool_name, check_parse_name, prompt_template=None, query_key="user_query", max_history_pairs=10, temperature=0.1) -> dict`**

Creates two nodes:
1. A `ConditionalNode` (named `check_tool_name`) that checks `state["{output_key}_success"]` — routes to `tool_analysis_node` on success, or to the retry LLM on failure.
2. A retry LLM node (named `retry_tool_name`) that includes the tool error and the attempted args in its prompt, then routes to `check_parse_name` (the parse-error conditional) so parse errors are also caught on the retry path.

Returns `{conditional_node.name: conditional_node, retry_node.name: retry_node}`.

---

### Usage

#### Simple tool + LLM pair (no retry)

```python
from llm_graph.utils import tool_call
from llm_graph.factories.tool import create_tool_llm_pair, create_tool_analysis_node
from llm_graph.core.graphrunner import GraphRunner

@tool_call(input_key="search_args", output_key="search_results")
def web_search(query: str) -> str:
    """Search the web for the given query and return results."""
    ...  # your implementation

analysis_node = create_tool_analysis_node(
    tool=web_search,
    response_fn=response_fn,
    name="analysis",
)

tool_pair = create_tool_llm_pair(
    tool=web_search,
    response_fn=response_fn,
    llm_node_name="llm",
    tool_node_name="search",
    tool_node_next_node_name="analysis",
)

runner = GraphRunner.build(
    node_dicts=[tool_pair, {"analysis": analysis_node}],
    start_node="llm",
)
output = runner.execute({"user_query": "What is the latest news on AI?"})
```

#### Full workflow with parse-error and tool-error retry

```python
from llm_graph.factories.tool import (
    create_tool_node,
    create_tool_llm_node,
    create_tool_analysis_node,
    create_retry_parse_error_pair,
    create_retry_tool_error_pair,
)
from llm_graph.core.graphrunner import GraphRunner

@tool_call(input_key="calc_args", output_key="calc_result")
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression and return the result."""
    return eval(expression)

# Terminal nodes
tool_node = create_tool_node(tool=calculate, name="calc", next_node_name="check_tool")
analysis_node = create_tool_analysis_node(tool=calculate, response_fn=response_fn, name="analysis")

# LLM that generates args (next node set to parse-error conditional)
llm_node = create_tool_llm_node(
    tool=calculate,
    response_fn=response_fn,
    name="llm",
    next_node_name="check_parse",
)

# Parse-error retry loop
parse_pair = create_retry_parse_error_pair(
    tool=calculate,
    response_fn=response_fn,
    retry_node_name="retry_parse",
    conditional_node_name="check_parse",
    tool_node=tool_node,
)

# Tool-error retry loop
tool_retry_pair = create_retry_tool_error_pair(
    tool=calculate,
    response_fn=response_fn,
    tool_analysis_node=analysis_node,
    retry_tool_name="retry_tool",
    check_tool_name="check_tool",
    check_parse_name="check_parse",
)

runner = GraphRunner.build(
    node_dicts=[
        {"llm": llm_node},
        parse_pair,
        {"calc": tool_node},
        tool_retry_pair,
        {"analysis": analysis_node},
    ],
    start_node="llm",
    max_node_visits=5,
    on_max_visits="exit",
)

output = runner.execute({"user_query": "What is 42 * 17?"})
print(output["state_dict"]["answer"])
```

Notes:
- Set `max_node_visits` on `GraphRunner` to prevent infinite retry loops.
- The retry nodes use `temperature=0.1` by default for more consistent JSON output.
- All pair factories return dicts directly passable to `GraphRunner.build(node_dicts=[...])`.
