## Component: tool_call

`tool_call` is a decorator that wraps a plain function so it can act as a `NodeFunc`. The wrapper reads arguments from the state dict, calls the function, and returns a result dict. It is the standard way to integrate external tools (databases, APIs, etc.) into a graph workflow.

### Parameters

`@tool_call(input_key, output_key, model=None)`

- `input_key`: state dict key whose value is a `dict` of keyword arguments to pass to the function.
- `output_key`: state dict key for the function's return value in the output delta.
- `model` (optional): a Pydantic `BaseModel` subclass. If provided, the arguments dict is validated against the model before calling the function. Validation failure returns a failure result rather than raising.

### Output Keys Written

The decorated function returns a delta dict with three keys:
- `{output_key}`: the function's return value, or the error message string on failure.
- `{output_key}_success`: `True` if the call succeeded, `False` otherwise.
- `{output_key}_args`: the kwargs dict that was passed to the function.

### `.tool_meta` Attribute

The decorated function gets a `.tool_meta` dict attached, containing:
- `input_key`, `output_key`, `schema_model`, `tool_name`, `tool_doc`, `tool_signature`

This is used by the factory modules to auto-generate prompts and wire up retry nodes.

### Usage

```python
from llm_graph.utils import tool_call
from llm_graph.core.nodes import FunctionalNode
from llm_graph.core.graphrunner import GraphRunner

@tool_call(input_key="tool_params", output_key="tool_output")
def query_strip(query: str):
    return query.strip()

def tool_prep(state):
    query = state["user_query"]
    return {"tool_params": {"query": query}}

prep_node = FunctionalNode(name="prep", func=tool_prep, next_node_name="tool")
tool_node = FunctionalNode(name="tool", func=query_strip)

graph_runner = GraphRunner(nodes=[prep_node, tool_node], start_node="prep")
result = graph_runner.execute({"user_query": "    This needs to be stripped. "})

# result["state_dict"]["tool_output"] == "This needs to be stripped."
# result["state_dict"]["tool_output_success"] == True
```

### With Pydantic Validation

```python
from pydantic import BaseModel

class SearchArgs(BaseModel):
    query: str
    n_results: int = 3

@tool_call(input_key="search_params", output_key="search_results", model=SearchArgs)
def search_database(query: str, n_results: int = 3):
    ...
```
