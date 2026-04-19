### Component: GraphRunner

Purpose:
GraphRunner manages execution of workflow graphs by controlling node traversal
and maintaining runtime state. It also tracks token usage and enforces optional node visit limits. It does not maintain state_dict across multiple calls.

Key Responsibilities:
- Executes nodes sequentially
- Maintains state dictionary across workflow execution
- Records execution trace logs

State Model:
- Nodes receive the full state_dict.
- Nodes return partial state updates.
- GraphRunner merges updates into state_dict.

### Constructor

`GraphRunner(nodes, start_node, max_node_visits=None, on_max_visits="exit")`

- `nodes`: list or dict of `GraphNode` instances. Dict keys must match each node's `name`.
- `start_node`: name of the node to begin execution.
- `max_node_visits` (optional): integer cap on how many times any single node may be visited per `execute()` call. `None` means unlimited.
- `on_max_visits` (optional): `"error"` raises `RuntimeError` when the cap is hit; `"exit"` (default) stops execution early and sets `terminated_early=True`.

### Attributes
- `nodes_dict`: dict mapping node names to node objects
- `state_dict`: accumulated state after execution
- `trace_log`: list of step dicts recorded during execution
- `start_node`: name of the starting node
- `in_tokens`: prompt tokens accumulated in last `execute()` call
- `out_tokens`: completion tokens accumulated in last `execute()` call
- `node_tracker`: visit count per node name (reset each `execute()`)

### Methods

**`execute(input, reset_trace=True) -> dict`**

Runs the graph from `start_node` with the provided input dict. Returns a dict with:
- `"state_dict"`: final accumulated state
- `"trace_log"`: list of step records (keys: `step_num`, `name`, `node_input`, `node_output`, `node_type`, `next_node_name`, `execution_time`, `timestamp`)
- `"token_usage"`: `{"in_tokens": N, "out_tokens": M, "total_tokens": N+M}`
- `"terminated_early"`: `bool`

Token tracking: if a node returns `{"usage": {"prompt_tokens": N, "completion_tokens": M}}`, those values are accumulated automatically.

**`get_token_usage() -> dict`**
Returns `{"in_tokens": N, "out_tokens": M, "total_tokens": N+M}`.

**`print_trace()`**
Pretty-prints the execution trace log.

**`matplotlib_trace(branch_color_map=None)`**
Draws the execution trace as a vertical box-and-arrow diagram. `branch_color_map` maps next-node names to arrow colors.

**`GraphRunner.build(node_dicts, start_node, connections=None, max_node_visits=None, on_max_visits="exit") -> GraphRunner`** (classmethod)

High-level assembly method. Merges multiple node dicts (as returned by factory functions) into a single `GraphRunner`.

- `node_dicts`: iterable of `dict[str, GraphNode]` — one dict per factory-created group.
- `start_node`: name of the starting node.
- `connections` (optional): `dict[str, str]` for explicit cross-group wiring, e.g. `{"source_node": "target_node"}`. Both source and target must exist or `ValueError` is raised.
- Raises `ValueError` on duplicate node names across dicts.

### Usage

```python
from llm_graph.core.graphrunner import GraphRunner

runner = GraphRunner(
    nodes=nodelist,
    start_node="start",
)

state = {"user_query": "What is the capital of France?"}
output = runner.execute(state)
final_state = output["state_dict"]
token_usage = output["token_usage"]
```

Using `build()` with factory-produced node dicts:

```python
from llm_graph.core.graphrunner import GraphRunner
from llm_graph.factories.rag import create_rag_query_pair

rag_nodes = create_rag_query_pair(
    path="vector_db",
    collection_name="my_docs",
    response_fn=response_fn,
    llm_node_name="llm",
)

runner = GraphRunner.build(
    node_dicts=[rag_nodes],
    start_node="retrieval",
)
```

Notes:
- GraphRunner does not depend on UI frameworks.
- State is not persisted between `execute()` calls; use `SessionRunner` for that.
