## Component: ConditionalNode

Purpose: Node used in the workflow to decide which node to go to next based on a conditional function. Does not modify state.

### Constructor

`ConditionalNode(name, condition_fn)`

- `name`: identifier for this node.
- `condition_fn`: `Callable[[dict], str]` — receives the current state and returns the name of the next node to execute.

### Attributes
- `name`
- `condition_fn`
- `next_node_name`: set dynamically on each `execute()` call; always `None` at construction.

### Method
- `execute(state)` — calls `condition_fn(state)`, sets `next_node_name` to the result, returns `{}`.

### Usage:

```python
from llm_graph.core.nodes import ConditionalNode

def conditional_func(state):
    query_type = state["query_type"]
    return query_type

branchnode = ConditionalNode(
    condition_fn=conditional_func,
    name="branch"
)
```
