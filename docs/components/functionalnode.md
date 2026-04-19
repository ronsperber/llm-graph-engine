## Component: FunctionalNode

Purpose: FunctionalNode is the main class to execute functions while executing the workflow in a graph created by GraphRunner. It takes the current state of the workflow and uses a function to output a state delta.

### Constructor

`FunctionalNode(func, name, next_node_name=None)`

- `func`: `Callable[[dict], dict]` — receives the full state dict and returns a delta dict. Raises `TypeError` if the return value is not a dict.
- `name`: identifier for this node.
- `next_node_name` (optional): name of the next node to execute. `None` means this is a terminal node.

### Attributes
- `func`
- `name`
- `next_node_name`
- `last_input`: deep copy of state passed to the last `execute()` call
- `last_output`: deep copy of delta returned from the last `execute()` call

### Method
- `execute(state)` — calls `func(state)`, validates the return type, and stores copies of input/output.

### Usage:

```python
from llm_graph.core.nodes import FunctionalNode

def simple_func(state: dict) -> dict:
    return {"lower_case": state["user_query"].lower()}

lowernode = FunctionalNode(
    func=simple_func,
    name="simple_node",
    next_node_name="use_lower"
)
```
