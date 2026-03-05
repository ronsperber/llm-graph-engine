Component: ConditionalNode

Purpose: Node used in the workflow to decide which node to go to next based on a conditional function

Attributes
- condition_fn
- name

Method
- execute(state)

example:

```python
from llm_graph.core.nodes import ConditionalNode
def conditional_func(state):
    query_type = state["query_type"]
    return query_type

branchnode = ConditionalNode(
    condition_fn=conditional_func
    name="branch"
)
```