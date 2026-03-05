Component: FunctionalNode

Purpose: FunctionalNode is the main class to execute functions while executing the workflow in a graph created by GraphRunner. It takes the current state of the workflow and uses a function to output a state delta.

Attributes
- func 
- name 
- last_input
- last_output
- next_node_name

Method
- execute(state)

example:

```python
 from llm_graph.core.nodes import FunctionalNode
def simple_func(state : dict) -> dict:
    return {"lower_case" : state["user_query"].lower()}

lowernode = FunctionalNode(
    func=simple_func,
    name="simple_node"
    next_node_name="use_lower" 
)
```