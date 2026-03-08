## Component : tool_call
tool_call is a decorator to wrap a function inside a function that takes in the state, passes the appropriate arguments to the function, and returns a dictionary with a key that has value equal to the output of the function. Doing any Retrieval Augmented Generation (RAG) uses tool_call

### Parameters
the decorator @tool_call takes two parameters : `input_key` and `output_key`. `input_key` should be a key in the `state_dict` whose value is a dict of the parameters to pass to the function. `output_key` is the name of key for the dict passed back as output


### Usage

```python
from llm_graph.utils import tool_call
from llm_graph.core.nodes import FunctionalNode
from llm_graph.core.graphrunner import GraphRunner

@tool_call(input_key="tool_params", output_key="tool_output")
def query_strip(query : str):
    return query.strip()

def tool_prep(state):
    # used to create the input_key for the tool calling node
    query = state["user_query"]
    return {"tool_params": {"query": query}}

prep_node = FunctionalNode(name="prep", func=tool_prep, next_node_name="tool")
tool_node = FunctionalNode(name="tool", func = query_strip)

graph_runner = GraphRunner(nodes=[prep_node, tool_node], start_node="prep")
graph_runner.execute({"user_query": "    This needs to be stripped. "})
```