Component: GraphRunner

Purpose:
GraphRunner manages execution of workflow graphs by controlling node traversal
and maintaining runtime state. It also tracks token usage and enforces optional node visit limits. 

Key Responsibilities:
- Executes nodes sequentially
- Maintains state dictionary across workflow execution
- Records execution trace logs

State Model:
- Nodes receive the full state_dict.
- Nodes return partial state updates.
- GraphRunner merges updates into state_dict.

Attributes:
- state_dict
- trace_log
- nodes_dict
- start_node
- out_tokens
- in_tokens
- node_tracker

Methods:
- execute(state)
- clear_message_history()
- get_token_usage()
- print_trace()
- matplotlib_trace(Optional: branch_color_map)

Usage:
```python
from llm_graph.core.runner import GraphRunner
runner = GraphRunner(
    nodes = nodelist,
    start_node = 'start',
)

state = {'user_query':'What is the capital of France?'}
output = runner.execute(state)
final_state = output['state_dict']
```

Notes:
- GraphRunner does not depend on UI frameworks.

