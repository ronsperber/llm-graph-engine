Example Workflow: Tool-Calling with Retry Using Factories

This example demonstrates how to build a tool-calling workflow using the factory modules. The factories handle prompt generation, node wiring, and retry logic automatically. Compare this to the manual approach in `RAG_workflow.md` to see the reduction in boilerplate.

The workflow uses:
1. An LLM node to generate arguments for the tool
2. A parse-error conditional node (retries if the LLM returned malformed JSON)
3. A tool execution node
4. A tool-error conditional node (retries if the tool raised an error)
5. An analysis LLM node to interpret the result and answer the query

Flow:

```
llm → check_parse → calc → check_tool → analysis
         ↑                      |
     retry_parse            retry_tool
         ↑                      |
         └──────────────────────┘
```

```python
from openai import OpenAI
from llm_graph.llm.response_functions import OpenAI_response_fn
from llm_graph.utils import tool_call
from llm_graph.factories.tool import (
    create_tool_node,
    create_tool_llm_node,
    create_tool_analysis_node,
    create_retry_parse_error_pair,
    create_retry_tool_error_pair,
)
from llm_graph.core.graphrunner import GraphRunner

client = OpenAI()
response_fn = OpenAI_response_fn(client=client)


@tool_call(input_key="calc_args", output_key="calc_result")
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression and return the numeric result."""
    return eval(expression)


# Build individual nodes
tool_node = create_tool_node(
    tool=calculate,
    name="calc",
    next_node_name="check_tool",
)

analysis_node = create_tool_analysis_node(
    tool=calculate,
    response_fn=response_fn,
    name="analysis",
)

llm_node = create_tool_llm_node(
    tool=calculate,
    response_fn=response_fn,
    name="llm",
    next_node_name="check_parse",
)

# Parse-error retry: check_parse routes to calc on success or retry_parse on failure
parse_pair = create_retry_parse_error_pair(
    tool=calculate,
    response_fn=response_fn,
    retry_node_name="retry_parse",
    conditional_node_name="check_parse",
    tool_node=tool_node,
)

# Tool-error retry: check_tool routes to analysis on success or retry_tool on failure
tool_retry_pair = create_retry_tool_error_pair(
    tool=calculate,
    response_fn=response_fn,
    tool_analysis_node=analysis_node,
    retry_tool_name="retry_tool",
    check_tool_name="check_tool",
    check_parse_name="check_parse",
)

# Assemble with GraphRunner.build(); max_node_visits caps retry loops
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

output = runner.execute({"user_query": "What is 144 divided by 12?"})
print(output["state_dict"]["answer"])
# → "144 divided by 12 is 12."
```

Notes:
- `max_node_visits=5` prevents an infinite loop if the LLM repeatedly returns bad JSON or the tool keeps failing.
- The retry nodes use `temperature=0.1` by default for more deterministic JSON output.
- To add a custom prompt, pass `prompt_template` to `create_tool_llm_node`; the factory will inject the tool signature and JSON instructions automatically via `wrap_tool_prompt`.
