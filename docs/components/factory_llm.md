## Component: LLM Factory (`llm_graph.factories.llm`)

The LLM factory provides a convenience function for creating `FunctionalNode` instances backed by an `LLMCall`, reducing boilerplate for the common case of a single LLM step in a graph.

### Functions

**`create_llm_node(response_fn, name, prompt_template=None, query_key="user_query", next_node_name=None, max_history_pairs=10, temperature=None, max_tokens=None) -> FunctionalNode`**

Creates a `FunctionalNode` wrapping an `LLMCall`.

- `response_fn`: callable matching the `ResponseFn` protocol.
- `name`: node name.
- `prompt_template` (optional): string or callable prompt template. If omitted, a default template is generated that instructs the LLM to return `{"answer": "..."}`.
- `query_key`: state key for the raw user query. Defaults to `"user_query"`.
- `next_node_name` (optional): name of the next node.
- `max_history_pairs`: number of user/assistant pairs to retain. Defaults to 10.
- `temperature`, `max_tokens` (optional): passed through to the response function.

**`default_llm_prompt(query_key) -> str`**

Returns a generic prompt template string that instructs the LLM to answer the query at `{query_key}` and return JSON with key `"answer"`. Used automatically by `create_llm_node` when no `prompt_template` is provided.

### Usage

```python
from openai import OpenAI
from llm_graph.llm.response_functions import OpenAI_response_fn
from llm_graph.factories.llm import create_llm_node
from llm_graph.core.graphrunner import GraphRunner

client = OpenAI()
response_fn = OpenAI_response_fn(client=client)

llm_node = create_llm_node(
    response_fn=response_fn,
    name="llm",
    prompt_template="Answer the following question in JSON with key 'answer': {user_query}",
)

runner = GraphRunner(nodes=[llm_node], start_node="llm")
output = runner.execute({"user_query": "What is the capital of France?"})
```
