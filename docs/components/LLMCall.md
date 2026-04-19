## Component: LLMCall

Purpose: A callable class that takes the graph state, formats a prompt, calls an LLM via a `response_fn`, and returns a state delta with the parsed response. Intended to be used as `func` in a `FunctionalNode`.

### Constructor

`LLMCall(response_fn, prompt_template=None, max_history_pairs=10, query_key="user_query", temperature=None, max_tokens=None)`

- `response_fn`: callable matching the `ResponseFn` protocol — takes `history` and optional kwargs, returns `{"raw_output": str, "usage": dict}`.
- `prompt_template` (optional): a format string (uses `**state` for substitution) or a callable `(state) -> str`. If omitted, `state[query_key]` is used directly as the prompt.
- `max_history_pairs`: number of user/assistant pairs to retain in `message_history`. Defaults to 10.
- `query_key`: state key containing the raw user query. Defaults to `"user_query"`.
- `temperature` (optional): passed through to `response_fn`.
- `max_tokens` (optional): passed through to `response_fn`.

### State Keys Read
- `message_history` (optional): list of prior chat messages; defaults to `[]` if absent.
- `{query_key}`: the user query, used when no `prompt_template` is set.
- Any keys referenced in a string `prompt_template` via `{key}` placeholders.

### State Keys Written
- `message_history`: updated list of messages including the new exchange.
- `raw_output`: the LLM's raw string response.
- `usage`: token usage dict from the response function.
- `parse_error`: `False` if the response was valid JSON, `True` otherwise.
- Any keys present in the parsed JSON (e.g. `"answer"` if the LLM returned `{"answer": "..."}`).
- `parse_error_message`: set when `parse_error` is `True`.

### Methods

**`__call__(state) -> dict`** — executes the LLM call and returns the state delta.

**`preview_prompt(state) -> str`** — returns the formatted prompt string without making an LLM call. Useful for debugging prompt templates.

### Usage

```python
from openai import OpenAI
from llm_graph.llm.llm_call import LLMCall
from llm_graph.llm.response_functions import OpenAI_response_fn

client = OpenAI()
response_fn = OpenAI_response_fn(client=client)

prompt_template = """
You are to find an answer to the query below and give your response in JSON format
with key 'answer'

{user_query}
"""

llm = LLMCall(
    response_fn=response_fn,
    prompt_template=prompt_template,
    query_key="user_query"
)

# Preview the formatted prompt before running
print(llm.preview_prompt({"user_query": "What is the capital of France?"}))
```
