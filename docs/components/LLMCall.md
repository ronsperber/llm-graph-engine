Component: LLMCall

Purpose: To create a callable that takes in the state, passes a prompt to an LLM and returns back a dict based on the output of the LLM. It is intended to fit into a FunctionalNode as `func`.

Attributes
- response_fn
- prompt_template (optional)
- max_history_pairs
- query_key

No exposed methods, just a call method

Usage
```python
from openai import OpenAI
from llm_graph.llm.llm_call import LLMCall
from llm_graph.llm.response_functions import OpenAI_response_fn
client = OpenAI()
response_fn = OpenAI_response_fn(client=client)
prompt_template = 
"""
You are to find an answer to the query below and give your response in JSON format
with key 'answer'

{user_query}
"""
llm = LLMCall(
    response_fn=response_fn,
    prompt_template=prompt_template,
    query_key="user_query"
)
```
