Example Workflow: Conditional Branching Graph

This example demonstrates how to build a workflow graph with a conditional branch.

The workflow consists of:

1. A routing LLM node that determines the query type
2. A ConditionalNode that selects the next node based on the routing output
3. Two downstream nodes that handle different query types

Flow:

start → conditional_node → {coding | general}

Graph Structure

start (LLM routing)
   ↓
conditional_node
   ├── coding
   └── general

```python
from openai import OpenAI
from llm_graph.core.nodes import FunctionalNode, ConditionalNode
from llm_graph.core.graphrunner import GraphRunner
from llm_graph.core.sessionrunner import SessionRunner
from llm_graph.llm.llm_call import LLMCall
from llm_graph.llm.response_functions import OpenAI_response_fn

client = OpenAI()
response_fn = OpenAI_response_fn(client=client)
branching_prompt = """
You are making a decision on the type of query given. This will either be a coding query or a general query. Respond with a JSON format response with key 'query_type' followed by coding or general

query is {user_query}
"""
coding_prompt = """
You are an expert at coding. You have been asked a question about how to write code to solve the query below. Give your answer using the programming language specified. If no coding language is specified, use Python. Make sure your code is complete, solves the problem, and has no syntax errors. You can use any publically available libraries to import if it helps solve the problem. Do not go beyond what is asked other than using normal good coding practices. Give your response in JSON format with the key 'answer'

The query is {user_query}
"""

general_prompt ="""
You are answering a general knowledge question. Do your best to give an accurate response. Format your response in JSON format with key 'answer'.

The query is {user_query}
"""

branching_llm = LLMCall(
    response_fn=response_fn,
    prompt_template=branching_prompt,
    query_key="user_query"
)

coding_llm = LLMCall(
    response_fn=response_fn,
    prompt_template=coding_prompt,
    query_key="user_query"
)

general_llm = LLMCall(
    response_fn=response_fn,
    prompt_template=general_prompt,
    query_key="user_query"
)

def conditional_fn(state):
    return state['query_type'] # query type here is coding or general

start_node = FunctionalNode(
    func=branching_llm,
    name="start",
    next_node_name="conditional_node"
)

conditional_node=ConditionalNode(
    condition_fn=conditional_fn,
    name="conditional_node"
)

coding_node = FunctionalNode(
    func=coding_llm,
    name="coding"
)

general_node = FunctionalNode(
    func=general_llm,
    name="general"
)

graphrunner = GraphRunner(
    nodes = [start_node, conditional_node, coding_node, general_node],
    start_node = "start"
)
runner = SessionRunner(
    graphrunner=graphrunner,
    session_keys=["message_history"]
)
response = runner.execute(
    {"user_query": "How do I implement quicksort in Scala?"}
)
```