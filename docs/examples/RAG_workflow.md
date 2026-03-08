Example workflow : Using a ChromaDB database for RAG to get results.

This example demonstrates how to build a workflow graph to use an existing ChromaDB database to inject context into a prompt to get better results.

The workflow consists of the following nodes:
1 Prep node : creates a key that can be used for the tool call necessary.

2. Retrieval node : gets the top 3 results in the database for context later

3. Combiner node : takes the top 3 results and stitches them together for one string to inject into the prompt

4. LLM node : creates a prompt that includes the result of the retrieved data to get the response

Flow:
prep  → retrieval node → combiner node → llm node

```python
from openai import OpenAI
from llm_graph.llm.response_functions import OpenAI_response_fn
from llm_graph.llm.llm_call import LLMCall
from llm_graph.tools import make_chroma_search_tool
from llm_graph.core.graphrunner import GraphRunner
from llm_graph.core.nodes import FunctionalNode

prompt_template = """ 
You are an expert at graph based workflows for large language models. You are answering questions about
a specific one that at the to level has a SessionRunner class. This class is responsible for maintaining
state keys between user queries. This invokes the GraphRunner class. The GraphRunner class holds a state_dict
and executes the nodes sequentially. Each node is a base class of GraphNode that holds its name of the node,
the name of the next node and an execute method intended to return a delta to the state_dict. 

Functional Nodes have a func that is Callable[dict, dict]. When executed they get passed the state_dict,
and return a delta. After that the state_dict updates via that delta. A ConditionalNode just has a
conditional_func that determines the name of the next_node. Nodes should not mutate state_dict directly.
There is a @tool_call decorator for tools, and an LLMCall class for handling a func to send something to 
an LLM.

some context for the query below is given by : {injected_retrieval}

Your answer to query should be in JSON format with key 'answer'

The query is {user_query}
"""

def prep_retrieval(state):
    query = state['user_query']
    return {'retrieval_args' : {'query' : query}}

def combine_results(state):
    results = state['retrieved_results']
    docs = results['documents'][0]
    return {"injected_retrieval" : "\n\n".join(docs)}

path = "vector_db"
collection_name = "llm_graph_docs"
chroma_tool = make_chroma_search_tool(path=path, collection_name=collection_name)

client = OpenAI()
response_fn = OpenAI_response_fn(client=client)
llm = LLMCall(response_fn=response_fn, prompt_template=prompt_template)

prep_node = FunctionalNode(
    func=prep_retrieval,
    name="start",
    next_node_name="retrieve"
    )

retrieval_node = FunctionalNode(
    func=chroma_tool,
    name="retrieve",
    next_node_name="combine"
)

combine_node = FunctionalNode(
    func=combine_results,
    name="combine",
    next_node_name="llm"
)

llm_node = FunctionalNode(
    func=llm,
    name="llm"
)

graph_runner = GraphRunner(
    nodes = [prep_node, retrieval_node, combine_node, llm_node],
    start_node = "start"
)

graph_runner.execute({"user_query": "How do you implement RAG in GraphRunner?"})
```