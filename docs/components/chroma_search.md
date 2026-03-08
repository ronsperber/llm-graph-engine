## Component : make_chroma_search_tool

make_chroma_search_tool is the the function used to create a function to search in ChromaDB vector database for things similar to the query. It will create a tool that can be used in a FunctionalNode.

### Parameters

There are two parameters, `path`, which is the path to the database and `collection_name` which is the name of the collection in the database. The result of `make_chroma_search_tool(path=path, collection_name=collection_name)` is a callable that can be inserted into a FunctionalNode. It will look for the key `retrieval_args` in the state that should be 'query':query and optionally 'n_results':n_results. 

### Usage

```python
from llm_graph.tools import make_chroma_search_tool
from llm_graph.core.nodes import FunctionalNode
from llm_graph.core.graphrunner import GraphRunner

path = "vector_db"
collection_name = "llm_graph_docs"
chroma_tool = make_chroma_search_tool(path=path, collection_name=collection_name)
def prep_for_retrieval(state):
    # create retrieval_args to be available
    query = state['user_query']
    return {'retrieval_args': {'query' : query}}

prep_node = FunctionalNode(
    func=prep_for_retrieval,
    name="start",
    next_node_name="retriever"
)

query_node = FunctionalNode(
    func=chroma_tool,
    name="retriever"
)

graph_runner = GraphRunner(nodes=[prep_node, query_node], start_node="start")
graph_runner.execute({"user_query": "What does a FunctionalNode do?"})
```