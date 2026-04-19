## Component: RAG Factory (`llm_graph.factories.rag`)

The RAG factory builds ChromaDB retrieval and LLM nodes for Retrieval-Augmented Generation workflows. The primary entry point is `create_rag_query_pair`, which creates a wired retrieval+LLM pair in one call.

### Functions

---

**`create_retrieval_node(path, collection_name, query_key="user_query", name="retrieval", next_node_name=None) -> FunctionalNode`**

Creates a node that retrieves similar documents from a ChromaDB collection.

- `path`: filesystem path to the ChromaDB persistent store.
- `collection_name`: name of the ChromaDB collection to query.
- `query_key`: state key containing the search query. Defaults to `"user_query"`.
- `name`: node name. Defaults to `"retrieval"`.
- `next_node_name` (optional): name of the next node.

State keys written:
- `retrieved_results`: raw ChromaDB response dict.
- `retrieved_results_success`: `True` on success.
- `retrieved_results_args`: the args passed to the search call.
- `retrieved_context`: a single string joining all retrieved documents, separated by `"\n\nDOCUMENT:\n"`. This is the key to reference in a RAG prompt template.

---

**`create_rag_llm_node(response_fn, name, next_node_name=None, query_key="user_query", prompt_template=None, max_history_pairs=10) -> FunctionalNode`**

Creates an LLM node whose prompt is automatically wrapped with a RAG context preamble. The prompt will include the `{retrieved_context}` value from state before passing to the LLM.

- `response_fn`: callable matching the `ResponseFn` protocol.
- `name`: node name.
- `prompt_template` (optional): a custom prompt string. If `{retrieved_context}` is not already present, it is injected automatically via `wrap_rag_prompt`. If omitted, a default RAG prompt is used.
- Other parameters match `create_llm_node`.

Expects `retrieved_context` to already be in state (set by `create_retrieval_node`).

---

**`create_rag_query_pair(path, collection_name, response_fn, llm_node_name, retrieval_node_name="retrieval", llm_next_node_name=None, query_key="user_query", prompt_template=None, max_history_pairs=10) -> dict[str, FunctionalNode]`**

The main high-level factory. Creates a retrieval node and an RAG LLM node, wires retrieval → llm, and returns both as a single dict compatible with `GraphRunner.build()`.

- `path`, `collection_name`: ChromaDB location.
- `response_fn`: the LLM response function.
- `llm_node_name`: name for the LLM node.
- `retrieval_node_name`: name for the retrieval node. Defaults to `"retrieval"`.
- `llm_next_node_name` (optional): next node after the LLM node.
- `query_key`, `prompt_template`, `max_history_pairs`: forwarded to the LLM node.

Returns `{retrieval_node.name: retrieval_node, llm_node.name: llm_node}`.

---

**`wrap_rag_prompt(prompt_template) -> str`**

Helper that injects a `{retrieved_context}` preamble into a prompt string if one is not already present. Used internally by `create_rag_llm_node`.

---

### Usage

Minimal RAG workflow using `create_rag_query_pair` and `GraphRunner.build()`:

```python
from openai import OpenAI
from llm_graph.llm.response_functions import OpenAI_response_fn
from llm_graph.factories.rag import create_rag_query_pair
from llm_graph.core.graphrunner import GraphRunner

client = OpenAI()
response_fn = OpenAI_response_fn(client=client)

rag_nodes = create_rag_query_pair(
    path="vector_db",
    collection_name="my_docs",
    response_fn=response_fn,
    llm_node_name="llm",
)

runner = GraphRunner.build(
    node_dicts=[rag_nodes],
    start_node="retrieval",
)

output = runner.execute({"user_query": "How does GraphRunner work?"})
print(output["state_dict"]["answer"])
```

With a custom prompt template:

```python
custom_prompt = """
You are a helpful assistant. Use the provided context to answer the question.
Respond in JSON with key 'answer'.

The query is: {user_query}
"""

rag_nodes = create_rag_query_pair(
    path="vector_db",
    collection_name="my_docs",
    response_fn=response_fn,
    llm_node_name="llm",
    prompt_template=custom_prompt,  # {retrieved_context} will be injected automatically
)
```

Notes:
- The retrieval node's `query_key` and the LLM node's `query_key` both default to `"user_query"` and must match your state dict.
- Use `scripts/build_rag_index.py` to populate a ChromaDB collection from local documents.
