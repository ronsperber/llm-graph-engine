# llm-graph-engine

A lightweight Python framework for building stateful LLM workflows as directed graphs. Nodes share a mutable state dictionary — each node receives the full state, computes a delta, and the runner merges it back. The library includes factory modules for common patterns (RAG, tool-calling, retry loops) that reduce a multi-node wiring task to a single function call.

Built as a learning project to explore how graph-based execution models simplify the design of multi-step LLM pipelines — particularly around retry logic, session state, and modular composition.

---

## What it does

- **Graph execution engine** — define nodes as plain Python callables, wire them by name, and let `GraphRunner` handle traversal, state accumulation, and token tracking.
- **Session management** — `SessionRunner` wraps a graph to persist chosen state keys (e.g. `message_history`) across multiple calls.
- **LLM integration** — `LLMCall` handles prompt formatting, message history trimming, and JSON parse attempts. Plug in any response function; an OpenAI wrapper is included.
- **RAG workflows** — `create_rag_query_pair` creates a full ChromaDB retrieval + LLM node pair with automatic context injection.
- **Tool-calling with retry** — factory functions build complete tool-calling workflows including conditional retry loops for both JSON parse failures and tool execution failures.
- **Execution tracing** — every run produces a structured trace log; `print_trace()` and `matplotlib_trace()` visualize the execution path.

---

## Project structure

```
src/llm_graph/
├── core/
│   ├── nodes.py            # GraphNode, FunctionalNode, ConditionalNode
│   ├── graphrunner.py      # GraphRunner — execution engine
│   └── sessionrunner.py    # SessionRunner — persistent state across runs
├── llm/
│   ├── llm_call.py         # LLMCall — prompt formatting + LLM invocation
│   └── response_functions.py  # OpenAI_response_fn, dummy_llm_response_fn
├── factories/
│   ├── llm.py              # create_llm_node
│   ├── rag.py              # create_rag_query_pair and friends
│   └── tool.py             # tool-calling + retry factory suite
└── utils.py                # @tool_call decorator, json_parse, ResponseFn protocol

scripts/
└── build_rag_index.py      # Populate a ChromaDB collection from docs/

example_notebooks/
└── rag_branching_example.ipynb   # RAG + conditional branching end-to-end

docs/
├── components/             # One doc per component/factory
└── examples/               # Annotated workflow examples
```

---

## Getting started

### Requirements

- Python 3.11+
- An OpenAI API key (for the example notebooks and OpenAI response function)

### Install

Clone the repo and install in editable mode. All runtime dependencies are declared in `pyproject.toml` and will be installed automatically:

```bash
git clone https://github.com/ronsperber/llm-graph-engine.git
cd llm-graph-engine
pip install -e .
```

To also install dev tools (pytest, jupyter, ipython):

```bash
pip install -e ".[dev]"
```

### API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

### Build the RAG index

The example notebook uses a ChromaDB vector store built from the `docs/` folder. Run this once from the project root to populate it:

```bash
python scripts/build_rag_index.py
```

This creates a `vector_db/` directory with an embedded index of the component documentation, using `all-MiniLM-L6-v2` as the embedding model (downloaded automatically on first run).

---

## Quick example

A minimal single-node LLM graph:

```python
from openai import OpenAI
from llm_graph.llm.response_functions import OpenAI_response_fn
from llm_graph.factories.llm import create_llm_node
from llm_graph.core.graphrunner import GraphRunner

client = OpenAI()
response_fn = OpenAI_response_fn(client)

llm_node = create_llm_node(
    response_fn=response_fn,
    name="llm",
    prompt_template='Answer in JSON with key "answer": {user_query}',
)

runner = GraphRunner(nodes=[llm_node], start_node="llm")
output = runner.execute({"user_query": "What is the capital of France?"})
print(output["state_dict"]["answer"])
```

A RAG workflow in a few lines using factories:

```python
from llm_graph.factories.rag import create_rag_query_pair
from llm_graph.core.graphrunner import GraphRunner

rag_nodes = create_rag_query_pair(
    path="vector_db",
    collection_name="llm_graph_docs",
    response_fn=response_fn,
    llm_node_name="llm",
)

runner = GraphRunner.build(node_dicts=[rag_nodes], start_node="retrieval")
output = runner.execute({"user_query": "How does GraphRunner.build() work?"})
```

See `example_notebooks/rag_branching_example.ipynb` for a complete walkthrough combining RAG, conditional branching, and session state.

---

## Demo app

`app.py` is a Streamlit chat interface built on the same RAG + conditional branching graph from the example notebook. It lets you ask questions about the system and see which branch (coding or general) the graph routed to for each answer.

Make sure the RAG index is built first (see above), then run:

```bash
streamlit run app.py
```

The sidebar contains suggested questions covering both branches. The graph persists `message_history` across turns so the LLM has full conversation context throughout the session.

---

## Running the tests

```bash
pytest
```

---

## Documentation

Component references and annotated workflow examples are in `docs/`. Start with:

- [docs/components/graphrunner.md](docs/components/graphrunner.md) — execution engine
- [docs/components/factory_rag.md](docs/components/factory_rag.md) — RAG factory
- [docs/components/factory_tool.md](docs/components/factory_tool.md) — tool-calling + retry factory
- [docs/examples/tool_factory_workflow.md](docs/examples/tool_factory_workflow.md) — end-to-end tool workflow
