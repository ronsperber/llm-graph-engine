import streamlit as st
from dotenv import load_dotenv
import logging
from openai import OpenAI, APITimeoutError
from typing import Any
import os
import warnings


from llm_graph.llm.response_functions import OpenAI_response_fn
from llm_graph.factories.llm import create_llm_node
from llm_graph.factories.rag import create_rag_query_pair
from llm_graph.core.nodes import ConditionalNode
from llm_graph.core.graphrunner import GraphRunner
from llm_graph.core.sessionrunner import SessionRunner

# filter out any warnings from transformers 
warnings.filterwarnings("ignore", module="transformers")
# create the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)
# load environment variables
load_dotenv()

def get_secret(key:str) -> str:
    """
    used to get API keys
    For streamlit cloud it uses the secret
    otherwise, uses the environment variable
    """
    try:
        return st.secrets.get(key, "") or os.environ.get(key, "")
    except Exception:
        return os.environ.get(key, "")

DB_PATH = "vector_db"
COLLECTION = "llm_graph_docs"
MAX_NODE_VISITS = 5
MAX_ATTEMPTS = 5

branching_prompt = """
You are making a decision on the type of query given.
This will either be a coding query (asking for code examples, implementation details, usage snippets)
or a general query (asking for conceptual explanations, descriptions, or comparisons).

Respond ONLY with valid JSON with key 'query_type' set to either 'coding' or 'general'.

The query is: {user_query}
"""

coding_prompt = """
You are an expert on the llm-graph-engine system.
A user has asked a coding question about this system.
Use the retrieved documentation to give a clear, complete code example that answers the query.
Include imports where relevant. Do not go beyond what is asked.

Respond ONLY with valid JSON with key 'answer' containing your response.

The query is: {user_query}
"""

general_prompt = """
You are an expert on the llm-graph-engine system.
A user has asked a general question about this system.
Use the retrieved documentation to give a clear conceptual explanation.
Your response should be in markdown, using bullet points, numbered lists, and headers as appropriate.
No sample code should be generated for answers to general questions.
Respond ONLY with valid JSON with key 'answer' containing your response.
 
The query is: {user_query}
"""

def check_branching_node(state: dict[str,Any]) -> str:
    if state.get("parse_error", False):
        return "branch_llm"
    return state.get("query_type","general")


def build_session(use_openAI=False) -> SessionRunner:
    if use_openAI:
        logger.info("Building session with OpenAI")
        client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))
        branch_response_fn = response_fn = OpenAI_response_fn(client)
    else:
        logger.info("Building session witn Nvidia (Llama/Qwen)")
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=get_secret("NVIDIA_API_KEY"),
            timeout=30.0,
            )
        response_fn = OpenAI_response_fn(client, model="qwen/qwen3-coder-480b-a35b-instruct")
        branch_response_fn = OpenAI_response_fn(
            client=client,
            model="meta/llama-3.3-70b-instruct"
            )    

    branch_node = create_llm_node(
        response_fn=branch_response_fn,
        name="branch_llm",
        prompt_template=branching_prompt,
        next_node_name="check_type",
    )

    check_type_node = ConditionalNode(
        name="check_type",
        condition_fn=check_branching_node,
    )

    coding_nodes = create_rag_query_pair(
        path=DB_PATH,
        collection_name=COLLECTION,
        response_fn=response_fn,
        retrieval_node_name="coding",
        llm_node_name="coding_llm",
        prompt_template=coding_prompt,
    )

    general_nodes = create_rag_query_pair(
        path=DB_PATH,
        collection_name=COLLECTION,
        response_fn=response_fn,
        retrieval_node_name="general",
        llm_node_name="general_llm",
        prompt_template=general_prompt,
    )

    graphrunner = GraphRunner.build(
        node_dicts=[
            {"branch_llm": branch_node, "check_type": check_type_node},
            coding_nodes,
            general_nodes,
        ],
        start_node="branch_llm",
        max_node_visits=MAX_NODE_VISITS,
    )

    return SessionRunner(graph=graphrunner, session_keys=["message_history"])


SUGGESTED_QUESTIONS = {
    "General": [
        "How does GraphRunner.build() work and when would I use it instead of the regular constructor?",
        "What is the difference between GraphRunner and SessionRunner?",
        "What are the parse-error and tool-error retry systems in the tool factory, and how do they relate to each other?",
        "What does a ConditionalNode do and when would I use one?",
    ],
    "Coding": [
        "How do I use create_rag_query_pair to build a RAG workflow?",
        "Show me how to use the @tool_call decorator with Pydantic validation.",
        "How do I create an LLM node with a custom prompt template?",
        "Can you show me a complete tool-calling workflow with retry using the factory functions?",
    ],
}

# --- Initialise session state ---
if "db_warmed" not in st.session_state:
    import chromadb
    with st.spinner("Loading..."):
        db_client = chromadb.PersistentClient(path=DB_PATH)
        collection = db_client.get_collection(COLLECTION)
        collection.query(query_texts=["warmup"], n_results=1)
        st.session_state.db_warmed = True
        logger.info("Database warmed up")

if "session_runner" not in st.session_state:
    st.session_state.session_runner = build_session()

if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role": "user"|"assistant", "content": str, "branch": str|None}

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# --- Sidebar ---

with st.sidebar:
    st.header("Suggested questions")
    for category, questions in SUGGESTED_QUESTIONS.items():
        st.subheader(category)
        for question in questions:
            if st.button(question, key=question, use_container_width=True):
                st.session_state.pending_question = question

# --- UI ---

st.title("llm-graph-engine")
st.caption("Ask anything about how this system works. Coding questions get a code-focused answer; general questions get a conceptual explanation.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("branch"):
            st.caption(f"routed to: **{msg['branch']}** branch")
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about llm-graph-engine...")

if st.session_state.pending_question:
    prompt = st.session_state.pending_question
    logger.info(f"User query: {prompt}")
    st.session_state.pending_question = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "branch": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = None
            for attempt in range(MAX_ATTEMPTS):
                try:
                    result = st.session_state["session_runner"].execute({"user_query": prompt})
                    break
                except APITimeoutError:
                    if attempt == MAX_ATTEMPTS - 1:
                        st.error("Request timed out after multiple attempts.")
                        logger.error("Request timed out")
                    else:
                        logger.error(f"Request timed out after {MAX_ATTEMPTS} attempts for query: {prompt}")
                        st.warning(f"Timeout, retrying... ({attempt + 2}/{MAX_ATTEMPTS})")
        answer = "Result timed out. Please try again"
        if result is not None:
            state:dict[str, Any] = result["state_dict"]
            answer = state.get("answer") or state.get("raw_output", "No answer returned.")

        # Determine which branch was taken from the trace
        branch = None
        if result is not None:
            for step in result.get("trace_log", []):
                if step["name"] in ("coding_llm", "general_llm"):
                    branch = "coding" if step["name"] == "coding_llm" else "general"
                    break

        if branch:
            st.caption(f"routed to: **{branch}** branch")
            logger.info(f"Query routed to {branch}")
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer, "branch": branch})
