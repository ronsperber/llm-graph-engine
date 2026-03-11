from typing import Callable, Any
from openai.types.chat import ChatCompletionMessageParam
from llm_graph.core.nodes import FunctionalNode
from llm_graph.tools import make_chroma_search_tool
from llm_graph.llm.llm_call import LLMCall


def default_llm_prompt(
    query_key = "user_query"
):
    prompt = f"Answer the user query : {{{query_key}}}"
    return prompt

def create_retrieval_node(
    path: str,
    collection_name: str,
    query_key: str = "user_query",
    name: str = "retrieval",
    next_node_name: str | None = None
):
    tool = make_chroma_search_tool(path=path, collection_name=collection_name)
    def func(state: dict):
        query = state.get(query_key, "")
        ret_key = {"retrieval_args" : {"query": query}}
        resp = tool(ret_key)
        if resp['retrieved_results_success']:
                docs = resp["retrieved_results"].get("documents", [[]])[0]
                combined = "\n\n".join(docs) if docs else ""
        else:
            combined = ""
        return resp | {"retrieved_context": combined}
    node = FunctionalNode(func=func, name=name, next_node_name=next_node_name)
    return node

def wrap_rag_prompt(prompt_template: str) -> str:
    if "{retrieved_context}" in prompt_template:
        return prompt_template
    rag_wrapper = """
    Use the following retrieved context to answer the question.
    Context :
    {{retrieved_context}}

    ---------------------

    {prompt_template}
    """

    return rag_wrapper.format(prompt_template=prompt_template.strip())

def create_rag_llm_node(
    response_fn :Callable[[list[ChatCompletionMessageParam]], dict[str,Any]],
    name : str,
    next_node_name:str | None = None,
    query_key: str = "user_query",
    prompt_template: str | None = None,
    max_history_pairs: int = 10
):
    if prompt_template is None:
        prompt_template = default_llm_prompt(query_key)
    rag_prompt = wrap_rag_prompt(prompt_template)
    llm = LLMCall(
        response_fn=response_fn,
        prompt_template=rag_prompt,
        max_history_pairs=max_history_pairs,
        query_key=query_key
    )

    return FunctionalNode(
        func=llm,
        name=name,
        next_node_name=next_node_name
    )

def create_rag_query_pair(
    path: str,
    collection_name: str,
    response_fn: Callable[[list], dict[str, Any]],
    llm_node_name: str,
    retrieval_node_name : str = "retrieval",
    llm_next_node_name : str | None = None,
    query_key: str = "user_query",
    prompt_template: str | None = None,
    max_history_pairs: int = 10
)-> tuple[FunctionalNode, FunctionalNode]:
    retrieval_node = create_retrieval_node(
        path=path,
        collection_name=collection_name,
        query_key=query_key,
        name=retrieval_node_name,
        next_node_name=llm_node_name
    )

    llm_node = create_rag_llm_node(
        response_fn=response_fn,
        name=llm_node_name,
        next_node_name=llm_next_node_name,
        query_key=query_key,
        prompt_template=prompt_template,
        max_history_pairs=max_history_pairs
    )

    return retrieval_node, llm_node