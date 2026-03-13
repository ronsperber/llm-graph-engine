from llm_graph.utils import ResponseFn
from llm_graph.core.nodes import FunctionalNode
from llm_graph.llm.llm_call import LLMCall


def default_llm_prompt(query_key: str):

    return f"""
Answer the user query.

Return ONLY valid JSON in the following format:

{{{{
  'answer': "<your answer>"
}}}}

The user query is:
{{{query_key}}}
"""


def create_llm_node(
    response_fn: ResponseFn,
    name: str,
    prompt_template: str | None = None,
    query_key: str = "user_query",
    next_node_name: str | None = None,
    max_history_pairs: int = 10,
) -> FunctionalNode:
    if prompt_template is None:
        prompt_template = default_llm_prompt(query_key)
    llm = LLMCall(
        response_fn=response_fn,
        prompt_template=prompt_template,
        max_history_pairs=max_history_pairs,
        query_key=query_key,
    )

    return FunctionalNode(func=llm, name=name, next_node_name=next_node_name)
