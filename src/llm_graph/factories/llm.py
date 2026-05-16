from llm_graph.utils import ResponseFn
from llm_graph.core.nodes import FunctionalNode
from llm_graph.llm.llm_call import LLMCall, PromptTemplate


def default_llm_prompt(query_key: str) -> str:
    """
    create default prompt for an LLM based on user query
    Parameter
    ---------
    query_key : str
        key that has user query
    Returns
    -------
    str
        default prompt template
    """

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
    prompt_template: PromptTemplate = None,
    query_key: str = "user_query",
    next_node_name: str | None = None,
    max_history_pairs: int = 10,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> FunctionalNode:
    """
    Create functional node for sending query to LLM
    Parameters
    ----------
    response_fn : ResponseFn
        response function for the LLM
    name : str
        name of the node
    prompt_template : PromptTemplate
        prompt template for LLM
    query_key : str
        key that has user query
    next_node_name : Optional[str]
        name of next node, if any
    max_history_pairs : int
        number of query/response pairs to keep in history
    temperature : Optional[float]
        temperature to use for LLM
    max_tokens: Optional[int]
        limit on tokens used by LLM
    Returns
    -------
    FunctionalNode
        functional node that has the appropriate LLMCall function
    """
    # generate default prompt template if none is given
    if prompt_template is None:
        prompt_template = default_llm_prompt(query_key)
    # create callable to be used for node
    llm = LLMCall(
        response_fn=response_fn,
        prompt_template=prompt_template,
        max_history_pairs=max_history_pairs,
        query_key=query_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return FunctionalNode(func=llm, name=name, next_node_name=next_node_name)
