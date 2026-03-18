from inspect import signature
from typing import Callable, Any
from llm_graph.core.nodes import FunctionalNode, ConditionalNode, NodeFunc
from .llm import create_llm_node
from llm_graph.utils import ResponseFn


def get_tool_metadata(tool: NodeFunc) -> dict[str, Any]:
    """
    gets metadata from tool. If tool was not created with tool_meta
    attribute, reconstruct the metadata from tool itself
    Parameters
    ----------
    tool : NodeFunc
        tool to extract metadata from
    Returns
    -------
    dict
        dict containing tool metadata
    """
    # if the tool was created with @tool_call or has an attribute called tool_meta
    # return that
    if hasattr(tool, "tool_meta") and isinstance(tool.tool_meta, dict):
        return tool.tool_meta
    # construct the metadata from tool when it wasn't ther
    name = tool.__name__
    doc = tool.__doc__ or "No docstring provided."
    tool_signature = str(signature(tool))
    input_key = f"tool_{name}_args"
    output_key = f"tool_{name}_output"
    return {
        "input_key": input_key,
        "output_key": output_key,
        "schema_model": None,
        "tool_name": name,
        "tool_doc": doc,
        "tool_signature": tool_signature,
    }


def default_tool_prompt(tool: NodeFunc, query_key: str = "user_query") -> str:
    """
    Generates a default prompt for calling a tool via LLM.
    Parameters
    ----------
    tool : NodeFunc
        tool to make default prompt for
    query_key : str
        name of key that has user query
    Returns
    -------
    prompt_template : str
        generic prompt template to use to call the tool based on a query
    """
    tool_meta = get_tool_metadata(tool)
    tool_name = tool_meta.get("tool_name", tool.__name__)
    tool_signature = tool_meta.get("tool_signature", str(signature(tool)))
    tool_doc = tool_meta.get("tool_doc") or "No docstring provided."
    input_key = tool_meta.get("input_key", f"tool_{tool_name}_args")

    prompt = f"""
You are calling a tool named '{tool_name}'.

Tool description:
{tool_doc}

Function signature:
{tool_signature}

Input should be wrapped as a dictionary under key '{input_key}'.

Generate a JSON object with a single key '{input_key}' containing valid arguments
for this tool. Ensure all arguments are properly typed.

Return ONLY valid JSON. Do not include explantions or markdown.

The output should be a JSON that looks as follows, making sure typing is correct :

{{{{
    '{input_key}' : {{{{'arg_1' : value, 'arg_2': value, ....,}}}}
}}}}

The user query is: {{{query_key}}}
"""
    return prompt.strip()


def wrap_tool_prompt(tool: NodeFunc, prompt_template: str) -> str:
    """
    Take an existing prompt template to get arguments for a tool
    and create a wrapped version of it with additional information from
    the tool.
    Parameters
    ----------
    tool : NodeFunc
        the tool that the prompt is for
    prompt_template : str
        the original prompt template
    Returns
    -------
    tool_wrapper : str
        original template wrapped with additional information

    """
    # get the metadata for the tool and extract information to add
    # to the prompt
    tool_meta = get_tool_metadata(tool)
    tool_name = tool_meta.get("tool_name", tool.__name__)
    tool_signature = tool_meta.get("tool_signature", str(signature(tool)))
    tool_doc = tool_meta.get("tool_doc") or "No docstring provided."
    input_key = tool_meta.get("input_key", f"tool_{tool_name}_args")
    # if the prompt template already has a spot for the input key, we don't wrap
    # This is in case the prompt template already is set up for the tool
    if f"{{{input_key}}}" in prompt_template:
        return prompt_template
    # create wrapped template
    tool_wrapper = f"""
You are to create return a JSON with key {input_key} and the arguments for the tool {tool_name}

The docstring (if provided) is:

{tool_doc}

The signature for the tool is:

 {tool_signature}


---------------------

{prompt_template}

---------------------
Return ONLY valid JSON. Do not include explantions or markdown.

The output should be a JSON that looks as follows, making sure typing is correct :

{{{{
    '{input_key}' : {{{{'arg_1' : value, 'arg_2': value, ....,}}}}
}}}}


"""
    return tool_wrapper


def create_tool_llm_node(
    tool: NodeFunc,
    response_fn: ResponseFn,
    name: str,
    next_node_name: str | None = None,
    prompt_template: str | None = None,
    query_key: str = "user_query",
    max_history_pairs: int = 10,
    temperature=0.1,
) -> FunctionalNode:
    """
    Create a node that will use an LLM response function to get the arguments
    for the tool
    Parameters
    ----------
    tool:Callable
        the tool for which arguments are needed
    response_fn : ResponseFn
        function used to send data to an LLM
    name : str
        name of the node
    next_node_name : str | None
        name of next node
    prompt_template : str | None
        prompt template used to get parameters for the tool
    query_key : str
        key used in the state that has the user query
    max_history_pairs: int
        maximum number of pairs of query/response to keep in the history
    temperature : float
        temperature to use for LLM to create query (low is recommended here)

    Returns
    -------
    FunctionaNode:
        functional node with LLM call using the prompt wrapped with extra information
    """
    # use default tool prompt template if none is provided
    # otherwise wrap prompt
    if prompt_template is None:
        tool_prompt_template = default_tool_prompt(tool=tool, query_key=query_key)
    else:
        tool_prompt_template = wrap_tool_prompt(
            tool=tool, prompt_template=prompt_template
        )
    # create the LLM node using the wrapped or default prompt
    return create_llm_node(
        response_fn=response_fn,
        name=name,
        prompt_template=tool_prompt_template,
        query_key=query_key,
        next_node_name=next_node_name,
        max_history_pairs=max_history_pairs,
        temperature=temperature,
    )


def create_tool_node(
    tool: NodeFunc, name: str, next_node_name: str | None = None
) -> FunctionalNode:
    """
    Create a node that uses a given tool as its function
    Parameters
    ----------
    tool : NodeFunc
        tool to be executed
    name: str
        name of the node
    next_node_name: str
        name of next node
    Returns
    -------
    FunctionalNode
        FunctionalNode using tool as func, name, next_node_ndame
    """
    return FunctionalNode(
        func=tool,
        name=name,
        next_node_name=next_node_name,
    )


def create_tool_llm_pair(
    tool: NodeFunc,
    response_fn: ResponseFn,
    llm_node_name: str,
    tool_node_name: str,
    prompt_template: str | None = None,
    tool_node_next_node_name: str | None = None,
    query_key: str = "user_query",
    max_history_pairs: int = 10,
    temperature: float = 0.1,
) -> tuple[FunctionalNode, FunctionalNode]:
    """
    Create a pair of node from a tool, one of which is an LLM node
    to create the argument key, the other to execute the tool
    Parameters
    ----------
    tool : NodeFunc
        tool being used
    response_fn : ResponseFn
        response function for the LLM node
    tool_node_name : str
        name for the tool node
    prompt_template : str | None
        prompt template used for generating tool args (when given)
    tool_node_next_node_name : str | None
        name of node that follows the tool calling node
    query_key : str
        the name of the key holding the user query
    max_history_pairs : int
        number of pairs of queries to keep in history
    temperature : float
        temperature used for LLM to construct tool args (low is recommended)
    Returns
    -------
    llm_node, tool_node : tuple[FunctionalNode, FunctionalNode]
        the llm_node and tool_node for these
    """
    llm_node = create_tool_llm_node(
        tool=tool,
        response_fn=response_fn,
        name=llm_node_name,
        next_node_name=tool_node_name,
        prompt_template=prompt_template,
        query_key=query_key,
        max_history_pairs=max_history_pairs,
    )

    tool_node = create_tool_node(
        tool=tool, name=tool_node_name, next_node_name=tool_node_next_node_name
    )

    return llm_node, tool_node


def wrap_tool_output(
    tool: NodeFunc,
    prompt_template: str,
) -> str:
    """
    Wrap a prompt_template with information including the output of a tool
    to help an LLM analyze the output
    Parameters
    ----------
    tool : NodeFunc:
        tool being used
    prompt_template: str
        initial prompt_template
    Returns
    -------
    prompt_template : str
        wrapped prompt_template with tool information
    """
    # get tool metadata and extract relevant fields to insert into the template
    metadata = get_tool_metadata(tool)
    tool_doc = metadata.get("tool_doc", "No docstring provided")
    tool_name = metadata.get("tool_name", tool.__name__)
    output_key = metadata.get("output_key", f"{tool_name}_output")
    # if the template already has the output_key, return as is
    if f"{{{output_key}}}" in prompt_template:
        return prompt_template
    # create the wrapped template
    prompt_template = f"""
The following tool was called:
{tool_name}

Tool description :
{tool_doc}

The tool returned the following output:
{{{output_key}}}

Use the tool output when generating your response.
------------------------------
If the tool output has the answer , use that directly.
Instructions:

{prompt_template}
"""
    return prompt_template.strip()


def default_tool_summary_prompt(tool: NodeFunc, query_key: str = "user_query") -> str:
    """
    Creates a default prompt to summarize tool output to answer user query
    Parameters
    ----------
    tool : NodeFunc
        tool being used
    query_key : str
        key that has user query
    Returns
    -------
    prompt : str
        prompt template for answering using tool output
    """
    # get metadata and extract relevant fields
    metadata = get_tool_metadata(tool)
    tool_name = metadata.get("tool_name", tool.__name__)
    tool_doc = metadata.get("tool_doc", "No docstring provided")
    output_key = metadata.get("output_key", f"{tool_name}_output")
    # create the prompt
    prompt = f"""
A tool named {tool_name} was called.

Tool description:
{tool_doc}

The tool returned the following output:
{{{output_key}}}

Use this output to answer the user's question.

The question is :
{{{query_key}}} 

Return ONLY valid JSON with the format and nothing else:

{{{{
    "answer" : "<your answer>"
}}}}

Your answer should clearly address the user question and incorporate the tool output.
"""

    return prompt.strip()


def create_tool_analysis_node(
    tool: NodeFunc,
    response_fn: ResponseFn,
    name: str,
    next_node_name: str | None = None,
    query_key: str = "user_query",
    prompt_template: str | None = None,
    max_history_pairs: int = 10,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> FunctionalNode:
    f"""
    Create an LLM node to answer user query using the output from a tool
    Parameters
    ----------
    tool : NodeFunc
        tool being used
    response_fn : ResponseFn
        response function for the LLM
    name : str
        name to be given to the node
    next_node_name : Optional[str]
        name for next node
    query_key : str
        key that has user query
    prompt_template : Optional[str]
        when given, template to be used to create answer to user query
    max_history_pairs : int
        number of pairs of responses to keep in history
    temperature : Optional[float]
        temperature to use for the LLM
    max_tokens : Optional[int]
        limit of tokens for LLM to use
    Returns
    -------
    node : FunctionalNode
        node to answer query with tool output
    """
    # get template that includes tool output. Default if none given, otherwise wrapped
    if prompt_template is None:
        prompt_template = default_tool_summary_prompt(tool=tool, query_key=query_key)
    else:
        prompt_template = wrap_tool_output(tool=tool, prompt_template=prompt_template)
    node = create_llm_node(
        response_fn=response_fn,
        name=name,
        prompt_template=prompt_template,
        query_key=query_key,
        next_node_name=next_node_name,
        max_history_pairs=max_history_pairs,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return node


def create_retry_llm_conditional(
    tool_node: FunctionalNode,
    retry_llm_node: FunctionalNode,
) -> Callable[[dict[str, Any]], str]:
    """
    create conditional function to check for parse error
    and point to retry for parse error node if there was error
    otherwise point to the tool calling node
    Parameters
    ----------
    tool_node : FunctionalNode
        node where the tool is called
    retry_llm_node: FunctionalNode
        node that has retry for parse errors
    Returns
    -------
    conditional_retry : Callable[[dict[str, Any], str]]
        condtional function to direct where to go
    """
    tool_name = tool_node.name
    retry_llm_name = retry_llm_node.name

    def conditional_retry(state: dict) -> str:
        if state.get("parse_error", False):
            return retry_llm_name
        return tool_name

    return conditional_retry


def retry_llm_prompt(
    tool: NodeFunc,
    state: dict[str, Any],
    prompt_template: str | None = None,
    query_key: str = "user_query",
) -> str:
    """
    Create prompt for retrying creating tool arguments
    when there was a parsing error
    Parameters
    ----------
    tool : NodeFunc
        tool being used
    state: dict[str, Any],
        the state dict
    prompt_template : Optional[str]
        when given, the base prompt
    query_key : str
        name of key with user query
    Returns
    -------
    retry_prompt : str
        prompt with information about parsing error to retry
        getting proper JSON
    """
    # get tool metadata and extract relevant fields
    metadata = get_tool_metadata(tool)
    tool_name = metadata.get("tool_name", tool.__name__)
    parse_error = state["parse_error_message"]
    raw_output = state["raw_output"]
    # get base prompt. When none given, it's the default tool prompt
    # when given we wrap in the tool prompt
    if prompt_template is None:
        base_prompt = default_tool_prompt(tool=tool, query_key=query_key)
    else:
        base_prompt = wrap_tool_prompt(tool=tool, prompt_template=prompt_template)
    # create prompt with injected error information
    retry_prompt = f"""
The previous attempt to create tool arguments for {tool_name} failed to parse as JSON

The raw output from the previous attempt was {raw_output}

This gave the error message {parse_error}

Please correct the errors to proper JSON for the tool call

-------------
{base_prompt}
"""
    return retry_prompt


def retry_tool_call_prompt(
    tool: NodeFunc,
    state: dict[str, Any],
    prompt_template: str | None = None,
    query_key: str = "user_query",
) -> str:
    """
    create prompt to retry getting tool arguments when tool call failed
    Parameters
    ---------
    tool : NodeFunc
        tool being used
    state : dict[str, Any]
        state dict
    prompt_template : Optional[str]
        When given base prompt template
    query_key : str
        key with user query
    Returns
    -------
    retry_prompt : str
        prompt to retry getting tool arguments with information injected
    """
    # get tool metadata and extract relevant fields
    metadata = get_tool_metadata(tool)
    tool_name = metadata.get("tool_name", tool.__name__)
    output_key = metadata.get("output_key", "tool_output")
    args_key = f"{output_key}_args"
    tool_error = state.get(output_key, "Unknown error")
    tool_args = state.get(args_key, {})
    # get default tool prompt or wrapped prompt_template as base prompt
    if prompt_template is None:
        base_prompt = default_tool_prompt(tool=tool, query_key=query_key)
    else:
        base_prompt = wrap_tool_prompt(tool=tool, prompt_template=prompt_template)
    # inject information about the failure into the prompt.
    retry_prompt = f"""
The attempt to call tool {tool_name} failed. 

The tool returned an error message of {tool_error}. 

The attempted arguments were {{{tool_args}}}

Please adjust the arguments to that the tool can be called successfully.

-----------------------------------------------------------------------

{base_prompt}
"""
    return retry_prompt


def create_retry_llm_prompt_func(
    tool: NodeFunc,
    prompt_template: str | None = None,
    query_key: str = "user_query",
) -> Callable[[dict[str, Any]], str]:
    """
    create callable prompt function that only uses state
    by using closure with retry_llm_prompt
    Parameters
    ----------
    tool : NodeFunc
        tool being used
    prompt_template : Optional[str]
        when given, base prompt template
    query_key : str
        key for user query
    Returns
    -------
    _call : Callable[[dict[str,Any]], str]
        function that can be used in LLMCall to help generate a prompt
    """

    def _call(state: dict[str, Any]) -> str:
        return retry_llm_prompt(
            tool=tool, state=state, prompt_template=prompt_template, query_key=query_key
        )

    return _call


def create_retry_llm_node(
    tool: NodeFunc,
    response_fn: ResponseFn,
    name: str,
    prompt_template: str | None = None,
    query_key: str = "user_query",
    next_node_name: str | None = None,
    max_history_pairs: int = 10,
    temperature: float = 0.1,
) -> FunctionalNode:
    """
    Create a node that will attempt to fix a parse error from
    a previous attempt to generate tool arguments
    Parameters
    ----------
    tool : NodeFunc
        tool being used
    response_fn : ResponseFn
        response function for the LLM
    name : str
        name of the node
    prompt_template : Optiona[str]
        when provided, base prompt template
    query_key : str
        key that contains the user query
    next_node_name : Optional[str]
        name for next node
    max_history_pairs : int
        number of pairs of responses to keep in history
    temperature : float
        temperature used to generate the response (low is recommended here)
    Returns
    -------
    FunctionalNode
        node that will take parse error info and attempt to try building
        tool arguments again
    """
    prompt_func = create_retry_llm_prompt_func(
        tool=tool, prompt_template=prompt_template, query_key=query_key
    )
    return create_llm_node(
        response_fn=response_fn,
        name=name,
        prompt_template=prompt_func,
        query_key=query_key,
        next_node_name=next_node_name,
        max_history_pairs=max_history_pairs,
        temperature=temperature,
    )


def create_conditional_parse_retry_node(
    tool_node: FunctionalNode,
    retry_llm_node: FunctionalNode,
    name: str,
) -> ConditionalNode:
    """
    Create conditional node to either move on to tool node
    or go to node to retry after parse error
    Parameters
    ----------
    tool_node : FunctionalNode
        Node to call the tool if parsing is successful
    retry_llm_node: FunctionalNode
        node to retry LLM call if there was a parse error
    name : str
        name for conditional node
    Returns
    -------
    ConditionalNode
        node to direct where to go next depending on parse error
    """
    conditional_func = create_retry_llm_conditional(
        tool_node=tool_node,
        retry_llm_node=retry_llm_node,
    )
    return ConditionalNode(name=name, condition_fn=conditional_func)


def create_retry_parse_error_pair(
    tool: NodeFunc,
    response_fn: ResponseFn,
    retry_node_name: str,
    conditional_node_name: str,
    tool_node: FunctionalNode,
    prompt_template: str | None = None,
    query_key: str = "user_query",
    max_history_pairs: int = 10,
    temperature: float = 0.1,
):
    f"""
    create a pair of nodes to deal with parse errors
    One ConditionalNode to route afterwards, one to attempt
    to fix parsing errors
    Parameters
    ----------
    tool : NodeFunc
        tool being used
    response_fn : ResponseFn
        response function to use for the LLM
    retry_node_name : str
        name to use for the retry node
    conditional_node_name : str
        name to use for the conditional node
    tool_node : FunctionalNode
        node that has the tool in it
    prompt_template : Optional[str]
        optional prompt template to use for the retry LLM node
    query_key: str
        key that holds the user query
    max_history_pairs : int
        number of pairs to keep in message history in LLM node
    temperature: float
        temperature parameter to use for LLM
    Returns
    -------
    tuple[ConditionalNode, FunctionalNode]
        the conditional node and functional node pair
    """
    # create the retry node. After it generates a new response
    # it will go to the conditional node. This will loop until
    # the LLM produces output that can be parsed as JSON
    retry_node = create_retry_llm_node(
        tool=tool,
        response_fn=response_fn,
        name=retry_node_name,
        prompt_template=prompt_template,
        query_key=query_key,
        next_node_name=conditional_node_name,
        max_history_pairs=max_history_pairs,
        temperature=temperature,
    )

    conditional_node = create_conditional_parse_retry_node(
        tool_node=tool_node, retry_llm_node=retry_node, name=conditional_node_name
    )
    return conditional_node, retry_node


def create_retry_tool_prompt_func(
    tool: NodeFunc,
    prompt_template: str | None = None,
    query_key: str = "user_query",
) -> Callable[[dict[str, Any]], str]:
    """
    create callable prompt function that only uses state
    by using closure with retry_llm_prompt
    Parameters
    ----------
    tool : NodeFunc
        tool being used
    prompt_template : Optional[str]
        when given, base prompt template
    query_key : str
        key for user query
    Returns
    -------
    _call : Callable[[dict[str,Any]], str]
        function that can be used in LLMCall to help generate a prompt
    """

    def _call(state: dict[str, Any]) -> str:
        return retry_tool_call_prompt(
            tool=tool, state=state, prompt_template=prompt_template, query_key=query_key
        )

    return _call


def create_retry_tool_error_node(
    tool: NodeFunc,
    response_fn: ResponseFn,
    name: str,
    prompt_template: str | None = None,
    query_key: str = "user_query",
    next_node_name: str | None = None,
    max_history_pairs: int = 10,
    temperature: float = 0.1,
) -> FunctionalNode:
    """
    Creates a node to retry LLM call after tool call failure
    Parameters
    ----------
    tool : NodeFunc
        tool being used
    response_fn : ResponseFn
        response function to use in the LLM
    name : str
        name of this node
    prompt_template: Optional[str]
        base prompt template for the tool call
    query_key : str
        key for the user query
    next_node_name : Optional[str]
        name of next node
    max_history_pairs : int
        number of pairs of query/response to keep
    temperature: float
        temperature to use for the LLM (recommended to be low here)
    Returns
    -------
    retry_node : FunctionalNode
        node that will incorporate tool errors into prompt to attempt retry
        to get tool arguments
    """
    prompt_func = create_retry_tool_prompt_func(
        tool=tool, prompt_template=prompt_template, query_key=query_key
    )
    retry_node = create_llm_node(
        response_fn=response_fn,
        name=name,
        prompt_template=prompt_func,
        query_key=query_key,
        next_node_name=next_node_name,
        max_history_pairs=max_history_pairs,
        temperature=temperature,
    )

    return retry_node


def create_retry_tool_conditional(
    tool_analysis_node: FunctionalNode,
    retry_tool_node: FunctionalNode,
    output_key: str,
) -> Callable[[dict[str, Any]], str]:
    """
    create conditional function to check for tool error
    and point to retry for tool error node if there was error
    otherwise point to the tool analysis node
    Parameters
    ----------
    tool_analysis_node : FunctionalNode
        node where the tool analysis is
    retry_tool_node: FunctionalNode
        node that has retry for tool call errors
    output_key : str
        key that has output from the tool
    Returns
    -------
    conditional_retry : Callable[[dict[str, Any], str]]
        conditional function to direct where to go
    """
    tool_analysis_name = tool_analysis_node.name
    retry_tool_name = retry_tool_node.name
    success_key = f"{output_key}_success"

    def conditional_retry(state: dict) -> str:
        if not state.get(success_key, False):
            return retry_tool_name
        return tool_analysis_name

    return conditional_retry


def create_conditional_tool_retry_node(
    tool: NodeFunc,
    tool_analysis_node: FunctionalNode,
    retry_tool_node: FunctionalNode,
    name: str,
) -> ConditionalNode:
    """
    creates conditional node to determine wither to go to the retry tool node
    or to the tool analysis node
    Parameters
    ----------
    tool : NodeFunc
        tool being used
    tool_analysis_node : FunctionalNode,
        node with the tool analysis
    retry_tool_node : FunctionalNode
        node to retry after tool failure
    name : str
        name for the conditional node
    Returns
    -------
    ConditionalNode
        either goes to tool call or to node to retry with tool
        error information
    """
    metadata = get_tool_metadata(tool)
    output_key = metadata.get("output_key", f"{tool.__name__}_output")
    conditional_func = create_retry_tool_conditional(
        tool_analysis_node=tool_analysis_node,
        retry_tool_node=retry_tool_node,
        output_key=output_key,
    )

    return ConditionalNode(name=name, condition_fn=conditional_func)


def create_retry_tool_error_pair(
    tool: NodeFunc,
    response_fn: ResponseFn,
    tool_node: FunctionalNode,
    retry_tool_name: str,
    check_tool_name: str,
    check_parse_name: str,
    prompt_template: str | None = None,
    query_key: str = "user_query",
    max_history_pairs: int = 10,
    temperature: float = 0.1,
):

    retry_tool_node = create_retry_tool_error_node(
        tool=tool,
        response_fn=response_fn,
        name=retry_tool_name,
        prompt_template=prompt_template,
        query_key=query_key,
        next_node_name=check_parse_name,
        max_history_pairs=max_history_pairs,
        temperature=temperature,
    )

    conditional_tool_retry_node = create_conditional_tool_retry_node(
        tool=tool,
        tool_node=tool_node,
        retry_tool_node=retry_tool_node,
        name=check_tool_name,
    )

    return conditional_tool_retry_node, retry_tool_node
