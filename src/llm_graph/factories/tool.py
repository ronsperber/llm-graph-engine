from inspect import signature
from typing import Callable, Any
from llm_graph.core.nodes import FunctionalNode
from .llm import create_llm_node
from llm_graph.utils import ResponseFn

def get_tool_metadata(tool:Callable) -> dict[str, Any]:
    """
    gets metadata from tool. If tool was not created with tool_meta
    attribute, reconstruct the metadata from tool itself
    Parameters
    ----------
    tool : Callable
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
        "input_key" : input_key,
        "output_key" : output_key,
        "schema_model" : None,
        "tool_name" : name,
        "tool_doc" : doc,
        "tool_signature": tool_signature,
    }

def default_tool_prompt(tool: Callable, query_key: str = "user_query") -> str:
    """
    Generates a default prompt for calling a tool via LLM.
    Parameters
    ----------
    tool : Callable
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


def wrap_tool_prompt(tool: Callable, prompt_template: str) -> str:
    tool_meta = get_tool_metadata(tool)
    tool_name = tool_meta.get("tool_name", tool.__name__)
    tool_signature = tool_meta.get("tool_signature", str(signature(tool)))
    tool_doc = tool_meta.get("tool_doc") or "No docstring provided."
    input_key = tool_meta.get("input_key", f"tool_{tool_name}_args")
    if f"{{{input_key}}}" in prompt_template:
        return prompt_template
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
    tool : Callable,
    response_fn: ResponseFn,
    name : str,
    next_node_name: str | None = None,
    prompt_template: str | None = None,
    query_key: str = "user_query",
    max_history_pairs: int =10,


):
    if prompt_template is None:
        tool_prompt_template = default_tool_prompt(tool=tool, query_key=query_key)
    else:
        tool_prompt_template = wrap_tool_prompt(tool=tool, prompt_template=prompt_template)
    
    return create_llm_node(
        response_fn=response_fn,
        name=name,
        prompt_template=tool_prompt_template,
        query_key=query_key,
        next_node_name=next_node_name,
        max_history_pairs=max_history_pairs,
        temperature=0.1
    )

def create_tool_node(
    tool: Callable,
    name: str,
    next_node_name: str | None = None
):
    return FunctionalNode(
        func=tool,
        name=name,
        next_node_name=next_node_name,
    )

def create_tool_llm_pair(
    tool: Callable,
    response_fn: ResponseFn,
    llm_node_name: str,
    tool_node_name : str,
    prompt_template : str | None = None,
    tool_node_next_node_name : str | None = None,
    query_key: str = "user_query",
    max_history_pairs: int = 10,
):
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
        tool=tool,
        name=tool_node_name,
        next_node_name=tool_node_next_node_name
    )
    
    return llm_node, tool_node
    
def wrap_tool_output(
    tool : Callable,
    prompt_template : str,
) -> str:
    metadata = get_tool_metadata(tool)
    tool_doc = metadata.get("tool_doc", "No docstring provided")
    tool_name = metadata.get("tool_name", tool.__name__)
    output_key = metadata.get("output_key",f"{tool_name}_output")
    if f"{{{output_key}}}" in prompt_template:
        return prompt_template
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

def default_tool_summary_prompt(
    tool : Callable,
    query_key : str = "user_query"
) -> str:
    metadata = get_tool_metadata(tool)
    tool_name = metadata.get("tool_name", tool.__name__)
    tool_doc = metadata.get("tool_doc", "No docstring provided")
    output_key = metadata.get("output_key", f"{tool_name}_output")
    prompt = f"""
A tool named {tool_name} was called.

Tool description:
{tool_doc}

The tool returned the following output:
{{{output_key}}}

Use this output to answer the user's quertion.

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
    tool : Callable,
    response_fn: ResponseFn,
    name : str,
    next_node_name : str | None = None,
    query_key : str = "user_query",
    prompt_template: str | None = None,
    max_history_pairs : int = 10,
    temperature: float | None = None,
    max_tokens : int | None = None,
):
    if prompt_template is None:
        prompt_template = default_tool_summary_prompt(tool=tool, query_key=query_key)
    else:
        prompt_template = wrap_tool_output(
            tool=tool,
            prompt_template=prompt_template
        )
    node = create_llm_node(
        response_fn = response_fn,
        name=name,
        prompt_template=prompt_template,
        query_key=query_key,
        next_node_name=next_node_name,
        max_history_pairs=max_history_pairs,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return node

def create_retry_conditional(
    llm_analysis_node: FunctionalNode,
    retry_llm_node : FunctionalNode,
    retry_tool_node : FunctionalNode,
    tool_output_key : str,
)-> Callable:
    analysis_name = llm_analysis_node.name
    retry_llm_name = retry_llm_node.name
    retry_tool_name = retry_tool_node.name
    success_key = f"{tool_output_key}_success"
    def conditional_retry(state: dict) -> str:
        if state.get("parse_error", False):
            return retry_llm_name
        if not state.get(success_key, True):
            return retry_tool_name
        return analysis_name
    
    return conditional_retry

def retry_llm_prompt(
    tool: Callable,
    state: dict[str, Any],
    prompt_template: str | None = None,
    query_key:str = "user_query",
    ):
    metadata = get_tool_metadata(tool)
    tool_name = metadata.get("tool_name", tool.__name__)
    parse_error = state["parse_error_message"]
    raw_output = state["raw_output"]
    if prompt_template is None:
        base_prompt = default_tool_prompt(tool=tool, query_key=query_key)
    else:
        base_prompt = wrap_tool_prompt(tool=tool, prompt_template=prompt_template)
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
    tool: Callable,
    state: dict[str, Any],
    prompt_template : str | None = None,
    query_key : str = "user_query",
):
    metadata = get_tool_metadata(tool)
    tool_name = metadata.get("tool_name", tool.__name__)
    output_key = metadata.get("output_key", "tool_output")
    args_key = f"{output_key}_args"
    tool_error = state.get(output_key, "Unknown error")
    tool_args = state.get(args_key, {})
    if prompt_template is None:
        base_prompt = default_tool_prompt(tool=tool, query_key=query_key)
    else:
        base_prompt = wrap_tool_prompt(tool=tool, prompt_template=prompt_template)
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
    tool: Callable,
    prompt_template: str | None = None,
    query_key: str = "user_query",
) -> Callable[[dict[str,Any]], str]:
    def _call(state: dict[str, Any]) -> str:
        return retry_llm_prompt(
            tool=tool,
            state=state,
            prompt_template=prompt_template,
            query_key=query_key
        )
    return _call

def create_retry_llm_node(
    tool: Callable,
    response_fn: ResponseFn,
    name : str,
    prompt_template : str | None = None,
    query_key : str = "user_query",
    next_node_name : str | None = None,
    max_history_pairs: int = 10,
) -> FunctionalNode:
    prompt_func = create_retry_llm_prompt_func(
        tool=tool,
        prompt_template=prompt_template,
        query_key=query_key
    )
    return create_llm_node(
        response_fn=response_fn,
        name=name,
        prompt_template=prompt_func,
        query_key=query_key,
        next_node_name=next_node_name,
        max_history_pairs=max_history_pairs,
        temperature=0.1
    )