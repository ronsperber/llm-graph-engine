from inspect import signature
from typing import Callable, Any
from llm_graph.core.nodes import FunctionalNode
from .llm import create_llm_node
from llm_graph.utils import tool_call, ResponseFn

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

The user query is: {{{query_key}}}
"""
    return prompt.strip()


def wrap_tool_prompt(tool: Callable, prompt_template: str) -> str:
    tool_meta = get_tool_metadata(tool)
    tool_name = tool_meta.get("tool_name", tool.__name__)
    tool_signature = tool_meta.get("tool_signature", str(signature(tool)))
    tool_doc = tool_meta.get("tool_doc") or "No docstring provided."
    input_key = tool_meta.get("input_key", f"tool_{tool_name}_args")
    tool_wrapper = f"""
You are to create return a JSON with key {input_key} and the arguments for the tool {tool_name}

The docstring (if provided) is:

{tool_doc}

The signature for the tool is:

 {tool_signature}


---------------------

{prompt_template}

---------------------

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
        max_history_pairs=max_history_pairs
    )

    tool_node = create_tool_node(
        tool=tool,
        name=tool_node_name,
        next_node_name=tool_node_next_node_name
    )
    
    return llm_node, tool_node
    
