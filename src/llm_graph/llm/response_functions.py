"""
module with factories to create
response functions used in LLMCall
"""

from typing import Any
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletion
from llm_graph.utils import ResponseFn


def dict_to_str(hist_dict: dict[str, str]) -> str:
    """
    converts one dict {"role":role, "content":content} to a string
    """
    return f"{hist_dict['role']} : {hist_dict['content']}"


def message_hist_to_str(history: list[dict[str, str]]) -> str:
    """
    takes a list of dicts that are role:rolename, content:content and
    stitch together to get one string
    """
    return "\n".join([dict_to_str(hist_dict) for hist_dict in history])


def dummy_llm_response_fn(
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    a simple dummy LLM response function
    used for testing
    """
    if history is None:
        history = []
    history_text = message_hist_to_str(history)
    return {"raw_output": history_text.lower(), "input": history[-1]}


def OpenAI_response_fn(
    client: OpenAI,
    model: str = "gpt-5.2",
) -> ResponseFn:
    """
    creates a response_fn that can go in to an LLMCall
    Parameters
    ----------
    client : OpenAI
        client used to send prompts and receive responses
    model : str
        name of model to be used

    Returns
    _call : callable
        function that takes a message history and
        returns the output and usage
    """

    def _call(
        history: list[ChatCompletionMessageParam],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        response function that gets plugged in
        Parameters
        ----------
        history : list[ChatCompletionMessageParam]
            the list of messages ending in the latest
            message from the user
        **kwargs
            optional additional arguments

        Returns
        -------
        dict:
            dictionary with keys 'raw_output' holding the LLM response text and
            'usage' with the token usage
        """
        params = {"messages": history, "model": model, **kwargs}

        params = {k: v for k, v in params.items() if v is not None}
        response: ChatCompletion = client.chat.completions.create(**params)

        return {
            "raw_output": response.choices[0].message.content or "",
            "usage": response.usage.to_dict() if response.usage else {},
        }

    return _call
