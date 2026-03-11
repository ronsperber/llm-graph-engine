"""
module to hold LLMCall class which
is callable to be used in a FunctionalNode to communicate
with an LLM
"""
from typing import Callable, Any
from openai.types.chat import ChatCompletionMessageParam
import json

def json_parse(s: str):
    """
    helper function to try to parse output into a dict
    Parameters
    ----------
    s : str
        raw text from LLM
    Returns
    -------
        parsed : dict
            when the string is in proper JSON format, this is the parsed dict from that
            when that was unable to be done, it returns a dict with 'raw_output' and 'parse_error'
            keys
    """
    try:
        parsed = json.loads(s)
    except Exception:
        parsed = {"raw_output": s, "parse_error": True}
    return parsed

    
class LLMCall:
    """
    class to use as a callable func in a node communicating with
    an LLM
    """
    def __init__(
            self,
            response_fn: Callable[[list[ChatCompletionMessageParam]], dict[str, Any]],
            prompt_template: str | None = None,
            max_history_pairs: int = 10,
            query_key: str = "user_query"
    ):
        """
        initialization

        Parameters
        ----------
        response_fn : Callable[[list[ChatCompletionMessageParam]], dict[str, Any]]
            the function that will pass the message history to the LLM and get a response
        prompt_template : str | None
            optional prompt template to format the raw query
        max_history_pairs : int
            how many query/response pairs to keep in the history
        query_key: str
            what the key is that has the user query in it
        """
        self.prompt_template = prompt_template
        self.response_fn = response_fn
        self.max_history_pairs = max_history_pairs
        self.query_key = query_key

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Callable function for this class
        
        Parameters
        ----------
        state: dict[str, Any]
            the current state of the graph that is going to be passed from
            the node
        Returns
        -------
        dict[str, Any]
            returns a dict that has the response from the LLM, the updated message
            history, and the parsed output
        
        """
        # get the history
        history = list(state.get("message_history", []))
        history = history.copy()
        # if there is a prompt_template, use the state to populate fields
        if self.prompt_template:
            try:
                prompt = self.prompt_template.format(**state)
            except KeyError as e:
                raise KeyError(f"Prompt template missing required state field: {e}")
        elif self.query_key in state:
            # if no prompt exists, just get the query directly
            prompt = state[self.query_key]
        else:
            raise KeyError(f"{self.query_key} key missing and no prompt template given")
        # update the message history to include the latest prompt
        messagelist = history + [{"role": "user", "content" : prompt}]
        # get the response from the response_fn
        response = self.response_fn(
            messagelist
        )
        # get the raw string and append the output to the message history
        raw = response.get("raw_output", "")
        new_history = messagelist + [{"role": "assistant", "content": raw}]
        # trim message history if necessary
        new_history = new_history[-2 * self.max_history_pairs:]
        # parse the output if possible and return that as part of the output
        parsed = json_parse(raw)
        return response | parsed | {"message_history": new_history}
        
        