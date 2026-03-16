"""
module to hold LLMCall class which
is callable to be used in a FunctionalNode to communicate
with an LLM
"""

from typing import Any, Callable
from llm_graph.utils import ResponseFn, json_parse

PromptTemplate = str | Callable[[dict(str, Any), str], str] | None


class LLMCall:
    """
    class to use as a callable func in a node communicating with
    an LLM
    """

    def __init__(
        self,
        response_fn: ResponseFn,
        prompt_template: PromptTemplate,
        max_history_pairs: int = 10,
        query_key: str = "user_query",
        temperature: float | None = None,
        max_tokens: int | None = None,
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
        temperature : float | None
            optional temperature to pass to the client for generating a response
        max_tokens : int | None
            optional limit on tokens to use
        """
        self.prompt_template = prompt_template
        self.response_fn = response_fn
        self.max_history_pairs = max_history_pairs
        self.query_key = query_key
        self.temperature = temperature
        self.max_tokens = max_tokens

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
        prompt = self._build_prompt(state)
        messagelist = history + [{"role": "user", "content": prompt}]
        # get the response from the response_fn
        response: dict[str, Any] = self.response_fn(
            history=messagelist,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        # get the raw string and append the output to the message history
        raw = response.get("raw_output", "")
        new_history = messagelist + [{"role": "assistant", "content": raw}]
        # trim message history if necessary
        new_history = new_history[-2 * self.max_history_pairs :]
        # parse the output if possible and return that as part of the output
        parsed = json_parse(raw)
        return response | parsed | {"message_history": new_history}

    def _build_prompt(self, state: dict[str, Any]) -> str:
        # if there is a prompt_template, use the state to populate fields
        template = (
            self.prompt_template(state)
            if callable(self.prompt_template)
            else self.prompt_template
        )
        if template:
            try:
                return template.format(**state)
            except KeyError as e:
                raise KeyError(f"Prompt template missing required state field: {e}")
        elif self.query_key in state:
            # if no prompt exists, just get the query directly
            return state[self.query_key]
        else:
            raise KeyError(f"{self.query_key} key missing and no prompt template given")

    def preview_prompt(self, state: dict[str, Any]) -> str:
        return self._build_prompt(state)
