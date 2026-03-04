
from typing import Callable
import json

def json_parse(s: str):
    try:
        parsed = json.loads(s)
    except Exception:
        parsed = {}
    return parsed

    
class LLMCall:
    def __init__(
            self,
            response_fn: Callable,
            prompt_template: str | None = None,
            max_history_pairs = 10
    ):
        self.prompt_template = prompt_template
        self.response_fn = response_fn
        self.max_history_pairs = max_history_pairs

    def __call__(self, state: dict):
        history = state.get("message_history", [])
        user_query = state.get("user_query")
        if self.prompt_template:
            prompt = self.prompt_template.format(**state)
        elif "user_query" in state:
            prompt = state["user_query"]
        else:
            raise KeyError("user_query key missing and no prompt template given")
        response = self.response_fn(
            input=prompt,
            history = history
        )
        raw = response["raw_output"]
        new_history = history + [
            {"role": "user", "content": user_query},
            {"role": "assistant" , "content": raw}
        ]
        new_history = new_history[-2 * self.max_history_pairs:]
        parsed = json_parse(raw)
        return response | parsed | {"message_history": new_history}
        
        