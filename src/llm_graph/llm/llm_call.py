
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
            max_history_pairs = 10,
            query_key: str = "user_query"
    ):
        self.prompt_template = prompt_template
        self.response_fn = response_fn
        self.max_history_pairs = max_history_pairs
        self.query_key = query_key

    def __call__(self, state: dict):
        history = state.get("message_history", [])
        if self.prompt_template:
            prompt = self.prompt_template.format(**state)
        elif self.query_key in state:
            prompt = state[self.query_key]
        else:
            raise KeyError(f"{self.query_key} key missing and no prompt template given")
        user_query = state.get(self.query_key, prompt)
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
        
        