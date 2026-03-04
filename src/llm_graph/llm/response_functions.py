from typing import Callable
from openai import OpenAI

def dict_to_str(hist_dict):
    return f"{hist_dict['role']} : {hist_dict['content']}"
    
def message_hist_to_str(history):
    return "\n".join([dict_to_str(hist_dict) for hist_dict in history])

def dummy_llm_response_fn(
        input : str,
        history = None,
):
        if history is None:
            history = []
        history_text = message_hist_to_str(history)
        if history_text:
            input = history_text + "\n" + input
        return {"raw_output": input.lower(), "input": input}
    
def OpenAI_response_fn(
        client: OpenAI,
        model : str = "gpt-5.2",
        instructions : str | None = None
) -> Callable:

    
    def _call(
            input: str,
            history: list[dict] | None = None,
    ):
        if history is None:
            history = []
        history_text = message_hist_to_str(history)
        if history_text:
            input = history_text + "\n" + input
        
        response = client.responses.create(
            input=input,
            model=model,
            instructions=instructions
        )
        return {
        "raw_output": response.output_text,
        "usage": response.usage.to_dict() if response.usage else None
        }
    return _call
