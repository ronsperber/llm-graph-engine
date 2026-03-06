from typing import Callable, Any
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

def dict_to_str(hist_dict):
    return f"{hist_dict['role']} : {hist_dict['content']}"
    
def message_hist_to_str(history):
    return "\n".join([dict_to_str(hist_dict) for hist_dict in history])

def dummy_llm_response_fn(
        history : list[ChatCompletionMessageParam]
):
        if history is None:
            history = []
        history_text = message_hist_to_str(history)
        return {"raw_output": history_text.lower(), "input": history[-1]}
    
def OpenAI_response_fn(
        client: OpenAI,
        model : str = "gpt-5.2",
) -> Callable:

    
    def _call(
            history: list[ChatCompletionMessageParam],
    ) -> dict[str, Any]:
        response = client.chat.completions.create(
            messages=history,
            model=model
        )
        return {
        "raw_output": response.choices[0].message.content or "",
        "usage": response.usage.to_dict() if response.usage else {}
        }
    return _call
