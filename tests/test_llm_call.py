import pytest

from llm_graph.llm.llm_call import LLMCall, json_parse


def test_json_parse_valid_json():
    s = '{"a": 1, "b": "two"}'
    parsed = json_parse(s)
    assert parsed == {"a": 1, "b": "two"}


def test_json_parse_invalid_json_returns_fallback_dict():
    s = "{not valid json}"
    parsed = json_parse(s)
    assert parsed["raw_output"] == s
    assert parsed["parse_error"] is True


def test_llmcall_uses_query_key_when_no_template():
    def response_fn(history):
        return {
            "raw_output": '{"answer": "ok"}',
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }

    llm_call = LLMCall(response_fn=response_fn)
    state = {"user_query": "Hello"}

    result = llm_call(state)

    assert result["raw_output"] == '{"answer": "ok"}'
    assert result["answer"] == "ok"
    assert result["usage"]["prompt_tokens"] == 1
    assert isinstance(result["message_history"], list)
    assert result["message_history"][-1]["role"] == "assistant"


def test_llmcall_with_prompt_template_formats_from_state():
    captured_prompts = []

    def response_fn(history):
        captured_prompts.append(history[-1]["content"])
        return {"raw_output": '{"done": true}'}

    llm_call = LLMCall(
        response_fn=response_fn,
        prompt_template="Hello {name}, you said: {query}",
        max_history_pairs=2,
    )

    prior_history = [
        {"role": "user", "content": "old q"},
        {"role": "assistant", "content": "old a"},
    ]

    state = {
        "name": "Alice",
        "query": "How are you?",
        "message_history": prior_history,
    }

    result = llm_call(state)

    assert captured_prompts[-1] == "Hello Alice, you said: How are you?"
    assert result["done"] is True
    assert len(result["message_history"]) <= 2 * llm_call.max_history_pairs


def test_llmcall_raises_when_missing_query_and_template():
    def response_fn(history):
        return {"raw_output": "{}"}

    llm_call = LLMCall(response_fn=response_fn)

    with pytest.raises(KeyError):
        llm_call({})


def test_llmcall_raises_when_prompt_template_missing_state_field():
    def response_fn(history):
        return {"raw_output": "{}"}

    llm_call = LLMCall(response_fn=response_fn, prompt_template="Hello {missing}")

    with pytest.raises(KeyError) as excinfo:
        llm_call({})

    assert "Prompt template missing required state field" in str(excinfo.value)
