from llm_graph.llm.response_functions import (
    dict_to_str,
    message_hist_to_str,
    dummy_llm_response_fn,
)


def test_dict_to_str_formats_role_and_content():
    d = {"role": "user", "content": "Hello"}
    assert dict_to_str(d) == "user : Hello"


def test_message_hist_to_str_joins_messages_with_newlines():
    hist = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    s = message_hist_to_str(hist)
    assert s == "user : Hi\nassistant : Hello"


def test_dummy_llm_response_fn_lowercases_history_and_returns_last_message():
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "THERE"},
        {"role": "user", "content": "LOUD"},
    ]

    result = dummy_llm_response_fn(history)

    assert result["raw_output"] == "user : hi\nassistant : there\nuser : loud"
    assert result["input"] == history[-1]


def test_dummy_llm_response_fn_with_none_history_raises_index_error():
    # current implementation converts None to [] but still indexes history[-1],
    # which results in an IndexError; this test captures that behavior.
    try:
        dummy_llm_response_fn(None)  # type: ignore[arg-type]
    except IndexError:
        # expected current behavior
        return
    # if no error was raised, then behavior changed and this should fail
    assert False, "Expected IndexError when history is None/empty"
