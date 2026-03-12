from llm_graph.core.graphrunner import GraphRunner
from llm_graph.core.nodes import FunctionalNode
from llm_graph.core.sessionrunner import SessionRunner


def _make_counter_graph():
    def counter_func(state):
        current = state.get("session_value", 0)
        return {"session_value": current + 1, "non_session": "temp"}

    node = FunctionalNode(func=counter_func, name="counter")
    return GraphRunner(nodes=[node], start_node="counter")


def test_session_keys_persist_across_executes():
    graph = _make_counter_graph()
    runner = SessionRunner(graph=graph, session_keys=["session_value"])

    first = runner.execute({})
    assert first["state_dict"]["session_value"] == 1
    assert runner.session_dict["session_value"] == 1

    second = runner.execute({})
    assert second["state_dict"]["session_value"] == 2
    assert runner.session_dict["session_value"] == 2


def test_no_session_keys_means_no_persistence():
    graph = _make_counter_graph()
    runner = SessionRunner(graph=graph, session_keys=[])

    first = runner.execute({})
    second = runner.execute({})

    assert "session_value" in first["state_dict"]
    assert "session_value" in second["state_dict"]
    assert runner.session_dict == {}


def test_input_overrides_session_values_when_keys_overlap():
    graph = _make_counter_graph()
    runner = SessionRunner(graph=graph, session_keys=["session_value"])

    runner.execute({})
    assert runner.session_dict["session_value"] == 1

    # when providing a new session_value in the input, it should override
    # the previous one when constructing graph_input
    result = runner.execute({"session_value": 10})
    assert result["state_dict"]["session_value"] == 11


def test_trace_logs_accumulate_for_each_execution():
    graph = _make_counter_graph()
    runner = SessionRunner(graph=graph, session_keys=["session_value"])

    runner.execute({})
    runner.execute({})

    assert len(runner.trace_logs) == 2
    assert all(isinstance(t, list) for t in runner.trace_logs)


def test_session_keys_default_none_behaves_like_empty_list():
    graph = _make_counter_graph()
    # passing None should behave the same as an empty list of session keys
    runner = SessionRunner(graph=graph, session_keys=None)

    first = runner.execute({})
    second = runner.execute({})

    assert "session_value" in first["state_dict"]
    assert "session_value" in second["state_dict"]
    assert runner.session_dict == {}
