import pytest

from llm_graph.core.graphrunner import GraphRunner
from llm_graph.core.nodes import FunctionalNode, ConditionalNode


def _make_noop_node(name: str, next_node_name: str | None = None):
    return FunctionalNode(
        func=lambda state: {}, name=name, next_node_name=next_node_name
    )


def test_execute_raises_type_error_for_non_dict_input():
    node = _make_noop_node("A", None)
    runner = GraphRunner(nodes=[node], start_node="A")

    with pytest.raises(TypeError):
        runner.execute("not-a-dict")  # type: ignore[arg-type]


def test_nodes_argument_accepts_list_set_tuple_and_dict():
    node_list = [_make_noop_node("A")]
    GraphRunner(nodes=node_list, start_node="A")

    node_set = {_make_noop_node("B")}
    GraphRunner(nodes=node_set, start_node="B")

    node_tuple = (_make_noop_node("C"),)
    GraphRunner(nodes=node_tuple, start_node="C")

    node_d = _make_noop_node("D")
    GraphRunner(nodes={"D": node_d}, start_node="D")


def test_nodes_dict_with_mismatched_keys_raises_value_error():
    node = _make_noop_node("A")
    with pytest.raises(ValueError):
        GraphRunner(nodes={"wrong": node}, start_node="A")


def test_duplicate_node_names_raise_value_error():
    node1 = _make_noop_node("A")
    node2 = _make_noop_node("A")
    with pytest.raises(ValueError):
        GraphRunner(nodes=[node1, node2], start_node="A")


def test_invalid_nodes_type_raises_type_error():
    with pytest.raises(TypeError):
        GraphRunner(nodes="not-iterable-of-nodes", start_node="A")  # type: ignore[arg-type]

def test_invalid_node_elements_raises_type_error():
    with pytest.raises(TypeError):
        GraphRunner(nodes=["a", "b", "c"], start_node="A")

def test_reset_trace_false_appends_to_existing_trace():
    node = _make_noop_node("A")
    runner = GraphRunner(nodes=[node], start_node="A")

    runner.execute({"x": 1})
    assert len(runner.trace_log) == 1

    runner.execute({"x": 2}, reset_trace=False)
    assert len(runner.trace_log) == 2


def test_token_usage_accumulates_from_node_usage():
    def usage_func(state):
        return {"usage": {"prompt_tokens": 3, "completion_tokens": 7}}

    node = FunctionalNode(func=usage_func, name="A")
    runner = GraphRunner(nodes=[node], start_node="A")

    result = runner.execute({"x": 1})
    assert "token_usage" in result

    usage = runner.get_token_usage()
    assert usage["in_tokens"] == 3
    assert usage["out_tokens"] == 7
    assert usage["total_tokens"] == 10


def test_conditional_node_drives_control_flow():
    def condition_fn(state):
        return "B" if state.get("go_b") else "C"

    cond = ConditionalNode(name="cond", condition_fn=condition_fn)

    def node_b_func(state):
        return {"went": "B"}

    def node_c_func(state):
        return {"went": "C"}

    node_b = FunctionalNode(func=node_b_func, name="B")
    node_c = FunctionalNode(func=node_c_func, name="C")

    runner = GraphRunner(nodes=[cond, node_b, node_c], start_node="cond")

    res_b = runner.execute({"go_b": True})
    assert res_b["state_dict"]["went"] == "B"

    res_c = runner.execute({"go_b": False})
    assert res_c["state_dict"]["went"] == "C"


def test_start_node_not_in_nodes_raises_value_error():
    node_a = FunctionalNode(func=lambda s: {}, name="A")
    with pytest.raises(ValueError, match="start_node"):
        GraphRunner(nodes=[node_a], start_node="MISSING")


def test_max_node_visits_zero_with_error_raises_before_node_executes():
    executed = []

    def func(state):
        executed.append(True)
        return {}

    node = FunctionalNode(func=func, name="A", next_node_name="A")
    runner = GraphRunner(
        nodes=[node], start_node="A", max_node_visits=0, on_max_visits="error"
    )

    with pytest.raises(RuntimeError):
        runner.execute({})

    assert executed == []


def test_max_node_visits_zero_with_exit_terminates_early_without_execution():
    executed = []

    def func(state):
        executed.append(True)
        return {}

    node = FunctionalNode(func=func, name="A", next_node_name="A")
    runner = GraphRunner(
        nodes=[node], start_node="A", max_node_visits=0, on_max_visits="exit"
    )

    result = runner.execute({})

    assert result["terminated_early"] is True
    assert executed == []
    assert result["state_dict"] == {}


def test_unrecognized_on_max_visits_value_treated_like_exit():
    executed = []

    def func(state):
        executed.append(True)
        return {}

    node = FunctionalNode(func=func, name="A", next_node_name="A")
    runner = GraphRunner(
        nodes=[node], start_node="A", max_node_visits=0, on_max_visits="something_else"
    )

    result = runner.execute({})

    assert result["terminated_early"] is True
    assert executed == []


# ---------------------------------------------------------------------------
# GraphRunner.build
# ---------------------------------------------------------------------------

class TestGraphRunnerBuild:

    def _make_dict(self, *names, chain=True):
        """Build a node dict. When chain=True, each node points to the next."""
        nodes = [_make_noop_node(n) for n in names]
        if chain:
            for i in range(len(nodes) - 1):
                nodes[i].set_next_node(nodes[i + 1].name)
        return {n.name: n for n in nodes}

    def test_single_node_dict_produces_valid_runner(self):
        d = self._make_dict("A")
        runner = GraphRunner.build(node_dicts=[d], start_node="A")
        assert isinstance(runner, GraphRunner)
        assert set(runner.nodes_dict.keys()) == {"A"}

    def test_two_node_dicts_merged_into_runner(self):
        d1 = self._make_dict("A", "B")
        d2 = self._make_dict("C", "D")
        runner = GraphRunner.build(node_dicts=[d1, d2], start_node="A")
        assert set(runner.nodes_dict.keys()) == {"A", "B", "C", "D"}

    def test_connections_wires_last_node_of_first_dict_to_first_node_of_second(self):
        # A → B (internal), C → D (internal); connection needed: B → C
        d1 = self._make_dict("A", "B")
        d2 = self._make_dict("C", "D")
        runner = GraphRunner.build(
            node_dicts=[d1, d2],
            start_node="A",
            connections={"B": "C"},
        )
        assert runner.nodes_dict["B"].next_node_name == "C"

    def test_connections_full_execution_crosses_both_dicts(self):
        # A → B, then connection B → C, then C → D (terminal)
        visited = []

        def make_tracking_node(name, next_name=None):
            def func(state):
                visited.append(name)
                return {}
            return FunctionalNode(func=func, name=name, next_node_name=next_name)

        d1 = {"A": make_tracking_node("A", "B"), "B": make_tracking_node("B")}
        d2 = {"C": make_tracking_node("C", "D"), "D": make_tracking_node("D")}

        runner = GraphRunner.build(
            node_dicts=[d1, d2],
            start_node="A",
            connections={"B": "C"},
        )
        runner.execute({})
        assert visited == ["A", "B", "C", "D"]

    def test_none_connections_defaults_to_empty(self):
        d = self._make_dict("A", "B")
        runner = GraphRunner.build(node_dicts=[d], start_node="A", connections=None)
        assert isinstance(runner, GraphRunner)

    def test_duplicate_keys_across_dicts_raise_value_error(self):
        d1 = self._make_dict("A", "B")
        d2 = self._make_dict("B", "C")
        with pytest.raises(ValueError, match="Duplicate"):
            GraphRunner.build(node_dicts=[d1, d2], start_node="A")

    def test_connection_with_unknown_source_raises_value_error(self):
        d = self._make_dict("A", "B")
        with pytest.raises(ValueError, match="source"):
            GraphRunner.build(node_dicts=[d], start_node="A", connections={"Z": "A"})

    def test_connection_with_unknown_target_raises_value_error(self):
        d = self._make_dict("A", "B")
        with pytest.raises(ValueError, match="target"):
            GraphRunner.build(node_dicts=[d], start_node="A", connections={"B": "Z"})

    def test_max_node_visits_forwarded_to_runner(self):
        d = {"A": FunctionalNode(func=lambda _: {}, name="A", next_node_name="A")}
        runner = GraphRunner.build(
            node_dicts=[d], start_node="A", max_node_visits=2, on_max_visits="exit"
        )
        result = runner.execute({})
        assert result["terminated_early"] is True
