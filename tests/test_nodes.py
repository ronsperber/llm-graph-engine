import pytest

from llm_graph.core.nodes import GraphNode, FunctionalNode, ConditionalNode


class DummyNode(GraphNode):
    def _execute_impl(self, state):
        return {"called": True, "input_snapshot": state.copy()}


def test_graphnode_execute_sets_last_input_and_output():
    node = DummyNode(name="dummy")
    state = {"x": 1}

    output = node.execute(state)

    assert output == {"called": True, "input_snapshot": state}
    assert node.last_input == state
    assert node.last_output == output


def test_functionalnode_raises_type_error_on_non_dict_output():
    def bad_func(state):
        return 42

    node = FunctionalNode(func=bad_func, name="bad")

    with pytest.raises(TypeError) as excinfo:
        node.execute({})

    msg = str(excinfo.value)
    assert "Node 'bad' using function bad_func returned" in msg
    assert "expected dict" in msg


def test_functionalnode_lambda_name_in_error_message():
    node = FunctionalNode(func=lambda s: 42, name="lambda_node")

    with pytest.raises(TypeError) as excinfo:
        node.execute({})

    assert "using function <lambda>" in str(excinfo.value)


def test_conditionalnode_sets_next_node_name_and_returns_empty_dict():
    def condition_fn(state):
        return "next_if_true" if state.get("flag") else "next_if_false"

    node = ConditionalNode(name="cond", condition_fn=condition_fn)

    out_true = node.execute({"flag": True})
    assert out_true == {}
    assert node.next_node_name == "next_if_true"

    out_false = node.execute({"flag": False})
    assert out_false == {}
    assert node.next_node_name == "next_if_false"

