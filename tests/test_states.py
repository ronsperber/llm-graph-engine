import pytest
from llm_graph.core.nodes import ConditionalNode, FunctionalNode
from llm_graph.core.graphrunner import GraphRunner


def test_propogation():
    f1 = lambda x: {"a": "node_1"}
    f2 = lambda x: {"b": "node_2"}
    f3 = lambda x: {"c": "node_3"}
    node1 = FunctionalNode(func=f1, name="node1", next_node_name="node2")
    node2 = FunctionalNode(func=f2, name="node2", next_node_name="node3")
    node3 = FunctionalNode(func=f3, name="node3")

    runner = GraphRunner(nodes=[node1, node2, node3], start_node="node1")
    result = runner.execute({"input": "start"})
    state = result["state_dict"]
    assert state["input"] == "start"
    assert state["a"] == "node_1"
    assert state["b"] == "node_2"
    assert state["c"] == "node_3"
    assert set(state.keys()) == {"input", "a", "b", "c"}
