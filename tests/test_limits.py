import pytest
from llm_graph.core.nodes import ConditionalNode, FunctionalNode
from llm_graph.core.graphrunner import GraphRunner

def test_max_node_visits_error():

    node = FunctionalNode(
        func=lambda s: s,
        name="A",
        next_node_name="A"
    )

    runner = GraphRunner(
        nodes=[node],
        start_node="A",
        max_node_visits=1,
        on_max_visits="error"
    )

    with pytest.raises(RuntimeError):
        runner.execute({"user_query": "test"})

def test_max_node_visits_break():
    node = FunctionalNode(
        func=lambda s: s, 
        name="A",
        next_node_name="A"
    )

    runner = GraphRunner(
        nodes=[node],
        start_node="A",
        max_node_visits=1,
    )

    response = runner.execute({"user_query" : "test"})
    assert response["terminated_early"]

def test_visit_tracker_reset():

    node = FunctionalNode(
        func=lambda s: s,
        name="A",
        next_node_name=None
    )

    runner = GraphRunner(
        nodes=[node],
        start_node="A",
        max_node_visits=1
    )

    runner.execute({"user_query": "query1"})
    final_resp = runner.execute({"user_query": "query2"})
    assert not final_resp['terminated_early']
