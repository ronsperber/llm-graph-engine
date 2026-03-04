import pytest
from llm_graph.core.nodes import ConditionalNode, FunctionalNode
from llm_graph.core.runner import GraphRunner

def test_max_node_visits_cycle():

    node = FunctionalNode(
        func=lambda s: s,
        name="A",
        next_node_name="A"
    )

    runner = GraphRunner(
        nodes=[node],
        start_node="A",
        max_node_visits=1
    )

    with pytest.raises(RuntimeError):
        runner.execute({"user_query": "test"})

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
    runner.execute({"user_query": "query2"})