from typing import Any
from copy import deepcopy
from .graphrunner import GraphRunner

class SessionRunner:
    def __init__(
            self,
            graphrunner: GraphRunner,
            session_keys: list | None = None
    ):
        self.graphrunner = graphrunner
        if session_keys is None:
            session_keys = []
        self.session_keys = session_keys
        self.session_dict = {}
        self.trace_logs = []

    def execute(
            self,
            input : dict[str, Any]
    ):
        graph_input = self.session_dict | input
        response = self.graphrunner.execute(graph_input)
        state_dict = response.get("state_dict") or {}
        trace_log = deepcopy(response.get("trace_log") or [])
        self.trace_logs.append(trace_log)
        self.session_dict.update(
            {k: state_dict[k] for k in self.session_keys if k in state_dict}
        )
        return response