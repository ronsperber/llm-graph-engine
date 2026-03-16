"""
module to hold SessionRunner. SessionRunner class is meant to hold a GraphRunner
for multiple executions and keep any state entries that need to be held across
multiple runs
"""

from typing import Any
from copy import deepcopy
from .graphrunner import GraphRunner


class SessionRunner:
    def __init__(self, graph: GraphRunner, session_keys: list | None = None):
        """
        Parameters
        ----------
        graph : GraphRunner
            the graphrunner to be used
        session_keys  :(Optional) list
            when not None the list of keys to be kept across executes
        """
        self.graph = graph
        if session_keys is None:
            session_keys = []
        self.session_keys = session_keys
        # set the session_dict and logs to empty dict/list respectively
        self.session_dict = {}
        self.trace_logs = []
        # set initial token usage to 0
        self.out_tokens = 0
        self.in_tokens = 0

    def execute(self, input: dict[str, Any]) -> dict[str, Any]:
        """
        main execution function for the runner
        Parameters
        ----------
        input : dict[str, Any]
            input to the graph
        Returns
        -------
        response : dict[str, Any]
            the response from the graph
        """
        # combine the existing session_dict with the input to pass to the graph
        graph_input = self.session_dict | input
        response = self.graph.execute(graph_input)
        # get the state dict and trace_log from the response
        state_dict = response.get("state_dict") or {}
        trace_log = deepcopy(response.get("trace_log") or [])
        # add the trace_log to the list of trace_logs
        self.trace_logs.append(trace_log)
        # add usage
        self.out_tokens += self.graph.out_tokens
        self.in_tokens += self.graph.in_tokens
        # update the session_dict with any key-value pairs returned where the key
        # is a session key
        self.session_dict.update(
            {k: state_dict[k] for k in self.session_keys if k in state_dict}
        )
        return response
