"""
module to hold GraphRunner class
which maintains state_dict and executes nodes
"""

import copy
import time
from datetime import datetime
from typing import List, Dict, Set, Literal, Any
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .nodes import GraphNode


class GraphRunner:
    """
    Class to hold a graph and execute from from the start node"""

    def __init__(
        self,
        nodes: List[GraphNode]
        | Set[GraphNode]
        | tuple[GraphNode, ...]
        | Dict[str, GraphNode],
        start_node: str,
        max_node_visits: int | None = None,
        on_max_visits: Literal["error", "exit"] = "exit",
    ):
        """
        Parameters
        ----------
        nodes : List[GraphNode] | Set[GraphNode] | tuple[GraphNode] | Dict[str, Graphnode]
            nodes that are in the graph
        start_node : str
            name of node to start execution
        max_node_visits : int or None
            when not None, the maximum time a node can be visited in a call to execute
        on_max_visits: str
            how to deal when max visits gets exceeded at a node
        """

        def make_nodes_dict(
            nodes: List[GraphNode]
            | Set[GraphNode]
            | tuple[GraphNode, ...]
            | Dict[str, GraphNode],
        ) -> dict[str, GraphNode]:
            """
            takes the nodes parameter and turns it into a dict if it's not already
            note: a dict passed should have the key for a node the same as node.name
            """
            if isinstance(nodes, dict):
                for k, v in nodes.items():
                    if k != v.name:
                        raise ValueError(
                            f"Node dict key {k} does not match node name {v.name}"
                        )
                return nodes
            if type(nodes) in (set, list, tuple):
                nodes_dict = {node.name: node for node in nodes}
                if len(nodes) != len(nodes_dict):
                    raise ValueError("Duplicate node names detected")
                return nodes_dict
            raise TypeError(
                f"Nodes needs to be a list, set, tuple, or dict, got {type(nodes)}"
            )

        self.nodes_dict = make_nodes_dict(nodes)
        self.start_node = start_node
        self.trace_log = []
        self.state_dict = {}
        self.max_node_visits = max_node_visits
        self.on_max_visits = on_max_visits

    def execute(self, input: dict, reset_trace: bool = True)->dict[str, Any]:
        """
        main execution path for a graph
        Parameters
        ----------
        input : dict
            the initial state being passed to the graph
        reset_trace : bool
            whether or not to reset the trace_log on execute

        Returns
        -------
        dict
            keys: state_dict: the state_dict after execution, trace_log: the trace log,
            token_usage: the number of tokens used, terminated_early: if the graph had to
            terminate early
        """
        self.out_tokens = 0
        self.in_tokens = 0
        self.node_tracker = {}
        if not isinstance(input, dict):
            raise TypeError("Input must be a dict")
        if reset_trace:
            self.trace_log: list[dict[str, Any]] = []
        # start by copying the input into the state_dict
        self.state_dict = copy.deepcopy(input)
        current_node_name = self.start_node
        delta = {}
        terminated_early = False
        # looping through the nodes
        while current_node_name:
            # track the visit to the current node
            self.node_tracker[current_node_name] = (
                self.node_tracker.get(current_node_name, 0) + 1
            )
            # if there is a max node visit limit, check if limit was exceeded
            if self.max_node_visits is not None:
                if self.node_tracker[current_node_name] > self.max_node_visits:
                    # when exceeded, behavior depends on on_max_visits
                    if self.on_max_visits == "error":
                        # Here we throw an exception
                        raise RuntimeError(
                            f"Node {current_node_name} exceeded max visit limit "
                            f"(visited {self.node_tracker[current_node_name]}, "
                            f"max allowed {self.max_node_visits})"
                        )
                    else:
                        # in this case we simply set a flag that we terminated early and
                        # break
                        terminated_early = True
                        break
            # pass the state to the node, execute, and get the delta back
            node = self.nodes_dict[current_node_name]
            start_time = time.time()
            delta = node.execute(self.state_dict)
            end_time = time.time()
            execution_time = end_time - start_time
            # update the trace_log
            self.trace_log.append(
                {
                    "step_num": len(self.trace_log) + 1,
                    "name": node.name,
                    "node_input": node.last_input,
                    "node_output": node.last_output,
                    "node_type": type(node).__name__,
                    "next_node_name": node.next_node_name,
                    "execution_time": execution_time,
                    "timestamp": time.time(),
                }
            )
            self.state_dict.update(delta)
            # if tokens were used, track those
            usage = delta.get("usage") or {}
            self.in_tokens += usage.get("prompt_tokens", 0)
            self.out_tokens += usage.get("completion_tokens", 0)
            current_node_name = node.next_node_name
        return {
            "state_dict": self.state_dict,
            "trace_log": copy.deepcopy(self.trace_log),
            "token_usage": self.get_token_usage(),
            "terminated_early": terminated_early,
        }

    def get_token_usage(self) -> dict[str, Any]:
        """
        return current token usage by the graph
        note: this does not reset at each execute call
        """
        return {
            "in_tokens": self.in_tokens,
            "out_tokens": self.out_tokens,
            "total_tokens": self.in_tokens + self.out_tokens,
        }

    def print_trace(self):
        """
        print through the trace
        """
        for step in self.trace_log:
            step_num = step.get("step_num")
            name = step.get("name")
            node_input = step.get("node_input")
            node_output = step.get("node_output")
            node_type = step.get("node_type")
            next_node = step.get("next_node_name")
            execution_time = step.get("execution_time")
            ts = step.get("timestamp")
            timestamp = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            print(f"Step {step_num}")
            print(f"  Node: {name}")
            print(f"  Node type: {node_type}")
            print(f"  Input: {node_input}")
            print(f"  Output: {node_output}")
            print(f"  Next: {next_node if next_node is not None else 'TERMINAL'}")
            print(f"  Execution time: {execution_time * 1000:.1f} ms")
            print(f"  Timestamp: {timestamp}")
            print()

    def matplotlib_trace(self, branch_color_map: dict | None = None):
        """
        Draw a simple linear trace of the executed graph using matplotlib.
        Works with both linear and conditional nodes.
        """
        if branch_color_map is None:
            branch_color_map = {}
        steps = self.trace_log
        n = len(steps)

        fig, ax = plt.subplots(figsize=(8, n * 1.2))
        ax.set_xlim(0, 2)
        ax.set_ylim(0, n)
        ax.axis("off")

        for i, step in enumerate(steps):
            y = n - i - 0.5  # vertical position
            x = 1

            # Draw node as a rectangle
            rect = Rectangle(
                (0.5, y - 0.25), 1, 0.5, facecolor="lightblue", edgecolor="black"
            )
            ax.add_patch(rect)

            # Node label: name and output summary
            output_str = str(step["node_output"])
            if len(output_str) > 30:
                output_str = output_str[:27] + "..."
            label = f"{step['name']}\n{output_str}"
            ax.text(x, y, label, ha="center", va="center", fontsize=10)

            # Draw arrow to next node if exists
            if i < n - 1:
                next_node = steps[i]["next_node_name"]
                arrow_color = branch_color_map.get(next_node, "gray")
                next_y = n - i - 1 - 0.5
                ax.annotate(
                    "",
                    xy=(1, next_y + 0.25),
                    xytext=(1, y - 0.25),
                    arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5),
                )

        plt.tight_layout()
        plt.show()
