import copy
from typing import List, Dict, Set, Literal
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .nodes import GraphNode


class GraphRunner:
    """
    Class to hold a graph and execute from from the start node"""
    def __init__(
            self,
            nodes: List[GraphNode] | Set[GraphNode] | tuple[GraphNode,...] | Dict[str, GraphNode],
            start_node: str,
            max_node_visits: int | None = None,
            on_max_visits : Literal["error", "exit"] = "exit"
            ):
        
        def make_nodes_dict(
                nodes:  List[GraphNode] | Set[GraphNode] | tuple[GraphNode,...] | Dict[str, GraphNode]
        ) -> dict[str, GraphNode]:
            if isinstance(nodes, dict):
                return nodes
            elif type(nodes) in (set, list, tuple):
                nodes_dict = {node.name : node for node in nodes}
                return nodes_dict
            raise TypeError(f"Nodes needs to be a list, set, tuple, or dict, got {type(nodes)}")
        self.nodes_dict = make_nodes_dict(nodes)
        self.start_node = start_node
        self.trace_log = []
        self.state_dict = {}
        self.max_node_visits = max_node_visits
        self.on_max_visits = on_max_visits
    
    def execute(self, input: dict, reset_trace: bool = True):
        self.out_tokens = 0
        self.in_tokens = 0
        self.node_tracker = {}
        if not isinstance(input, dict):
            raise TypeError("Input must be a dict")
        if reset_trace:
            self.trace_log = []
        self.state_dict = copy.deepcopy(input)
        current_node_name = self.start_node
        delta = {}
        terminated_early = False
        while current_node_name:
            self.node_tracker[current_node_name] = self.node_tracker.get(current_node_name, 0) + 1
            if self.max_node_visits is not None:
                if self.node_tracker[current_node_name] > self.max_node_visits:
                    if self.on_max_visits == "error":
                        raise RuntimeError(
                            f"Node {current_node_name} exceeded max visit limit "
                            f"(visited {self.node_tracker[current_node_name]}, "
                            f"max allowed {self.max_node_visits})"
                        )
                    else:
                        terminated_early = True
                        break
            node = self.nodes_dict[current_node_name]
            delta = node.execute(self.state_dict)
            self.trace_log.append(
                {
                    "step_num" : len(self.trace_log) + 1,
                    "name" : node.name,
                    "node_input": node.last_input,
                    "node_output": node.last_output,
                    "next_node_name": node.next_node_name
                }
            )
            self.state_dict.update(delta)
            usage = delta.get('usage') or {}
            self.in_tokens += usage.get('prompt_tokens', 0)
            self.out_tokens +=usage.get('completion_tokens', 0)
            current_node_name = node.next_node_name
        return {
            "state_dict": self.state_dict,
            "trace_log": copy.deepcopy(self.trace_log),
            "token_usage": self.get_token_usage(),
            "terminated_early": terminated_early
            }
    
    def get_token_usage(self):
        return {
            "in_tokens": self.in_tokens,
            "out_tokens": self.out_tokens,
            "total_tokens": self.in_tokens + self.out_tokens
        }
    
    def print_trace(self):
        for step in self.trace_log:
            step_num = step.get("step_num")
            name = step.get("name")
            node_input = step.get("node_input")
            node_output = step.get("node_output")
            next_node = step.get("next_node_name")
            print(f"Step {step_num}")
            print(f"  Node: {name}")
            print(f"  Input: {node_input}")
            print(f"  Output: {node_output}")
            print(f"  Next: {next_node if next_node is not None else 'TERMINAL'}")
            print()

    def matplotlib_trace(self, branch_color_map: dict|None = None):
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
        ax.axis('off')
    
        for i, step in enumerate(steps):
            y = n - i - 0.5  # vertical position
            x = 1
        
            # Draw node as a rectangle
            rect = Rectangle((0.5, y - 0.25), 1, 0.5, facecolor='lightblue', edgecolor='black')
            ax.add_patch(rect)
        
            # Node label: name and output summary
            output_str = str(step['node_output'])
            if len(output_str) > 30:
                output_str = output_str[:27] + "..."
            label = f"{step['name']}\n{output_str}"
            ax.text(x, y, label, ha='center', va='center', fontsize=10)
        
            # Draw arrow to next node if exists
            if i < n - 1:
                next_node = steps[i]['next_node_name']
                arrow_color = branch_color_map.get(next_node, 'gray')
                next_y = n - i - 1 - 0.5
                ax.annotate(
                    '', xy=(1, next_y + 0.25), xytext=(1, y - 0.25),
                    arrowprops=dict(arrowstyle="->", color= arrow_color, lw=1.5)
                )
    
        plt.tight_layout()
        plt.show()