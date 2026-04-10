"""
module with node classes used
"""
from __future__ import annotations
from typing import Callable, Any
import copy

NodeFunc = Callable[[dict[str, Any]], dict[str, Any]]


class GraphNode:
    """
    base class for single node in a graph
    """

    def __init__(self, name: str, next_node_name: str | None = None):
        """
        Parameters
        name : str
            name to use for the node
        next_node_name : str
            name of the next node
        """
        self.name = name
        self.last_input = None
        self.last_output = None
        self.next_node_name = next_node_name

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        basic node execution
        Parameters
        ----------
        state : dict[str, Any]
            the current state_dict
        Returns
        -------
        output : dict[str, Any]
            the delta to update the state_dict with
        """
        self.last_input = copy.deepcopy(state)
        output = self._execute_impl(state)
        self.last_output = copy.deepcopy(output)
        return output

    def set_next_node(self, next: str | "GraphNode") -> None:
        """
        set next node
        Parameters
        ----------
        next : str | GraphNode
            either the name of the next node or the node itself.
        """
        if isinstance(next, GraphNode):
            self.next_node_name = next.name
        else:
            self.next_node_name = next

    def _execute_impl(self, state: dict[str, Any]) -> dict[str, Any]:
        # this is expected to be implemented in any class inheriting from GraphNode
        raise NotImplementedError


class FunctionalNode(GraphNode):
    """
    node that applies a function to the state
    """

    def __init__(self, func: NodeFunc, name: str, next_node_name: str | None = None):
        """
        Parameters
        ----------
        func : Callable
            function to take the state dict and return the delta
        name : str
            name of node
        next_node_name : str | None:
            name of next node. When None, this is a terminal node
        """
        super().__init__(name, next_node_name)
        self.func = func

    def _execute_impl(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        implementation for the execute
        Parameters
        ----------
        state : dict[str, Any]
            the state_dict
        Returns
        output : dict[str, Any]:
            the delta to update
        """
        output = self.func(state)
        func_name = getattr(self.func, "__name__", self.func.__class__.__name__)
        # verify that we got a dict out
        if not isinstance(output, dict):
            raise TypeError(
                f"Node '{self.name}' using function "
                f"{func_name} returned {type(output)}, expected dict."
            )
        return output


class ConditionalNode(GraphNode):
    """
    Node to determine which node to go to based on conditional
    """

    def __init__(self, name: str, condition_fn: Callable):
        """
        parameters
        name : str
            name of node
        condition_fn: Callable
            function to determin the next node based on the state"""
        super().__init__(name)
        self.condition_fn = condition_fn
        self.next_node_name = None

    def _execute_impl(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Implementation of execute that just sets the next_node_name based on the state
        Parameters
        ----------
        state : dict[str, Any]
             state_dict when called
        Returns
        -------
        dict[str, Any]
            empty dict, as conditional nodes do not update state
        """
        self.set_next_node(self.condition_fn(state))
        return {}
