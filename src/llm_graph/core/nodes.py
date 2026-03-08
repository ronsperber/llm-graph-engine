from typing import Callable, Any
import copy

class GraphNode:
    """
    base class for single node in a graph
    """
    def __init__(self, name: str, next_node_name : str | None = None):
        self.name = name
        self.last_input = None
        self.last_output = None
        self.next_node_name = next_node_name

    def execute(self, state:dict[str, Any]) -> dict[str, Any]:
        self.last_input = copy.deepcopy(state)
        output = self._execute_impl(state)
        self.last_output = copy.deepcopy(output)
        return output

    def _execute_impl(self, state:dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
    
class FunctionalNode(GraphNode):
    """
    node that applies a function to the state
    """

    def __init__(
            self,
            func: Callable,
            name: str,
            next_node_name: str|None = None
            ):
        super().__init__(name, next_node_name)
        self.func = func
    
    def _execute_impl(self, state: dict[str, Any]) -> dict[str, Any]:
            output = self.func(state)
            func_name = getattr(self.func, "__name__", self.func.__class__.__name__)
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
    def __init__(
            self,
            name: str,
            condition_fn:Callable
            ):
        super().__init__(name)
        self.condition_fn = condition_fn
        self.next_node_name = None
    def _execute_impl(self, state: dict[str, Any]) -> dict[str, Any]:
        self.next_node_name = self.condition_fn(state)
        return {}