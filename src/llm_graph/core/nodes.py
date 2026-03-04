from typing import Callable
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

    def execute(self, state:dict) -> dict:
        self.last_input = copy.deepcopy(state)
        output = self._execute_impl(state)
        self.last_output = copy.deepcopy(output)
        return output

    def _execute_impl(self, state):
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
        self.last_input = None
        self.last_output = None
    

    def _execute_impl(self, state: dict):
        return self.func(state)
        
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
    def _execute_impl(self, state: dict):
        self.next_node_name = self.condition_fn(state)
        return {}