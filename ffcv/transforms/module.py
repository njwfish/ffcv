"""
Wrapper for a torch.nn.Module
"""
import torch as ch
from torch import Tensor
from numpy.random import permutation, rand
from typing import Callable, Optional, Tuple
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State

class ModuleWrapper(Operation):
    """Transform using the given torch.nn.Module

    Parameters
    ----------
    module: torch.nn.Module
        The module for transformation
    """
    def __init__(self, module: ch.nn.Module):
        super().__init__()
        self.module: ch.nn.Module = module

    def generate_code(self) -> Callable:
        def apply_module(inp, _) -> Tensor:
            res: Tensor = self.module(inp)
            return res

        return apply_module

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return previous_state, None
