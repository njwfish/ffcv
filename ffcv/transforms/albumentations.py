from typing import Callable, Optional, Tuple
from dataclasses import replace

import albumentations as A

from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.compiler import Compiler


class AlbumentationsWrapper(Operation):
    """Transform using the given albumentations.BasicTransform
    Parameters
    ----------
    transform: albumentations.BasicTransform
        The transform for transformation
    """

    def __init__(self, field_name: str, transform: A.BasicTransform):
        super().__init__()
        self.field_name = field_name
        self.transform = transform

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()

        def apply_transform(inp, _):
            for i in my_range(inp.shape[0]):
                inp[i] = self.transform(**{self.field_name: inp[i]})[self.field_name]
            return inp

        return apply_transform

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        new_shape = [
            getattr(self.transform, "height", previous_state.shape[0]),
            getattr(self.transform, "width", previous_state.shape[1]),
            *previous_state.shape[2:],
        ]
        new_state = replace(previous_state, jit_mode=False, shape=new_shape)
        return (new_state, None)
