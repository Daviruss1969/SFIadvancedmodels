from typing import Callable, Literal
from abc import abstractmethod

import torch

class CudaWrapper:
    _cuda_function: Callable
    _mode: Literal["ptx", "cubin"]

    @abstractmethod
    def __init__(self,
                 mode: Literal["ptx", "cubin"]):
        self._mode = mode

    @abstractmethod
    def __call__(self,
                 golden_tensor: torch.Tensor,
                 faulty_tensor: torch.Tensor,
                 size: int) -> float:
        pass