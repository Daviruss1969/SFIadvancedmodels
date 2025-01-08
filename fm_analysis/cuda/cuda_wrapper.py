from typing import Literal
from abc import abstractmethod

import torch

class CudaWrapper:
    _mode: Literal["ptx", "cubin"]

    @abstractmethod
    def __init__(self,
                 mode: Literal["ptx", "cubin"]):
        self._mode = mode

    @abstractmethod
    def __call__(self,
                 golden_tensor: torch.Tensor,
                 faulty_tensor: torch.Tensor,
                 batch_size: int,
                 size: int) -> float:
        pass