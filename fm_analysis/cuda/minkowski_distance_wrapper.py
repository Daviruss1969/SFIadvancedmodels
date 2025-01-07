from typing import Callable, Literal

import pycuda.driver as cuda
import torch
import numpy as np
import math

from fm_analysis.cuda.cuda_wrapper import CudaWrapper
import SETTINGS

class MinkowskiDistanceWrapper(CudaWrapper):
    _sum_absolute_error_kernel: Callable
    _order_norm: float

    def __init__(self,
                 mode: Literal["ptx", "cubin"]):
        super().__init__(mode)
        self._sum_absolute_error_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/sum_absolute_error.{self._mode}").get_function("sum_absolute_error")
        self._order_norm = np.float32(SETTINGS.MINKOWSKI_DISTANCE_ORDER_NORM)

    def __call__(self,
                golden_tensor: torch.Tensor,
                faulty_tensor: torch.Tensor,
                size: int) -> float:
        # Define result in GPU memory
        result = torch.zeros(1).cuda()

        # Define size of grid/blocks
        threads_per_block = (1024, 1, 1)
        blocks_per_grid = (int(size/threads_per_block[0]) + 1, 1, 1)

        # Call the kernel and get the sum of absolute errors with the order_norm
        self._sum_absolute_error_kernel(
            golden_tensor,
            faulty_tensor,
            result,
            size,
            self._order_norm,
            block=threads_per_block,
            grid=blocks_per_grid
        )

        # Return the minkowski distance in the cpu memory
        return math.pow(result.item(), 1/self._order_norm)
    