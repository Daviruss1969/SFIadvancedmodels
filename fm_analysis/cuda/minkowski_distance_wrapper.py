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
        self._power_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/power.{self._mode}").get_function("power")
        self._order_norm = np.float32(SETTINGS.MINKOWSKI_DISTANCE_ORDER_NORM)

    def __call__(self,
                golden_tensor: torch.Tensor,
                faulty_tensor: torch.Tensor,
                batch_size: int,
                size: int) -> float:
        # Define result in GPU memory
        results = torch.zeros(batch_size).cuda()

        # Define size of grid/blocks
        threads_per_block = (1024, 1, 1)
        blocks_per_grid_size = (int(size/threads_per_block[0]) + 1, int(batch_size), 1)
        blocks_per_grid_batch = (int(batch_size/threads_per_block[0]) + 1, 1, 1)

        # Call the kernel and get the sum of absolute errors with the order_norm
        self._sum_absolute_error_kernel(
            golden_tensor,
            faulty_tensor,
            results,
            size,
            self._order_norm,
            block=threads_per_block,
            grid=blocks_per_grid_size
        )

        # Perform the power to obtain the minkowski distance
        self._power_kernel(
            results,
            np.float32(1/self._order_norm),
            batch_size,
            block=threads_per_block,
            grid=blocks_per_grid_batch
        )

        # Return the minkowski distance
        return results
    