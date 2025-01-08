from typing import Callable, Literal

import pycuda.driver as cuda
import torch
import numpy as np

from fm_analysis.cuda.cuda_wrapper import CudaWrapper

class MeanAbsoluteErrorWrapper(CudaWrapper):
    _sum_absolute_error_kernel: Callable
    def __init__(self,
                 mode: Literal["ptx", "cubin"]):
        super().__init__(mode)
        self._sum_absolute_error_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/sum_absolute_error.{self._mode}").get_function("sum_absolute_error")
        self._divide_vector_scalar_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/divide_vector_scalar.{self._mode}").get_function("divide_vector_scalar")

    def __call__(self,
                golden_tensor: torch.Tensor,
                faulty_tensor: torch.Tensor,
                batch_size: int,
                size: int) -> float:
        # Define result in GPU memory
        results = torch.zeros(batch_size).cuda()

        # Define size of grid/blocks
        threads_per_block = (1024, 1, 1)
        blocks_per_grid = (int(size/threads_per_block[0]) + 1, int(batch_size), 1)

        # Call the kernel and get the sum of absolute errors
        self._sum_absolute_error_kernel(
            golden_tensor,
            faulty_tensor,
            results,
            size,
            np.float32(1),
            block=threads_per_block,
            grid=blocks_per_grid
        )

        # Define size of grid/blocks
        threads_per_block = (1024, 1, 1)
        blocks_per_grid = (int(batch_size/threads_per_block[0]) + 1, 1, 1)

        # Perform the division to get the mean on each inputs
        self._divide_vector_scalar_kernel(
            results,
            size,
            batch_size,
            block=threads_per_block,
            grid=blocks_per_grid
        )

        # Return the mean absolute error
        return results
    