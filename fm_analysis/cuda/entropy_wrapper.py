from typing import Callable, Literal

import pycuda.driver as cuda
import torch
import numpy as np

from fm_analysis.cuda.cuda_wrapper import CudaWrapper


class EntropyWrapper(CudaWrapper):
    _entropy_kernel: Callable

    def __init__(self,
                 mode: Literal["ptx", "cubin"]):
        super().__init__(mode)
        self._entropy_kernel = cuda.module_from_file(
            f"fm_analysis/cuda/{self._mode}/entropy.{self._mode}").get_function("entropy")
        self._sum_kernel = cuda.module_from_file(
            f"fm_analysis/cuda/{self._mode}/sum.{self._mode}").get_function("sum")

    def __call__(self,
                 tensor: torch.Tensor,
                 batch_size: int,
                 size: int) -> float:
        # Define result in GPU memory
        sum_value = 0
        results = torch.zeros(batch_size).cuda()

        # Define size of grid/blocks
        threads_per_block = (1024, 1, 1)
        blocks_per_grid = (int(size / threads_per_block[0]) + 1, int(batch_size), 1)

        # Call the kernel and get the sum of all the values
        self._sum_kernel(
            tensor,
            sum,
            size,
            block=threads_per_block,
            grid=blocks_per_grid
        )

        # Define size of grid/blocks
        threads_per_block = (1024, 1, 1)
        blocks_per_grid = (int(batch_size / threads_per_block[0]) + 1, 1, 1)

        # Perform the calcul of the entropy
        self._entropy_kernel(
            tensor,
            results,
            batch_size,
            sum_value,
            block=threads_per_block,
            grid=blocks_per_grid
        )

        # Return the mean absolute error
        return results
