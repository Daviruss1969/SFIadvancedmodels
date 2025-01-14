from typing import Callable, Literal

import pycuda.driver as cuda
import torch
import numpy as np

from fm_analysis.cuda.cuda_wrapper import CudaWrapper
import SETTINGS

class ActivationRangeWrapper(CudaWrapper):
    def __init__(self,
                 mode: Literal["ptx", "cubin"]):
        super().__init__(mode)
        self._maximum_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/maximum.{self._mode}").get_function("maximum")
        self._minimum_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/minimum.{self._mode}").get_function("minimum")
        self._subtract_vector_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/subtract_vector.{self._mode}").get_function("subtract_vector")

    def __call__(self,
                golden_tensor: torch.Tensor,
                faulty_tensor: torch.Tensor,
                batch_size: int,
                size: int) -> float:
        # Define results in GPU memory
        max_results = torch.zeros(batch_size).cuda()
        min_results = torch.zeros(batch_size).cuda()
        results = torch.zeros(batch_size).cuda()

        # Define size of grid/blocks
        threads_per_block = (1024, 1, 1)
        blocks_per_grid_size = (int(size/threads_per_block[0]) + 1, int(batch_size), 1)
        blocks_per_grid_unrolled = (int((batch_size*size)/threads_per_block[0]) + 1, 1, 1)

        tensor = faulty_tensor if SETTINGS.SINGLE_INPUT_TENSOR == "faulty" else golden_tensor

        # Call the kernel and get the maximums
        self._maximum_kernel(
            tensor,
            max_results,
            size,
            block=threads_per_block,
            grid=blocks_per_grid_size
        )

        # Call the kernel and get the minimums
        self._minimum_kernel(
            tensor,
            min_results,
            size,
            block=threads_per_block,
            grid=blocks_per_grid_size
        )

        # Subtract the results
        self._subtract_vector_kernel(
            max_results,
            min_results,
            results,
            size*batch_size,
            block=threads_per_block,
            grid=blocks_per_grid_unrolled
        )

        # Return the activation ranges
        return results
    