from typing import Callable, Literal

import pycuda.driver as cuda
import torch
import numpy as np

from fm_analysis.cuda.cuda_wrapper import CudaWrapper
import SETTINGS

class SparsityRatioWrapper(CudaWrapper):
    def __init__(self,
                 mode: Literal["ptx", "cubin"]):
        super().__init__(mode)
        self._zero_activations_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/zero_activations.{self._mode}").get_function("zero_activations")
        self._divide_vector_scalar_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/divide_vector_scalar.{self._mode}").get_function("divide_vector_scalar")

    def __call__(self,
                golden_tensor: torch.Tensor,
                faulty_tensor: torch.Tensor,
                batch_size: int,
                size: int) -> float:
        # Define results in GPU memory
        results = torch.zeros(batch_size).cuda()

        # Define size of grid/blocks
        # Define size of grid/blocks
        threads_per_block = (1024, 1, 1)
        blocks_per_grid_size = (int(size/threads_per_block[0]) + 1, int(batch_size), 1)
        blocks_per_grid_batch = (int(batch_size/threads_per_block[0]) + 1, 1, 1)

        # Call the kernel and get the number of zero activations
        tensor = faulty_tensor if SETTINGS.SPARSITY_RATIO_TENSOR == "faulty" else golden_tensor
        self._zero_activations_kernel(
            tensor,
            results,
            size,
            block=threads_per_block,
            grid=blocks_per_grid_size
        )

        # Get the sparsity ratios for each inputs
        self._divide_vector_scalar_kernel(
            results,
            size,
            batch_size,
            block=threads_per_block,
            grid=blocks_per_grid_batch
        )

        # Return the sparsity ratios
        return results
    