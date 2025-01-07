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

    def __call__(self,
                golden_tensor: torch.Tensor,
                faulty_tensor: torch.Tensor,
                size: int) -> float:
        # Define result in GPU memory as an int32
        result = torch.zeros(1, dtype=torch.int32).cuda()

        # Define size of grid/blocks
        threads_per_block = (1024, 1, 1)
        blocks_per_grid = (int(size/threads_per_block[0]) + 1, 1, 1)

        # Call the kernel and get the number of zero activations
        tensor = faulty_tensor if SETTINGS.SPARSITY_RATIO_TENSOR == "faulty" else golden_tensor
        self._zero_activations_kernel(
            tensor,
            result,
            size,
            block=threads_per_block,
            grid=blocks_per_grid
        )

        # Return the sparsity ratio in the cpu memory
        return result.item()/size
    