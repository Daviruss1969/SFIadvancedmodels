from typing import Callable, Literal

import pycuda.driver as cuda
import torch
import numpy as np

from fm_analysis.cuda.cuda_wrapper import CudaWrapper
import SETTINGS


class EntropyWrapper(CudaWrapper):
    _entropy_kernel: Callable

    def __init__(self,
                 mode: Literal["ptx", "cubin"]):
        super().__init__(mode)
        self._entropy_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/entropy.{self._mode}").get_function("entropy")
        self._sum_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/sum.{self._mode}").get_function("sum")

    def __call__(self,
                golden_tensor: torch.Tensor,
                faulty_tensor: torch.Tensor,
                batch_size: int,
                size: int) -> float:
        # Define result in GPU memory
        sum_results = torch.zeros(batch_size).cuda()
        entropy_results = torch.zeros(batch_size).cuda()

        # Define size of grid/blocks
        threads_per_block = (1024, 1, 1)
        blocks_per_grid_size = (int(size/threads_per_block[0]) + 1, int(batch_size), 1)

        # Call the kernel and get the sum of all the values
        tensor = faulty_tensor if SETTINGS.SINGLE_INPUT_TENSOR == "faulty" else golden_tensor
        self._sum_kernel(
            tensor,
            sum_results,
            size,
            block=threads_per_block,
            grid=blocks_per_grid_size
        )

        # Perform the calcul of the entropy
        self._entropy_kernel(
            tensor,
            entropy_results,
            sum_results,
            size,
            block=threads_per_block,
            grid=blocks_per_grid_size
        )

        # Return the entropies
        return entropy_results
