from typing import Callable, Literal

import pycuda.driver as cuda
import torch
import numpy as np

from fm_analysis.cuda.cuda_wrapper import CudaWrapper

class ActivationSensitivityWrapper(CudaWrapper):
    def __init__(self,
                 mode: Literal["ptx", "cubin"]):
        super().__init__(mode)
        self._subtract_vector_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/subtract_vector.{self._mode}").get_function("subtract_vector")
        self._euclidian_norm_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/euclidian_norm.{self._mode}").get_function("euclidian_norm")

    def __call__(self,
                golden_tensor: torch.Tensor,
                faulty_tensor: torch.Tensor,
                size: int) -> float:
        # Define results in GPU memory
        subtract_result = torch.zeros(size).cuda()
        euclidian_norm_perturbated_result = torch.zeros(1).cuda()
        euclidian_norm_golden_result = torch.zeros(1).cuda()

        # Define size of grid/blocks
        threads_per_block = (1024, 1, 1)
        blocks_per_grid = (int(size/threads_per_block[0]) + 1, 1, 1)

        # Call the kernel and get the subtraction of fms
        self._subtract_vector_kernel(
            golden_tensor,
            faulty_tensor,
            subtract_result,
            size,
            block=threads_per_block,
            grid=blocks_per_grid
        )

        # Call the norm kernel on subtract result and perform the square root on cpu
        self._euclidian_norm_kernel(
            subtract_result,
            euclidian_norm_perturbated_result,
            size,
            block=threads_per_block,
            grid=blocks_per_grid
        )
        euclidian_norm_perturbated = np.sqrt(euclidian_norm_perturbated_result.item())

        # Call the norm kernel on golden fm and perform the square root on cpu
        self._euclidian_norm_kernel(
            golden_tensor,
            euclidian_norm_golden_result,
            size,
            block=threads_per_block,
            grid=blocks_per_grid
        )
        euclidian_norm_golden = np.sqrt(euclidian_norm_golden_result.item())

        # Return  the activation sensitivity in the cpu
        return euclidian_norm_perturbated/euclidian_norm_golden
    