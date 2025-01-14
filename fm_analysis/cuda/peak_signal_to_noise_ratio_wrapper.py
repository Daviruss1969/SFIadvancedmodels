from typing import Callable, Literal

import pycuda.driver as cuda
import torch
import numpy as np
import SETTINGS

from fm_analysis.cuda.cuda_wrapper import CudaWrapper

class PeakSignalToNoiseRatioWrapper(CudaWrapper):
    _sum_absolute_error_kernel: Callable
    def __init__(self,
                 mode: Literal["ptx", "cubin"]):
        super().__init__(mode)
        self._sum_absolute_error_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/sum_absolute_error.{self._mode}").get_function("sum_absolute_error")
        self._divide_vector_scalar_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/divide_vector_scalar.{self._mode}").get_function("divide_vector_scalar")
        self._maximum_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/maximum.{self._mode}").get_function("maximum")
        self._power_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/power.{self._mode}").get_function("power")
        self._divide_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/divide.{self._mode}").get_function("divide")
        self._logarithm10_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/logarithm10.{self._mode}").get_function("logarithm10")



    def __call__(self,
                golden_tensor: torch.Tensor,
                faulty_tensor: torch.Tensor,
                batch_size: int,
                size: int) -> float:
        # Define result in GPU memory
        mean_absolute_error_results = torch.zeros(batch_size).cuda()
        max_results = torch.zeros(batch_size).cuda()
        divide_results = torch.zeros(batch_size).cuda()


        # Define size of grid/blocks
        threads_per_block = (1024, 1, 1)
        blocks_per_grid_size = (int(size/threads_per_block[0]) + 1, int(batch_size), 1)
        blocks_per_grid_batch = (int(batch_size/threads_per_block[0]) + 1, 1, 1)

        tensor = faulty_tensor if SETTINGS.SINGLE_INPUT_TENSOR == "faulty" else golden_tensor

        # Call the kernel and get the sqaure of absolute errors
        self._sum_absolute_error_kernel(
            golden_tensor,
            faulty_tensor,
            mean_absolute_error_results,
            size,
            np.float32(2),
            block=threads_per_block,
            grid=blocks_per_grid_size
        )

        # Perform the division to get the mean on each inputs
        self._divide_vector_scalar_kernel(
            mean_absolute_error_results,
            size,
            batch_size,
            block=threads_per_block,
            grid=blocks_per_grid_batch
        )

        # Compute the maximum possible values
        self._maximum_kernel(
            tensor,
            max_results,
            size,
            block=threads_per_block,
            grid=blocks_per_grid_size
        )

        # Compute the power of maximums values
        self._power_kernel(
            max_results,
            np.float32(2),
            batch_size,
            block=threads_per_block,
            grid=blocks_per_grid_batch
        )

        # Divide both results
        self._divide_kernel(
            max_results,
            mean_absolute_error_results,
            divide_results,
            batch_size,
            block=threads_per_block,
            grid=blocks_per_grid_batch
        )

        # Perform 10 * log10(results)
        self._logarithm10_kernel(            
            divide_results,
            batch_size,
            np.float32(10),
            block=threads_per_block,
            grid=blocks_per_grid_batch
        )

        # Return the peak signal to noise ratios
        return divide_results
    