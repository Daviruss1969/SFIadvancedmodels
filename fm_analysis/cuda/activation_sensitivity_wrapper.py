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
        self._square_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/square.{self._mode}").get_function("square")
        self._divide_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/divide.{self._mode}").get_function("divide")

    def __call__(self,
                golden_tensor: torch.Tensor,
                faulty_tensor: torch.Tensor,
                batch_size: int,
                size: int) -> float:
        # Define results in GPU memory
        subtract_results = torch.zeros(batch_size, size).cuda()
        euclidian_norm_perturbated_results = torch.zeros(batch_size).cuda()
        euclidian_norm_golden_results = torch.zeros(batch_size).cuda()
        activation_sensitivity = torch.zeros(batch_size).cuda()

        # Define size of grid/blocks
        threads_per_block = (1024, 1, 1)
        blocks_per_grid_size = (int(size/threads_per_block[0]) + 1, int(batch_size), 1)
        blocks_per_grid_batch = (int(batch_size/threads_per_block[0]) + 1, 1, 1)
        blocks_per_grid_unrolled = (int((batch_size*size)/threads_per_block[0]) + 1, 1, 1)

        # Call the kernel and get the subtraction of fms
        self._subtract_vector_kernel(
            faulty_tensor,
            golden_tensor,
            subtract_results,
            size*batch_size,
            block=threads_per_block,
            grid=blocks_per_grid_unrolled
        )

        # Call the euclidian norm kernel, perform the square root in another kernel
        self._euclidian_norm_kernel(
            golden_tensor,
            euclidian_norm_golden_results,
            size,
            block=threads_per_block,
            grid=blocks_per_grid_size
        )
        self._square_kernel(
            euclidian_norm_golden_results,
            batch_size,
            block=threads_per_block,
            grid=blocks_per_grid_batch
        )

        # Call the euclidian norm kernel, perform the square root in another kernel
        self._euclidian_norm_kernel(
            subtract_results,
            euclidian_norm_perturbated_results,
            size,
            block=threads_per_block,
            grid=blocks_per_grid_size
        )
        self._square_kernel(
            euclidian_norm_perturbated_results,
            batch_size,
            block=threads_per_block,
            grid=blocks_per_grid_batch
        )

        # Call the divide kernel to get the activation sensitivity
        self._divide_kernel(
            euclidian_norm_perturbated_results,
            euclidian_norm_golden_results,
            activation_sensitivity,
            batch_size,
            block=threads_per_block,
            grid=blocks_per_grid_batch
        )

        # Return  the activation sensitivity
        return activation_sensitivity
    