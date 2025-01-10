from typing import Callable, Literal

import pycuda.driver as cuda
import torch
import numpy as np
import math

from fm_analysis.cuda.cuda_wrapper import CudaWrapper

class StructuralSimilarityIndexWrapper(CudaWrapper):
    def __init__(self,
                 mode: Literal["ptx", "cubin"]):
        super().__init__(mode)
        self._sum_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/sum.{self._mode}").get_function("sum")
        self._divide_vector_scalar_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/divide_vector_scalar.{self._mode}").get_function("divide_vector_scalar")
        self._variance_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/variance.{self._mode}").get_function("variance")
        self._covariance_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/covariance.{self._mode}").get_function("covariance")
        self._structural_similarity_index_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/structural_similarity_index.{self._mode}").get_function("structural_similarity_index")

    def __call__(self,
                golden_tensor: torch.Tensor,
                faulty_tensor: torch.Tensor,
                batch_size: int,
                size: int) -> float:
        # Define results in GPU memory
        results_mean_golden = torch.zeros(batch_size).cuda()
        results_mean_faulty = torch.zeros(batch_size).cuda()
        results_var_golden = torch.zeros(batch_size).cuda()
        results_var_faulty = torch.zeros(batch_size).cuda()
        results_covariance = torch.zeros(batch_size).cuda()
        results_structural_similarity_index = torch.zeros(batch_size).cuda()

        # Define size of grid/blocks
        threads_per_block = (1024, 1, 1)
        blocks_per_grid_size = (int(size/threads_per_block[0]) + 1, int(batch_size), 1)
        blocks_per_grid_batch = (int(batch_size/threads_per_block[0]) + 1, 1, 1)

        # Perform the sum, then divide to get the golden mean
        self._sum_kernel(
            golden_tensor,
            results_mean_golden,
            size,
            block=threads_per_block,
            grid=blocks_per_grid_size
        )
        self._divide_vector_scalar_kernel(
            results_mean_golden,
            size,
            batch_size,
            block=threads_per_block,
            grid=blocks_per_grid_batch
        )

        # Perform the sum, then divide to get the faulty mean
        self._sum_kernel(
            faulty_tensor,
            results_mean_faulty,
            size,
            block=threads_per_block,
            grid=blocks_per_grid_size
        )
        self._divide_vector_scalar_kernel(
            results_mean_faulty,
            size,
            batch_size,
            block=threads_per_block,
            grid=blocks_per_grid_batch
        )

        # Perform the variance, then compute last division to get the golden variance
        self._variance_kernel(
            golden_tensor,
            results_mean_golden,
            results_var_golden,
            size,
            block=threads_per_block,
            grid=blocks_per_grid_size
        )
        self._divide_vector_scalar_kernel(
            results_var_golden,
            size,
            batch_size,
            block=threads_per_block,
            grid=blocks_per_grid_batch
        )

        # Perform the variance, then compute last division to get the faulty variance
        self._variance_kernel(
            faulty_tensor,
            results_mean_faulty,
            results_var_faulty,
            size,
            block=threads_per_block,
            grid=blocks_per_grid_size
        )
        self._divide_vector_scalar_kernel(
            results_var_faulty,
            size,
            batch_size,
            block=threads_per_block,
            grid=blocks_per_grid_batch
        )


        # Compute the covariance, then last division to get the covariance
        self._covariance_kernel(
            golden_tensor,
            faulty_tensor,
            results_mean_golden,
            results_mean_faulty,
            results_covariance,
            size,
            block=threads_per_block,
            grid=blocks_per_grid_size
        )
        self._divide_vector_scalar_kernel(
            results_covariance,
            size,
            batch_size,
            block=threads_per_block,
            grid=blocks_per_grid_batch
        )

        # Compute sism
        self._structural_similarity_index_kernel(
            results_mean_golden,
            results_mean_faulty,
            results_var_golden,
            results_var_faulty,
            results_covariance,
            results_structural_similarity_index,
            np.float32(1e-8),
            np.float32(1e-9),
            batch_size,
            block=threads_per_block,
            grid=blocks_per_grid_batch
        )

        return results_structural_similarity_index