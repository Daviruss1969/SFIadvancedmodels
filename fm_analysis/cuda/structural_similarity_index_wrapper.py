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
        self._variance_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/variance.{self._mode}").get_function("variance")
        self._covariance_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/covariance.{self._mode}").get_function("covariance")

    def __call__(self,
                golden_tensor: torch.Tensor,
                faulty_tensor: torch.Tensor,
                size: int) -> float:
        # Define results in GPU memory
        result_sum_golden = torch.zeros(1).cuda()
        result_sum_faulty = torch.zeros(1).cuda()
        result_var_golden = torch.zeros(1).cuda()
        result_var_faulty = torch.zeros(1).cuda()
        result_covariance = torch.zeros(1).cuda()

        # Define size of grid/blocks
        threads_per_block = (1024, 1, 1)
        blocks_per_grid = (int(size/threads_per_block[0]) + 1, 1, 1)

        # Perform the mean on the cpu with the sum kernel
        self._sum_kernel(
            golden_tensor,
            result_sum_golden,
            size,
            block=threads_per_block,
            grid=blocks_per_grid
        )
        golden_mean = result_sum_golden.item()/size

        # Perform the mean on the cpu with the sum kernel
        self._sum_kernel(
            faulty_tensor,
            result_sum_faulty,
            size,
            block=threads_per_block,
            grid=blocks_per_grid
        )
        faulty_mean = result_sum_faulty.item()/size

        # Perform the variance (compute last division on cpu)
        self._variance_kernel(
            golden_tensor,
            result_var_golden,
            np.float32(golden_mean),
            size,
            block=threads_per_block,
            grid=blocks_per_grid
        )
        golden_var = result_var_golden.item()/size

        # Perform the variance (compute last division on cpu)
        self._variance_kernel(
            faulty_tensor,
            result_var_faulty,
            np.float32(faulty_mean),
            size,
            block=threads_per_block,
            grid=blocks_per_grid
        )
        faulty_var = result_var_faulty.item()/size

        # Compute the covariance (last division on cpu)
        self._covariance_kernel(
            golden_tensor,
            faulty_tensor,
            result_covariance,
            np.float32(golden_mean),
            np.float32(faulty_mean),
            size,
            block=threads_per_block,
            grid=blocks_per_grid
        )
        covariance = result_covariance.item()/size

        # Compute sism
        C1 = 1e-8
        C2 = 1e-9
        sism_dividend = (2*golden_mean*faulty_mean + C1)*(2*covariance + C2)
        sism_divisor = (math.pow(golden_mean, 2)*math.pow(faulty_mean, 2 ) + C1)*(golden_var*faulty_var + C2)

        return sism_dividend/sism_divisor