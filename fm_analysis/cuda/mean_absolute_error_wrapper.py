from typing import Callable, Literal

import pycuda.driver as cuda
import torch

from fm_analysis.cuda.cuda_wrapper import CudaWrapper

class MeanAbsoluteErrorWrapper(CudaWrapper):
    _sum_absolute_error_kernel: Callable
    def __init__(self,
                 mode: Literal["ptx", "cubin"]):
        super().__init__(mode)
        self._sum_absolute_error_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/sum_absolute_error.{self._mode}").get_function("sum_absolute_error")

    def __call__(self,
                golden_tensor: torch.Tensor,
                faulty_tensor: torch.Tensor,
                size: int) -> float:
        # Define result in GPU memory
        result = torch.zeros(1).cuda()

        # Define size of grid/blocks
        threads_per_block = (1024, 1, 1)
        blocks_per_grid = (int(size/threads_per_block[0]) + 1, 1, 1)

        # Call the kernel and get the sum of absolute errors
        self._sum_absolute_error_kernel(
            golden_tensor,
            faulty_tensor,
            result,
            size,
            block=threads_per_block,
            grid=blocks_per_grid
        )
        
        # Return the mean in the cpu memory
        return result.item()/size
    