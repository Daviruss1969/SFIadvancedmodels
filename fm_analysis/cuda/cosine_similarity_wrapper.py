from typing import Callable, Literal

import pycuda.driver as cuda
import torch
import math

from fm_analysis.cuda.cuda_wrapper import CudaWrapper

class CosineSimilarityWrapper(CudaWrapper):
    def __init__(self,
                 mode: Literal["ptx", "cubin"]):
        super().__init__(mode)
        self._dot_product_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/dot_product.{self._mode}").get_function("dot_product")
        self._euclidian_norm_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/euclidian_norm.{self._mode}").get_function("euclidian_norm")

    def __call__(self,
                golden_tensor: torch.Tensor,
                faulty_tensor: torch.Tensor,
                size: int) -> float:
        # Define results in GPU memory
        dot_product = torch.zeros(1).cuda()
        golden_norm = torch.zeros(1).cuda()
        faulty_norm = torch.zeros(1).cuda()

        # Define size of grid/blocks
        threads_per_block = (1024, 1, 1)
        blocks_per_grid = (int(size/threads_per_block[0]) + 1, 1, 1)

        # Call the dot product kernel
        self._dot_product_kernel(
            golden_tensor,
            faulty_tensor,
            dot_product,
            size,
            block=threads_per_block,
            grid=blocks_per_grid
        )

        # Call the euclidian norm kernel and perform the square root on cpu
        self._euclidian_norm_kernel(
            golden_tensor,
            golden_norm,
            size,
            block=threads_per_block,
            grid=blocks_per_grid
        )
        golden_norm = golden_norm.sqrt().item()

        # Call the euclidian norm kernel and perform the square root on cpu
        self._euclidian_norm_kernel(
            golden_tensor,
            faulty_norm,
            size,
            block=threads_per_block,
            grid=blocks_per_grid
        )
        faulty_norm = faulty_norm.sqrt().item()
        
        # Compute and return the cosine similarity in the cpu memory
        norm_product = golden_norm * faulty_norm
        result = dot_product.item() / max(norm_product, 1e-6) # Avoid division by zero

        if math.isinf(result): # If norm is inifinite (overflow) return 0 : no correlation between vectors
            result = 0.0

        return round(result, 3)