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
        self._square_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/square.{self._mode}").get_function("square")
        self._multiply_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/multiply.{self._mode}").get_function("multiply")
        self._divide_kernel = cuda.module_from_file(f"fm_analysis/cuda/{self._mode}/divide.{self._mode}").get_function("divide")


    def __call__(self,
                golden_tensor: torch.Tensor,
                faulty_tensor: torch.Tensor,
                batch_size: int,
                size: int) -> float:
        # Define results in GPU memory
        dot_products = torch.zeros(batch_size).cuda()
        golden_norms = torch.zeros(batch_size).cuda()
        faulty_norms = torch.zeros(batch_size).cuda()
        multiply_results = torch.zeros(batch_size).cuda()
        cosine_similarities = torch.zeros(batch_size).cuda()

        # Define size of grid/blocks
        threads_per_block = (1024, 1, 1)
        blocks_per_grid_size = (int(size/threads_per_block[0]) + 1, int(batch_size), 1)
        blocks_per_grid_batch = (int(batch_size/threads_per_block[0]) + 1, 1, 1)

        # Call the dot product kernel
        self._dot_product_kernel(
            golden_tensor,
            faulty_tensor,
            dot_products,
            size,
            block=threads_per_block,
            grid=blocks_per_grid_size
        )

        # Call the euclidian norm kernel, perform the square root in another kernel
        self._euclidian_norm_kernel(
            golden_tensor,
            golden_norms,
            size,
            block=threads_per_block,
            grid=blocks_per_grid_size
        )
        self._square_kernel(
            golden_norms,
            batch_size,
            block=threads_per_block,
            grid=blocks_per_grid_batch
        )

        # Call the euclidian norm kernel, perform the square root in another kernel
        self._euclidian_norm_kernel(
            faulty_tensor,
            faulty_norms,
            size,
            block=threads_per_block,
            grid=blocks_per_grid_size
        )
        self._square_kernel(
            faulty_norms,
            batch_size,
            block=threads_per_block,
            grid=blocks_per_grid_batch
        )

        # Perform the multiplication on gpu
        self._multiply_kernel(
            golden_norms,
            faulty_norms,
            multiply_results,
            batch_size,
            block=threads_per_block,
            grid=blocks_per_grid_batch
        )
        
        # Perform the division on gpu
        self._divide_kernel(
            dot_products,
            multiply_results,
            cosine_similarities,
            batch_size,
            block=threads_per_block,
            grid=blocks_per_grid_batch
        )

        # When we have an overflow, since the cosine similarity is bewteen -1 and 1, just return 0 : no corelations between vectors
        cosine_similarities = torch.nan_to_num(cosine_similarities, nan=0.0)

        # Return cosine similarities
        return cosine_similarities