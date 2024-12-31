from typing import Literal

import torch
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from fm_analysis.FmAnalysisManager import FmAnalysisManager
from fm_analysis.MetricEnum import MetricEnum

class ConvolutionFmAnalysisManager(FmAnalysisManager):
    def __init__(self,
                 mode: Literal["ptx", "cubin"],
                 metric: MetricEnum):
        super().__init__(mode, metric)
        self._cuda_function = cuda.module_from_file(f"fm_analysis/cuda/{mode}/convolution.{mode}").get_function(str(metric))

    def __call__(self, golden_tensor: torch.Tensor, faulty_tensor: torch.Tensor) -> torch.Tensor:
        # Unroll shape
        batchs, channels, width, height = golden_tensor.shape
        print(golden_tensor.shape)
        print(faulty_tensor.shape)

        threads_per_block = (1, 1, 1)
        blocks_per_grid = (1, 1, 1)
        self._cuda_function(
            faulty_tensor,
            block=threads_per_block,
            grid=blocks_per_grid
        )

        print(faulty_tensor[0][0][0][0])
