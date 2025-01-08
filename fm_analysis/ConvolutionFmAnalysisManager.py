from typing import Literal

import torch
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

from fm_analysis.FmAnalysisManager import FmAnalysisManager
from fm_analysis.MetricEnum import MetricEnum

class ConvolutionFmAnalysisManager(FmAnalysisManager):
    def __init__(self,
                 mode: Literal["ptx", "cubin"],
                 metric: MetricEnum):
        super().__init__(mode, metric)

    def __call__(self, golden_tensor: torch.Tensor, faulty_tensor: torch.Tensor) -> float:
        # Unroll the convolution tensor by its C,W,H keep the batchs
        batchs, channels, width, height = golden_tensor.shape
        size = np.int32(channels*width*height)
        batch_size = np.int32(batchs)

        # Call the wrapper of the kernel
        return self._wrapper(golden_tensor,
                             faulty_tensor,
                             batch_size,
                             size)