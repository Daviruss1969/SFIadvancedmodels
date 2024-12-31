from typing import Literal, Type, Callable
from abc import abstractmethod

import torch
import pycuda.autoinit # To init the pycuda module

from fm_analysis.MetricEnum import MetricEnum

class FmAnalysisManager:

    _mode: Literal["ptx", "cubin"]
    _metric: MetricEnum
    _cuda_function: Callable

    def __init__(self,
                 mode: Literal["ptx", "cubin"],
                 metric: MetricEnum):
        self._mode = mode
        self._metric = metric

    @abstractmethod
    def __call__(self, golden_tensor: torch.Tensor, faulty_tensor: torch.Tensor) -> torch.Tensor:
        pass
        
    @staticmethod
    def torch_module_to_analysis_manager(module: torch.nn.modules.Module) -> Type['FmAnalysisManager']:
        from fm_analysis.ConvolutionFmAnalysisManager import ConvolutionFmAnalysisManager

        if module == torch.nn.Conv2d:
            return ConvolutionFmAnalysisManager
        
        raise Exception("specified module not handled")