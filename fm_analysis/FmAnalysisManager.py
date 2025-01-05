from typing import Literal, Type, Callable
from abc import abstractmethod

import torch
import pycuda.autoinit # To init the pycuda module

from fm_analysis.MetricEnum import MetricEnum
from fm_analysis.cuda.cuda_wrapper import CudaWrapper

class FmAnalysisManager:

    _metric: MetricEnum
    _wrapper: CudaWrapper

    def __init__(self,
                 mode: Literal["ptx", "cubin"],
                 metric: MetricEnum):
        self._metric = metric
        self._wrapper = self._get_cuda_wrapper(mode)

    @abstractmethod
    def __call__(self, golden_tensor: torch.Tensor, faulty_tensor: torch.Tensor) -> float:
        pass
        
    @staticmethod
    def torch_module_to_analysis_manager(module: torch.nn.modules.Module) -> Type['FmAnalysisManager']:
        from fm_analysis.ConvolutionFmAnalysisManager import ConvolutionFmAnalysisManager

        if module == torch.nn.Conv2d:
            return ConvolutionFmAnalysisManager
        
        raise Exception("specified module not handled")
    
    def _get_cuda_wrapper(self,
                          mode: Literal["ptx", "cubin"]) -> CudaWrapper:
        from fm_analysis.cuda.mean_absolute_error_wrapper import MeanAbsoluteErrorWrapper
        from fm_analysis.cuda.cosine_similarity_wrapper import CosineSimilarityWrapper

        if self._metric == MetricEnum.MEAN_ABSOLUTE_ERROR:
            return MeanAbsoluteErrorWrapper(mode)
        
        if self._metric == MetricEnum.COSINE_SIMILARITY:
            return CosineSimilarityWrapper(mode)
        
        raise Exception("specified metric not handled")