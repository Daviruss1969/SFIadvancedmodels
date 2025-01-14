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
        from fm_analysis.cuda.minkowski_distance_wrapper import MinkowskiDistanceWrapper
        from fm_analysis.cuda.structural_similarity_index_wrapper import StructuralSimilarityIndexWrapper
        from fm_analysis.cuda.sparsity_ratio_wrapper import SparsityRatioWrapper
        from fm_analysis.cuda.activation_sensitivity_wrapper import ActivationSensitivityWrapper
        from fm_analysis.cuda.entropy_wrapper import EntropyWrapper
        from fm_analysis.cuda.activation_range_wrapper import ActivationRangeWrapper

        if self._metric == MetricEnum.MEAN_ABSOLUTE_ERROR:
            return MeanAbsoluteErrorWrapper(mode)
        
        if self._metric == MetricEnum.COSINE_SIMILARITY:
            return CosineSimilarityWrapper(mode)
        
        if self._metric == MetricEnum.MINKOWSKI_DISTANCE:
            return MinkowskiDistanceWrapper(mode)
        
        if self._metric == MetricEnum.STRUCTURAL_SIMILARITY_INDEX:
            return StructuralSimilarityIndexWrapper(mode)
        
        if self._metric == MetricEnum.SPARSITY_RATIO:
            return SparsityRatioWrapper(mode)

        if self._metric == MetricEnum.ACTIVATION_SENSITIVITY:
            return ActivationSensitivityWrapper(mode)
        
        if self._metric == MetricEnum.ENTROPY:
            return EntropyWrapper(mode)
        
        if self._metric == MetricEnum.ACTIVATION_RANGE:
            return ActivationRangeWrapper(mode)

        raise Exception("specified metric not handled")
    
    def new_batch(self):
        self._wrapper.new_batch()