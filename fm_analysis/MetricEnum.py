from enum import Enum

class MetricEnum(Enum):
    MEAN_ABSOLUTE_ERROR = 1
    COSINE_SIMILARITY = 2
    STRUCTURAL_SIMILARITY_INDEX = 3
    PEAK_SIGNAL_TO_NOISE_RATIO = 4
    ACTIVATION_SENSITIVITY = 5
    SPARSITY_RATIO = 6
    ACTIVATION_RANGE = 7
    ENTROPY = 8
    FEATURE_MAP_CONSISTENCY = 9
    MINKOWSKI_DISTANCE = 10

    def __str__(self) -> str:
        return self.name.lower()
