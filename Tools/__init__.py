"""
A package of useful tools for photogrammetry processing.
"""
from strenum import StrEnum


class ModelExecutionEngines(StrEnum):
    """
    An enum of common neural network execution engines.
    """

    CUDA = "cuda"
    CPU = "cpu"
    TENSORRT = "tensorRT"
