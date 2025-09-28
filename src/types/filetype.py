"""
Author: Ratan Lal
Date : November 4, 2023
"""

import enum


class FileType(enum.Enum):
    """
    It captures different neural network file types
    """
    # for sherlock
    SHERLOCK = 1

    # for neural network (nnet)
    NEURALNET = 2

    # for open neural network
    ONNX = 3

    # for not defined file format
    UNKNOWN = -1
