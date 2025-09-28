"""
Author: Ratan Lal
Date : November 4, 2023
"""
from typing import List, Tuple, Dict
import numpy.typing as npt

from src.gnn.gnn import GNN
from src.types.networktype import NetworkType
from abc import ABCMeta, abstractmethod


class Parser(metaclass=ABCMeta):
    """
    Abstract class for parsing different input formats of nn
    """

    @abstractmethod
    def getNetwork(self) -> GNN:
        """
        Get a neural network of any type
        :return: (objGNN: GNN)
        An instance of GNN class
        """
        pass

