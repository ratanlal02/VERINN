"""
Author: Ratan Lal
Date : November 18, 2024
"""
from abc import ABCMeta, abstractmethod
from typing import Dict, List

from src.set.set import Set


class Technique(metaclass=ABCMeta):
    """
    Technique abstract base class for reachability as well as
    for safety
    """

    @abstractmethod
    def reachSet(self) -> List[Set]:
        """
        Return reach set as a list of Set instance
        :return: (objSet -> List[Set])
        Reach set
        """
        pass

    @abstractmethod
    def checkSatisfy(self, objTargetSet: Set) -> bool:
        """
        Return safety of GNN instance against unsafe set
        :param objTargetSet: a Set instance
        :type objTargetSet: Set
        :return: (status -> bool)
        True if no valuation of GNN satisfies unsafe set
        otherwise false
        """
        pass

    @abstractmethod
    def checkSafety(self) -> bool:
        """
        Return safety status of a neural network
        :return: (status -> bool)
        True if safe, False otherwise
        """
        pass


