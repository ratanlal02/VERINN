"""
Author: Ratan Lal
Date : January 28, 2025
"""
from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Dict

import numpy.typing as npt
from gurobipy import Model, Var
#from ppl import NNC_Polyhedron

from src.types.sign import Sign


class Set(metaclass=ABCMeta):
    """
    Abstract class for capturing different classes of sets
    """

    @abstractmethod
    def getLowerBound(self) -> npt.ArrayLike:
        """
        Returns the lower bound of the set
        :return: (arrayLow -> npt.ArrayLike)
        lower bound of the set
        """
        pass

    @abstractmethod
    def getUpperBound(self) -> npt.ArrayLike:
        """
        Returns the lower bound of the set
        :return: (arrayHigh -> npt.ArrayLike)
        upper bound of the set
        """
        pass

    @abstractmethod
    def getDimension(self) -> int:
        """
        Returns the dimension of the set
        :return: (intDim -> int)
        dimension of the set
        """
        pass

    @abstractmethod
    def getSameSignPartition(self) -> List['Set']:
        """
        Partition a set in such a way that for each sub set,
        values in each dimension will have the same sign
        :return: (listSets -> List[Set]) list of subsets
        """
        pass

    @abstractmethod
    def getSign(self) -> List[Sign]:
        """
        Returns the list of Sign instances for all
        the dimensions of the set
        :return: (listSigns -> List[Sign])
        """
        pass

    @abstractmethod
    def toAbsolute(self) -> npt.ArrayLike:
        """
        Returns the absolute value of the set
        :return: (arrayPoint -> npt.ArrayLike)
        """
        pass

    # @abstractmethod
    # def getPoly(self) -> NNC_Polyhedron:
    #     """
    #     Returns the polyhedron of the set
    #     :return: (objNNCPoly -> NNC_Polyhedron)
    #     """
    #     pass

    @abstractmethod
    def intersect(self, objSet: 'Set') -> 'Set':
        """
        Compute intersection between two sets
        :param objSet: an instance of Set
        :type objSet: Set
        :return: (intesectSet -> Set)
        """
        pass

    @abstractmethod
    def isEmpty(self) -> bool:
        """
        Checks if the set is empty
        :return: (status -> bool)
        """
        pass

    @abstractmethod
    def getModelAndDictVars(self) -> Tuple[Model, Dict[int, Dict[int, Var]]]:
        """
        Checks if the set is empty
        :return: (status -> bool)
        """
        pass

    @abstractmethod
    def getExpressions(self) -> List[str]:
        """
        Return list of expressions
        :return:(listExpr -> List[str])
        """
        pass

    @abstractmethod
    def display(self) -> str:
        """
        Display the set
        :return: None
        """
        pass