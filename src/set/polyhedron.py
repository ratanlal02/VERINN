"""
Author: Ratan Lal
Date : February 15, 2024
"""
from abc import ABC
from typing import List, Dict, Tuple

from gurobipy import Model, Var
from ppl import *
import numpy.typing as npt
from src.SET.box import Box
from src.SET.doulbeToIntegerExpr import DTIE
import numpy as np
import re
from src.SET.set import Set
import copy

from src.Types.sign import Sign


#class Polyhedron(Set, ABC):
class Polyhedron:
    """
    The set captures polyhedron
    """

    def __init__(self, objNNCPoly: NNC_Polyhedron):
        """
        Initialize an instance of Polyhedron
        :param objNNC_Poly: an instance of NNC_Polyhedron
        :type objNNC_Poly: NNC_Polyhedron
        """
        self.__objNNCPoly__ = objNNCPoly

    def getLowerBound(self) -> npt.ArrayLike:
        """
        Returns the lower bound of the set
        :return: (arrayLow -> npt.ArrayLike)
        lower bound of the set
        """
        intDim: int = self.getDimension()
        arrayLow: npt.ArrayLike = np.array([0.0 for i in range(intDim)], dtype='f')

        return arrayLow

    def getUpperBound(self) -> npt.ArrayLike:
        """
        Returns the lower bound of the set
        :return: (arrayHigh -> npt.ArrayLike)
        upper bound of the set
        """
        intDim: int = self.getDimension()
        arrayHigh: npt.ArrayLike = np.array([0.0 for i in range(intDim)], dtype='f')

        return arrayHigh

    def getDimension(self) -> int:
        """
        Returns the dimension of the set
        :return: (intDim -> int)
        dimension of the set
        """
        intDim: int = self.__objNNCPoly__.space_dimension()
        return intDim

    def getSameSignPartition(self) -> List['Set']:
        """
        Partition a set in such a way that for each sub set,
        values in each dimension will have the same sign
        :return: (listSets -> List[Set]) list of subsets
        """
        listSets: List['Set'] = []

        return listSets

    def getSign(self) -> List[Sign]:
        """
        Returns the list of Sign instances for all
        the dimensions of the set
        :return: (listSigns -> List[Sign])
        """
        intDim: int = self.getDimension()
        listSign: List[Sign] = []
        for i in range(intDim):
            listSign.append(Sign.POS)

        return listSign

    def toAbsolute(self) -> npt.ArrayLike:
        """
        Returns the absolute value of the set
        :return: (arrayPoint -> npt.ArrayLike)
        """
        intDim: int = self.getDimension()
        arrayAbsolute: npt.ArrayLike = np.array([0.0 for i in range(intDim)], dtype='f')

        return arrayAbsolute

    def getPoly(self) -> NNC_Polyhedron:
        """
        Returns the polyhedron of the set
        :return: (objNNCPoly -> objNNC_Polyhedron)
        """
        return self.__objNNCPoly__

    def intersect(self, objSet: Set) -> Set:
        """
        Compute an intersection with an other set
        :param objSet: an instance of Set
        :type objSet: Set
        :return: (intersectSet -> Set)
        """
        intDim: int = objSet.getDimension()
        objPolyhedron: Polyhedron = copy.deepcopy(self)
        if isinstance(objSet, Box):
            arrayLow: npt.ArrayLike = objSet.getLowerBound()
            arrayHigh: npt.ArrayLike = objSet.getUpperBound()
            for i in range(intDim):
                objPolyhedron.__objNNCPoly__.add_constraint(Variable(i) >= arrayLow[i])
                objPolyhedron.__objNNCPoly__.add_constraint(Variable(i) <= arrayHigh[i])
            return objPolyhedron
        elif isinstance(objSet, Polyhedron):
            # Create an instance of Universal NNC_Polyhedron with intDim dimension
            intersectPoly: NNC_Polyhedron = NNC_Polyhedron(intDim, 'universe')
            # Add constraints of the current polyhedron
            for c in self.__objNNCPoly__.constraints():
                intersectPoly.add_constraint(c)
            # Add constraints of the objPolyhedron
            for c in objSet.getPoly().constraints():
                intersectPoly.add_constraint(c)
            # Create an instance of a Poyhedron class for the intersection
            #intersectPoly.minimized_constraints()
            intersectPolyhedron: Set = Polyhedron(intersectPoly)
            return intersectPolyhedron
        else:
            return None

    def isEmpty(self) -> bool:
        """
        Check the emptiness of the set
        :return:
        """
        if self.getPoly().is_empty():
            return True
        else:
            return False

    def getModelAndDictVars(self) -> Tuple[Model, Dict[int, Dict[int, Var]]]:
        """
        Checks if the set is empty
        :return: (status -> bool)
        """
        objModel: Model = Model()
        dictVars: Dict[int, Dict[int, Var]] = {}

        return objModel, dictVars

    def getExpressions(self) -> List[str]:
        """
        Return list of expressions
        :return:(listExpr -> List[str])
        """
        listExpr: List[str] = []
        return listExpr
