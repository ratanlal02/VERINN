"""
Author: Ratan Lal
Date : November 18, 2024
"""
from abc import ABC
from typing import List, Tuple, Dict
import numpy as np
import numpy.typing as npt
from gurobipy import Model, Var

from src.set.set import Set
from src.types.datatype import DataType
from src.types.sign import Sign
import itertools
from src.utilities.log import Log


class Box(Set, ABC):
    """
    The class Box captures multidimensional rectangular set
    """

    def __init__(self, arrayLow: npt.ArrayLike, arrayHigh: npt.ArrayLike):
        """
        Initialize an instance of the class Box
        :param arrayLow: one dimensional array for the lower end point
        :type arrayLow: npt.ArrayLike
        :param arrayHigh: one dimensional array for the upper end point
        :type arrayHigh: npt.ArrayLike
        """
        self.arrayLow: npt.ArrayLike = arrayLow
        self.arrayHigh: npt.ArrayLike = arrayHigh

    def getLowerBound(self) -> npt.ArrayLike:
        """
        Returns the lower end point of the Box instance
        :return: (arrayLow -> npt.ArrayLike)
        One dimensional array for the lower end point fo the box
        """
        return self.arrayLow

    def getUpperBound(self) -> npt.ArrayLike:
        """
        Returns the upper end point of the Box instance
        :return: (arrayHigh -> npt.ArrayLike)
        One dimensional array for the upper end point fo the box
        """
        return self.arrayHigh

    def getDimension(self) -> int:
        """
        Returns the dimension of the Box instance
        :return: (intDim -> int)
        Dimension of the Box instance
        """
        # Compute dimension of the Box instance
        intDim: int = len(self.arrayLow)
        return intDim

    def getSameSignPartition(self) -> List['Set']:
        """
        Partition a set in such a way that for each sub set,
        values in each dimension will have the same sign
        :return: (listSets -> List[Set]) list of subsets
        """
        listSign: List[Sign] = self.getSign()
        listOfListofArrays: List[List[npt.ArrayLike]] = []
        for i in range(len(listSign)):
            tempListOfArrays: List[npt.ArrayLike] = []
            if listSign[i] == Sign.POS or listSign[i] == Sign.NEG:
                tempListOfArrays.append(np.array([self.arrayLow[i], self.arrayHigh[i]]))
            else:
                tempListOfArrays.append(np.array([self.arrayLow[i], DataType.RealType(0.0)]))
                tempListOfArrays.append(np.array([DataType.RealType(0.0), self.arrayHigh[i]]))
            listOfListofArrays.append(tempListOfArrays)

        cartesianProduct = itertools.product(*listOfListofArrays)
        listSets: List[Set] = []
        for idx, item in enumerate(cartesianProduct):
            arrayLow: npt.ArrayLike = np.array([arr[0] for arr in item], dtype=object)
            arrayHigh: npt.ArrayLike = np.array([arr[1] for arr in item], dtype=object)
            listSets.append(Box(arrayLow, arrayHigh))

        return listSets

    def getSign(self) -> List[Sign]:
        """
        extract sign of each dimension
        :return: (listSign -> List[Sign])
        """
        intDim: int = self.getDimension()
        listSign: List[Sign] = []
        for i in range(intDim):
            # Find the sign of index i
            enumSign = self.__getSignByIndex__(i)
            listSign.append(enumSign)

        # Return sign for all the dimensions
        return listSign

    def toAbsolute(self) -> npt.ArrayLike:
        """
        Returns the absolute value of the set
        :return: (arrayPoint -> npt.ArrayLike)
        """
        arrayPoint: npt.ArrayLike = np.array([max(abs(self.arrayLow[i]), abs(self.arrayHigh[i]))
                                              for i in range(self.getDimension())], dtype=object)

        return arrayPoint

    def __getSignByIndex__(self, intIndex: int) -> Sign:
        """
        Find the type of values at intIndex that self has
        :param intIndex: index of the variable
        :type intIndex: int
        :return: (enumSign -> Sign) Sign is an enumeration class
        """
        sign: Sign = Sign.NONE
        if self.arrayLow[intIndex] >= 0.0:
            sign = Sign.POS
        elif self.arrayHigh[intIndex] <= 0.0:
            sign = Sign.NEG
        else:
            sign = Sign.BOTH

        return sign

    # def getPoly(self) -> NNC_Polyhedron:
    #     """
    #     Returns the polyhedron of the set
    #     :return: (objNNCPoly -> NNC_Polyhedron)
    #     """
    #     intDim: int = self.getDimension()
    #     objNNCPoly: NNC_Polyhedron = NNC_Polyhedron(intDim, 'universe')
    #
    #     return objNNCPoly

    def intersect(self, objSet: 'Set') -> 'Set':
        """
        Compute intersection between two sets
        :param objSet: an instance of Set
        :type objSet: Set
        :return: (intesectSet -> Set)
        """
        # It is not fully implemented
        return self

    def isEmpty(self) -> bool:
        """
        Checks if the set is empty
        :return: (status -> bool)
        """
        intDim: int = self.getDimension()
        for i in range(intDim):
            if self.arrayLow[i] > self.arrayHigh[i]:
                return True
        return False

    def getModelAndDictVars(self) -> Tuple[Model, Dict[int, Dict[int, Var]]]:
        """
        Checks if the set is empty
        :return: (status -> bool)
        """
        # This is not related to Box
        objModel: Model = Model()
        dictVars: Dict[int, Dict[int, Var]] = {}

        return objModel, dictVars

    def getExpressions(self) -> List[str]:
        """
        Return list of expressions
        :return:(listExpr -> List[str])
        """
        # This is not related to Box
        listExpr: List[str] = []
        return listExpr

    def display(self) -> str:
        """
        Display lower and upper bounds of the Box
        :return: None
        """
        strBox: str = "         Lower Bound\n"
        strBox += "         "+str(self.getLowerBound()) + "\n"
        strBox += "         Upper Bound\n"
        strBox += "         "+str(self.getUpperBound()) +"\n"

        return strBox
