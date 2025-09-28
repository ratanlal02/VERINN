"""
Author: Ratan Lal
Date : November 18, 2024
"""
from abc import ABC
from typing import Dict, List, Tuple
from gurobipy import Model, Var, GRB, quicksum
from ppl import NNC_Polyhedron

from src.SET.box import Box
from src.SET.set import Set
import numpy.typing as npt
import numpy as np

from src.SOLVER.gurobi import Gurobi
from src.SOLVER.solver import Solver
from src.Types.datatype import DataType
from src.Types.sign import Sign


class GRBSet(Set, ABC):
    def __init__(self, objModel: Model, dictVars: Dict[int, Dict[int, Var]]):
        """
        Initialize GRB set
        """
        self.__objModel__ = objModel
        self.__dictVars__ = dictVars

    def getExpressions(self) -> List[str]:
        """
        Return list of expressions
        :return:(listExpr -> List[str])
        """
        numVars: int = self.__objModel__.NumVars
        listVars: List[Var] = self.__objModel__.getVars()
        listExpr: List[str] = []
        for constr in self.__objModel__.getConstrs():
            listCoeff: List[float] = []
            constantTerm = 0.0
            for i in range(numVars):
                listCoeff.append(self.__objModel__.getCoeff(constr, listVars[i]))
            constantTerm = constr.getAttr("rhs")
            if constr.Sense == GRB.LESS_EQUAL:
                listExpr.append(str(quicksum(listVars[i] * listCoeff[i] for i in range(numVars) if listCoeff[i]!=0) <= constantTerm))
            elif constr.Sense == GRB.GREATER_EQUAL:
                listExpr.append(str(quicksum(listVars[i] * listCoeff[i] for i in range(numVars) if listCoeff[i]!=0) >= constantTerm))
            else:
                listExpr.append(str(quicksum(listVars[i] * listCoeff[i] for i in range(numVars) if listCoeff[i]!=0) == constantTerm))

        return listExpr

    def getLowerBound(self) -> npt.ArrayLike:
        """
        Returns the lower bound of the set
        :return: (arrayLow -> npt.ArrayLike)
        lower bound of the set
        """
        intDim: int = self.getDimension()
        arrayLow: npt.ArrayLike = [0.0 for i in range(intDim)]
        return arrayLow

    def getUpperBound(self) -> npt.ArrayLike:
        """
        Returns the lower bound of the set
        :return: (arrayHigh -> npt.ArrayLike)
        upper bound of the set
        """
        intDim: int = self.getDimension()
        arrayHigh: npt.ArrayLike = [0.0 for i in range(intDim)]
        return arrayHigh

    def getDimension(self) -> int:
        """
        Returns the dimension of the set
        :return: (intDim -> int)
        dimension of the set
        """
        key: int = [v for v in self.__dictVars__.keys()][0]
        return len(self.__dictVars__[key])

    def getSameSignPartition(self) -> List['Set']:
        """
        Partition a set in such a way that for each sub set,
        values in each dimension will have the same sign
        :return: (listSets -> List[Set]) list of subsets
        """
        intDim: int = self.getDimension()
        arrayLow: npt.ArrayLike = [DataType.RealType(0.0) for i in range(intDim)]
        arrayHigh: npt.ArrayLike = [DataType.RealType(0.0) for i in range(intDim)]
        listSets: List['Set'] = [Box(arrayLow, arrayHigh)]
        return listSets

    def getSign(self) -> List[Sign]:
        """
        Returns the list of Sign instances for all
        the dimensions of the set
        :return: (listSigns -> List[Sign])
        """
        intDim: int = self.getDimension()
        listSign: List[Sign] = [Sign.POS for i in range(intDim)]
        return listSign

    def toAbsolute(self) -> npt.ArrayLike:
        """
        Returns the absolute value of the set
        :return: (arrayPoint -> npt.ArrayLike)
        """
        intDim: int = self.getDimension()
        arrayPoint: npt.ArrayLike = [DataType.RealType(0.0) for i in range(intDim)]
        return arrayPoint

    def getPoly(self) -> NNC_Polyhedron:
        """
        Returns the polyhedron of the set
        :return: (objNNCPoly -> NNC_Polyhedron)
        """
        intDim: int = self.getDimension()
        objNNCPoly: NNC_Polyhedron = NNC_Polyhedron(intDim, 'universe')
        return objNNCPoly

    def intersect(self, objSet: 'Set') -> 'Set':
        """
        Compute intersection between two sets
        :param objSet: an instance of Set
        :type objSet: Set
        :return: (intesectSet -> Set)
        """
        intDim: int = self.getDimension()
        arrayLow: npt.ArrayLike = [DataType.RealType(0.0) for i in range(intDim)]
        arrayHigh: npt.ArrayLike = [DataType.RealType(0.0) for i in range(intDim)]
        objSet: Set = Box(arrayLow, arrayHigh)
        return objSet

    def isEmpty(self) -> bool:
        """
        Checks if the set is empty
        :return: (status -> bool)
        """
        objSolver: Solver = Gurobi(self.__objModel__, self.__dictVars__)
        return not (objSolver.satisfy())

    def getModelAndDictVars(self) -> Tuple[Model, Dict[int, Dict[int, Var]]]:
        """
        Checks if the set is empty
        :return: (status -> bool)
        """
        return self.__objModel__, self.__dictVars__

    def display(self) -> str:
        """
        Display all the constraints
        :return: None
        """
        strGRBSet: str = ""
        listVars: List[Var] = self.__objModel__.getVars()
        numVars: int = len(listVars)
        for constr in self.__objModel__.getConstrs():
            listCoeff: List[float] = []
            for i in range(numVars):
                listCoeff.append(self.__objModel__.getCoeff(constr, listVars[i]))
            constantTerm = constr.getAttr("rhs")
            isFirst: bool = True
            for i in range(numVars):
                if (listCoeff[i] != 0.0):
                    if isFirst:
                        strGRBSet += "          "+str(listVars[i].VarName)+"*"+str(listCoeff[i])
                        isFirst = False
                    else:
                        strGRBSet += " + " + str(listVars[i].VarName)+"*"+str(listCoeff[i])

            if constr.Sense == GRB.LESS_EQUAL:
                strGRBSet += " <= "+ str(constantTerm)
            elif constr.Sense == GRB.GREATER_EQUAL:
                strGRBSet += " >= "+ str(constantTerm)
            else:
                strGRBSet += " == "+ str(constantTerm)
            strGRBSet += "\n"

        return strGRBSet