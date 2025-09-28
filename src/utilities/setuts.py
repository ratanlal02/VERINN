"""
Author: Ratan Lal
Date : January 28, 2025
"""
from typing import List, Dict

from gurobipy import Model, GRB, quicksum, Var

from src.SET.box import Box
from src.SET.grbset import GRBSet
from src.SET.polyhedron import Polyhedron
from src.SET.set import Set
import numpy.typing as npt
import numpy as np

from src.SOLVER.gurobi import Gurobi
from src.SOLVER.solver import Solver
from src.Types.datatype import DataType
from src.Types.solvertype import SolverType
from src.UTILITIES.log import Log


class SetUTS:
    """
    Common functionality related to Set
    """

    @staticmethod
    def encodeSet(grbModel: Model, objSet: Set, stateVars) -> Model:
        """
        Encode a set into a gurobi model
        :param grbModel: an instance of a gurobi model
        :type grbModel: Model
        :param objSet: an instance of a Set
        :type objSet: Set
        :param stateVars: list of gurobi variables
        :type stateVars: List[Var]
        :return: (grbModel -> Model)
        """
        intDim: int = len(stateVars)
        if isinstance(objSet, Box):
            arrayLow: npt.ArrayLike = objSet.getLowerBound()
            arrayHigh: npt.ArrayLike = objSet.getUpperBound()
            for j in range(intDim):
                grbModel.addConstr(stateVars[j] >= np.float64(arrayLow[j]))
                grbModel.addConstr(stateVars[j] <= np.float64(arrayHigh[j]))
        elif isinstance(objSet, Polyhedron):
            for constraint in objSet.getPoly().constraints():
                type = constraint.type()
                coefficient = constraint.coefficients()
                constant = constraint.inhomogeneous_term()
                if type == 'equality':
                    grbModel.addConstr(quicksum(coefficient[k] * stateVars[k] for k in range(intDim)) + constant == 0)
                elif type == 'nonstrict_inequality':
                    grbModel.addConstr(quicksum(coefficient[k] * stateVars[k] for k in range(intDim)) + constant >= 0)
                elif type == 'strict_inequality':
                    grbModel.addConstr(quicksum(coefficient[k] * stateVars[k] for k in range(intDim)) + constant > 0)
        else:
            objModel, dictVars = objSet.getModelAndDictVars()
            numVars: int = objModel.NumVars
            listVars: List[Var] = objModel.getVars()
            for constr in objModel.getConstrs():
                listCoeff: List[float] = []
                constantTerm = 0.0
                for i in range(numVars):
                    listCoeff.append(objModel.getCoeff(constr, listVars[i]))
                constantTerm = constr.getAttr("rhs")
                if constr.Sense == GRB.LESS_EQUAL:
                    grbModel.addConstr(
                        quicksum(stateVars[i] * listCoeff[i] for i in range(numVars)) <= constantTerm)
                elif constr.Sense == GRB.GREATER_EQUAL:
                    grbModel.addConstr(
                        quicksum(stateVars[i] * listCoeff[i] for i in range(numVars)) >= constantTerm)
                else:
                    grbModel.addConstr(
                        quicksum(stateVars[i] * listCoeff[i] for i in range(numVars)) == constantTerm)
        # Update gurobi model
        grbModel.update()

        # Return the updated gurobi model
        return grbModel

    @staticmethod
    def getNonIntersectByIndices(objSetOne: Set, objSetTwo: Set, listIndices: List[int], solverType: SolverType) -> \
    Dict[int, bool]:
        """
        Return dictionary between indices and status that checks whether a point is common in both sets
        with respect to indices
        :param objSetOne: an instance of Set
        :type objSetOne: Set
        :param objSetTwo: an instance of Set
        :type objSetTwo: Set
        :param listIndices: list of indices
        :type listIndices: List[int]
        :param solverType: solver type
        :type solverType: SolverType
        :return: (dictIndices -> Dict[int, bool])
        """
        dictIndices: Dict[int, bool] = dict()
        intDim: int = objSetOne.getDimension()
        numVars: int = 2 * intDim

        grbModel: Model = Model()
        stateVars: List[Var] = []
        for i in range(numVars):
            stateVars.append(grbModel.addVar(lb=-np.float64('inf'), vtype=GRB.CONTINUOUS, name='x' + str(i)))

        # Encoding first set
        stateVarsForSetOne: List[Var] = stateVars[0: intDim]
        grbModel = SetUTS.encodeSet(grbModel, objSetOne, stateVarsForSetOne)
        # Encode second set
        stateVarsForSetTwo: List[Var] = stateVars[intDim: numVars]
        grbModel = SetUTS.encodeSet(grbModel, objSetTwo, stateVarsForSetTwo)
        Log.message("           Constraints for post set followed by Concrete set\n")
        for intIndex in listIndices:
            grbModelCopy: Model = grbModel.copy()
            varMap = {v: grbModelCopy.getVarByName(v.varName) for v in grbModel.getVars()}
            grbModelCopy.addConstr((varMap[stateVars[intIndex - 1]] == varMap[stateVars[intIndex - 1 + intDim]]))
            grbModelCopy.update()
            objSet: Set = GRBSet(grbModelCopy, None)
            Log.message(objSet.display() + "\n")
            if solverType == SolverType.Gurobi:
                objSolver: Solver = Gurobi(grbModelCopy, None)
                status = objSolver.satisfy()
                if not (status):
                    dictIndices[intIndex] = True
                else:
                    dictIndices[intIndex] = False

        return dictIndices

    @staticmethod
    def getNonIntersectBySetIndices(objSetOne: Set, objSetTwo: Set, listIndices: List[int], solverType: SolverType) \
            -> bool:
        """
        Return dictionary between indices and status that checks whether a point is common in both sets
        with respect to indices
        :param objSetOne: an instance of Set
        :type objSetOne: Set
        :param objSetTwo: an instance of Set
        :type objSetTwo: Set
        :param listIndices: list of indices
        :type listIndices: List[int]
        :param solverType: solver type
        :type solverType: SolverType
        :return: (status -> bool)
        True means that there is no intersection sets for the set of indices
        """
        NonIntersectStatus: bool = False
        intDim: int = objSetOne.getDimension()
        numVars: int = 2 * intDim

        grbModel: Model = Model()
        stateVars: List[Var] = []
        for i in range(numVars):
            stateVars.append(grbModel.addVar(lb=-np.float64('inf'), vtype=GRB.CONTINUOUS, name='x' + str(i)))

        # Encoding first set
        stateVarsForSetOne: List[Var] = stateVars[0: intDim]
        grbModel = SetUTS.encodeSet(grbModel, objSetOne, stateVarsForSetOne)

        # Encode second set
        stateVarsForSetTwo: List[Var] = stateVars[intDim: numVars]
        grbModel = SetUTS.encodeSet(grbModel, objSetTwo, stateVarsForSetTwo)

        for intIndex in listIndices:
            grbModel.addConstr((stateVars[intIndex - 1] == stateVars[intIndex - 1 + intDim]))
        if solverType == SolverType.Gurobi:
            objSolver: Solver = Gurobi(grbModel, None)
            status = objSolver.satisfy()
            if not (status):
                NonIntersectStatus = True
            else:
                NonIntersectStatus = False

        return NonIntersectStatus

    @staticmethod
    def displayLinConstr(A: npt.ArrayLike, b: npt.ArrayLike):
        """
        Print Ax <= b in the form of A b
        :param A: Two dimensional array
        :type A: npt.ArrayLike
        :param b: One dimensional array
        :type b: npt.ArrayLike
        :return: None
        """
        for row in range(len(A)):
            paddedRows = []
            paddedRows.append("         " + str(A[row]))
            paddedRows.append(str(b[row]))
            Log.message(str('     '.join(paddedRows)) + "\n")

    @staticmethod
    def displayDictOfDictOfSets(dictPartition: Dict[int, Dict[int, set]]):
        """
        Print dictionary of dictionaries of sets
        :param dictPartition: dictionary of dictionaries of sets
        :type dictPartition: Dict[int, Dict[int, Set]]
        :return: None
        """
        for intLayer in dictPartition.keys():
            paddedRows = []
            paddedRows.append("       " + str(intLayer) + " : " + str(dictPartition[intLayer]))
            Log.message(str('     '.join(paddedRows)) + "\n")

    @staticmethod
    def displayDictOfDictOfValues(dictCE: Dict[int, Dict[int, DataType.RealType]]):
        """
        Print dictionary of dictionaries of floats
        :param dictPartition: dictionary of dictionaries of floats
        :type dictPartition: Dict[int, Dict[int, float]]
        :return: None
        """
        for intLayer in dictCE.keys():
            paddedRows = []
            paddedRows.append("         " + str(intLayer) + " : " + str(dictCE[intLayer]))
            Log.message(str('     '.join(paddedRows)) + "\n")

    @staticmethod
    def displayListOfSets(listOfSets: List[Set]):
        """
        Print list of Sets
        :param listOfSets: list of Sets
        :type listOfSets: List[Set]
        :return: None
        """
        i: int = 0
        for objSet in listOfSets:
            Log.message("           Set " + str(i+1) + "\n")
            i += 1
            Log.message(objSet.display())
