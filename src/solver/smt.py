"""
Author: Ratan Lal
Date : January 30, 2025
"""
import os
from abc import ABC
from typing import Dict, List
from gurobipy import Model, Var, GRB
from src.set.set import Set
from src.solver.solver import Solver
from fractions import Fraction
from decimal import Decimal
from src.types.datatype import DataType


class SMT(Solver, ABC):
    """
    Implement all the functions using SMT
    """

    def __init__(self, grbModel: Model, dictVarsX: Dict[int, Dict[int, Var]]):
        """
        Perform operation related to grbModel through SMT solver
        :param grbModel: an instance of a gurobi model
        :type grbModel: Model
        :param dictVarsX: dictionary of gurobi variables
        :type dictVarsX: Dict[int, Dict[int, Var]]
        """
        self.__model__ = grbModel
        self.__dictVars__ = dictVarsX

    def satisfy(self) -> bool:
        """
        Check the satisfiability of a set of constraints
        :return: (status -> bool)
        True if satisfies all constraints, False otherwise
        """
        self.__toSMT__()
        # store the result in a temporary file
        f = open("temp.txt", 'w')
        f.close()
        os.system('z3 ce.smt2 > temp.txt')
        f = open("temp.txt")
        lines = f.readlines()
        f.close()
        status = lines[0].strip("\n")
        print(status)
        if status == "sat":
            return True
        else:
            return False

    def getInstance(self, varMap: Dict[Var, Var]) -> Dict[int, Dict[int, DataType.RealType]]:
        """
        Extract a satisfiable instance of the model
        :return: (dictOutput -> Dict[int, Dict[int, float]])
        the satisfiable instance of the model
        """
        self.__toSMT__()
        # store the result in a temporary file
        f = open("temp.txt", 'w')
        f.close()
        os.system('z3 ce.smt2 > temp.txt')
        f = open("temp.txt")
        dictInstance: Dict[int, Dict[int, DataType.RealType]] = {}
        lines = f.readlines()
        f.close()
        status = lines[0].strip("\n")
        if status != "sat":
            return dictInstance
        for inLayer in self.__dictVars__.keys():
            dictTemp: Dict[int, DataType.RealType] = dict()
            for id in self.__dictVars__[inLayer].keys():
                dictTemp[id] = 0.0
            dictInstance[inLayer] = dictTemp
        i: int = 2
        while i < len(lines) - 1:
            index = lines[i].find('x')
            if index != -1:
                strLayer = ''
                while lines[i][index + 2] != '_':
                    strLayer += lines[i][index + 2]
                    index += 1
                strId = ''
                while lines[i][index + 3] != ' ':
                    strId += lines[i][index + 3]
                    index += 1

                intLayer = int(strLayer)
                intId = int(strId)

                floatVal = 0.0
                i += 1
                if lines[i].find('(') != -1 and lines[i].find(')') != -1:
                    line = lines[i].strip(' ').strip('(').strip('/').strip(' ')
                    sign = 0
                    if line[0] == '-':
                        line = line.strip('-').strip(' ').strip('(').strip('/').strip(' ')
                        sign = 1
                    firstSpaceIndex = line.find(' ')
                    firstCloseBracket = line.find(')')
                    floatOpernd1 = Decimal(0.0)
                    if firstSpaceIndex == -1:
                        floatOpernd1 = Decimal(line[0:firstSpaceIndex - 2])
                    else:
                        floatOpernd1 = Decimal(line[0:firstSpaceIndex - 1])
                    floatOpernd2 = Decimal(0.0)
                    if firstSpaceIndex != -1:
                        floatOpernd2 = Decimal(line[firstSpaceIndex + 1:firstCloseBracket - 1])
                    if firstSpaceIndex != -1:
                        floatVal = DataType.RealType(floatOpernd1 / floatOpernd2)
                    else:
                        floatVal = floatOpernd1
                    if sign == 1:
                        floatVal = -floatVal

                elif lines[i].find('(') != -1 and lines[i].find(')') == -1:
                    line = lines[i].strip(' ').strip('(').strip('/').strip(' ')
                    sign = 0
                    if line[0] == '-':
                        line = line.strip('-').strip(' ').strip('(').strip('/').strip(' ')
                        sign = 1

                    floatOpernd1 = DataType.RealType(line[0:len(line) - 1])
                    i += 1
                    line = lines[i].strip(' ').strip('(').strip('/').strip(' ')
                    firstCloseBracket = line.find(')')
                    floatOpernd2 = DataType.RealType(line[0:firstCloseBracket - 1])
                    floatVal = DataType.RealType(floatOpernd1 / floatOpernd2)
                    if sign == 1:
                        floatVal = -floatVal
                else:
                    line = lines[i].strip(' ').strip('\n').strip(')')
                    floatVal = DataType.RealType(line)
                i += 1
                # print(floatVal)
                dictInstance[intLayer][intId] = floatVal
            else:
                i += 2
        f.close()
        return dictInstance

    def outputRange(self) -> Set:
        """
        Compute range of a set of constraints
        :return: (objSet -> Set)
        """
        objSet: Set = None
        return objSet

    def __writeVariables__(self):
        """
        Declare SMT variables corresponding to gurobi model variables
        :return: None
        """
        listVarNames: List[str] = [var.VarName for var in self.__model__.getVars()]
        f = open("ce.smt2", "a")
        for var in listVarNames:
            if var.find('q') == -1:
                f.write("(declare-const " + var + " Real)\n")
            else:
                f.write("(declare-const " + var + " Bool)\n")
        f.close()

    def __writeConstraints__(self):
        """
        Write SMT constraints corresponding to gurobi model constraints
        :return: None
        """
        f = open("ce.smt2", "a")
        numVars: int = self.__model__.NumVars
        listVars: List[Var] = self.__model__.getVars()
        for constr in self.__model__.getConstrs():
            f.write('(assert ')
            listCoeff: List[float] = []
            constantTerm = 0.0
            for i in range(numVars):
                listCoeff.append(self.__model__.getCoeff(constr, listVars[i]))
            zeroCoeff = 0
            for i in range(numVars):
                if listCoeff[i] == 0.0:
                    zeroCoeff += 1
            constantTerm = constr.getAttr("rhs")
            operator = ''
            if constr.Sense == GRB.LESS_EQUAL:
                operator = "<="
            elif constr.Sense == GRB.GREATER_EQUAL:
                operator = ">="
            else:
                operator = "="

            f.write("(" + str(operator) + " ")
            for i in range(1, numVars - zeroCoeff, 1):
                f.write("(+")
            isFirst: bool = True
            for i in range(numVars):
                if listCoeff[i] == 0.0:
                    continue
                rational = Fraction(listCoeff[i]).limit_denominator()
                numerator = rational.numerator
                denominator = rational.denominator
                if isFirst:
                    if numerator < 0 or denominator < 0:
                        f.write("(* " + str('(/ ') + "(- " + str(-numerator) + ") " + str(denominator)
                                + ") " + listVars[i].VarName + ")")
                    else:
                        f.write("(* " + str('(/ ') + str(numerator) + " "+str(denominator)
                                + ") " + listVars[i].VarName + ")")
                    isFirst = False
                else:
                    if numerator < 0 or denominator < 0:
                        f.write(" (* " + str('(/ ') + "(- " + str(-numerator) + ") " + str(denominator)
                                + ") " + listVars[i].VarName + "))")
                    else:
                        f.write(" (* " + str('(/ ') + str(numerator) + " " + str(denominator)
                                + ") " + listVars[i].VarName + "))")

            # Convert into rational number
            constRational = Fraction(constantTerm).limit_denominator()
            constNumerator = constRational.numerator
            constDenominator = constRational.denominator
            if constantTerm < 0:
                f.write(" " + "(/ (- " + str(-constNumerator) + ") " + str(constDenominator)+")))\n")
                #f.write(" (- "+str(-constantTerm)+")))\n")
            else:
                f.write(" " + "(/ " + str(constNumerator) + " " + str(constDenominator)+")))\n")
                #f.write(" "+ str(constantTerm) + "))\n")
        f.write("(check-sat)\n")
        f.write("(get-model)\n")
        f.close()

    def __toSMT__(self):
        """
        Convert gurobi model to SMTLib
        :return: None
        """
        f = open("ce.smt2", 'w')
        f.close()
        self.__writeVariables__()
        self.__writeConstraints__()
