import os
import sys
import re
from src.set.doubleArrayToInteger import DATI
import numpy as np


class DTIE:
    # This transform the expression into real coefficient without division operator
    @staticmethod
    def Realexp_wo_denominator(expr):
        """
        input:  expr - '1.6x+2.5y<=5' (inequality with real coefficients)
        output: expr - '16x+25y <= 50' (inequality with integer coefficients)
        """
        length = len(expr)
        L1 = expr
        L2 = []
        unit = ''
        flag = True
        preUnit = ''
        for i in range(length):
            if (L1[i].isdigit() or L1[i] == '.'):
                flag = True
                unit += L1[i]
            else:
                if (DTIE.isOperator(L1[i])):
                    Flag = True
                    if (unit == ''):
                        L2.append(L1[i])
                    else:
                        L2.append(unit)
                        L2.append(L1[i])
                        unit = ''
                else:
                    if (flag == True):
                        if (i == 0):
                            L2.append('1')
                            L2.append('*')
                        else:
                            if (not (L1[i - 1] == '*') and not (L1[i - 1].isdigit())):
                                L2.append('1')
                                L2.append('*')
                            else:
                                L2.append(unit)
                        unit = L1[i]
                        flag = False
                    else:
                        unit += L1[i]
        L2.append(unit)

        # Log.message(str(L2)+'\n')

        L3 = []
        for i in range(len(L2)):
            if (DTIE.contains_digits(L2[i])):
                L3.append(L2[i])
        # Log.message("L3\n")
        # Log.message(str(L3) + '\n')
        #print(L3)
        L4 = DATI.real2integer(L3)
        # Log.message("L4\n")
        # Log.message(str(L4) + '\n')
        exp = ''
        k = 0
        for i in range(len(L2)):
            # Log.message(str(L2[i])+"\n")
            if (DTIE.contains_digits(L2[i])):
                '''
                if (i != len(L2) - 1):
                    if (L4[k] == 1 and not (contains_digits(L2[i + 1])) and L2[i + 1] != '*'):
                        k = k + 1
                        continue
                '''
                exp += str(np.format_float_positional(L4[k], trim='-'))
                k = k + 1
            else:
                exp += L2[i]
            # Log.message(exp+"\n")
        # Log.message(exp+'\n')
        return exp

    # string contains digit	checking
    @staticmethod
    def contains_digits(d):
        try:
            s = d.split('_')
            digit = re.compile('\d')
            return bool(digit.search(s[0]))
        except SyntaxError:
            print
            "Error in function contains_digits"

    # string is operator
    @staticmethod
    def isOperator(op):
        try:
            if (
                    op == '+' or op == '-' or op == '(' or op == ')' or op == '*' or op == '>' or op == '<' or op == '=' or op == '==' or op == ' ' or op == '*'):
                return True
            else:
                return False
        except SyntaxError:
            print
            "Error in function isOperator"
