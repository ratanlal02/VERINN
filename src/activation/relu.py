"""
Author: Ratan Lal
Date : January 28, 2025
"""
from src.types.datatype import DataType


class Relu:
    """
    perform the computation of different activation functions
    """

    @staticmethod
    def point(floatValue: DataType.RealType) -> DataType.RealType:
        """
        compute relu of a real value
        :param floatValue: a real value
        :type floatValue: float
        :return: (floatRelu -> float)
        """
        floatRelu: DataType.RealType = DataType.RealType(0.0)
        if floatValue > 0.0:
            floatRelu = floatValue

        # return relu of floatValue
        return floatRelu