from fractions import Fraction
from math import gcd
from functools import reduce


class DATI:
    @staticmethod
    def lcm(numbers):
        return reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers)

    # ----------------------------------------------------------------------------------------------#
    @staticmethod
    def proportional_integer_vector(v):
        # print v
        # Determine the list for numerators and denominators
        numerator_list = []
        denominator_list = []

        for value in v:
            numerator_list.append(int(value.numerator))
            denominator_list.append(int(value.denominator))

        # Least common multiple of the denominators
        least_cm = DATI.lcm(denominator_list)

        integer_list = [int(least_cm * a / b) for a, b in zip(numerator_list, denominator_list)]

        # Divide every integer by the greatest common divisor to obtain the smallest possible values
        GCD = reduce(gcd, integer_list)
        for i in range(len(integer_list)):
            integer_list[i] = integer_list[i] / GCD

        return integer_list

    # ----------------------------------------------------------------------------------------------#
    @staticmethod
    def real2integer(real_array):
        """ Given a list of reals it returns a list of integers. """
        # print "real"
        # print real_array
        rational = []
        for value in real_array:
            rational.append(Fraction(value))
        # print 'rational solution =',rational

        integer = DATI.proportional_integer_vector(rational)
        # print 'integer solution =',integer

        return integer
