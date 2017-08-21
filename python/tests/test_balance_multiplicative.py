import numpy
import unittest
import warnings
import timeit
import zerosum.balance
import tests.common

class TestMultiplicativeBalance(tests.common.TestNonSymmetricBalanceBase):
    class_to_test = zerosum.balance.MultiplicativeBalance
    
    def generate_random_args(self, rows, cols):
        value = numpy.random.rand() + 1.0
        base_matrix = numpy.random.random((rows, cols))
        kwargs = {
            'value' : value,
            'base_matrix' : base_matrix,
        }
        return kwargs, value
