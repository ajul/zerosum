import numpy
import unittest
import warnings
import timeit
import zerosum.balance
import tests.common

class TestLanchesterNonSymmetricBalance(tests.common.TestNonSymmetricBalanceBase):
    class_to_test = zerosum.balance.LanchesterNonSymmetricBalance
    
    def generate_random_args(self, rows, cols):
        value = 0.5 * (numpy.random.rand() - 0.5)
        base_matrix = 0.1 + numpy.random.random((rows, cols))
        exponent = 2.0
        kwargs = {
            'base_matrix' : base_matrix,
            'value' : value,
            'exponent' : exponent,
        }
        return kwargs, value

class TestLanchesterSymmetricBalance(tests.common.TestSymmetricBalanceBase):
    class_to_test = zerosum.balance.LanchesterSymmetricBalance
    
    def generate_random_args(self, rows):
        value = 0.0
        base_matrix = 0.1 + numpy.random.random((rows, rows))
        base_matrix_it = 1.0 / base_matrix.transpose()
        base_matrix = base_matrix * base_matrix_it
        exponent = 2.0
        kwargs = {
            'base_matrix' : base_matrix,
            'exponent' : exponent,
        }
        return kwargs, value
    
