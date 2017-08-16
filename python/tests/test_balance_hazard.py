import numpy
import unittest
import warnings
import timeit
import zerosum.balance
import tests.common

class TestHazardNonSymmetricBalance(tests.common.TestInitialMatrixNonSymmetricBalanceBase):
    class_to_test = zerosum.balance.HazardNonSymmetricBalance
    
    def generate_random_args(self, rows, cols):
        value = 1.0 * (numpy.random.rand() - 0.5)
        base_matrix = 0.1 + numpy.random.random((rows, cols))
        kwargs = {
            'base_matrix' : base_matrix,
            'value' : value,
        }
        return kwargs, value

class TestHazardSymmetricBalance(tests.common.TestInitialMatrixSymmetricBalanceBase):
    class_to_test = zerosum.balance.HazardSymmetricBalance
    
    def generate_random_args(self, rows):
        value = 0.0
        base_matrix = 0.1 + numpy.random.random((rows, rows))
        base_matrix_it = 1.0 / base_matrix.transpose()
        base_matrix = base_matrix * base_matrix_it
        kwargs = {
            'base_matrix' : base_matrix,
        }
        return kwargs, value
