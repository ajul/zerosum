import numpy
import unittest
import warnings
import timeit
import zerosum.balance
import tests.common

class TestMultiplicativeBalance(tests.common.TestInitialMatrixNonSymmetricBalanceBase):
    class_to_test = zerosum.balance.MultiplicativeBalance
    
    def generate_random_args(self, rows, cols):
        value = numpy.random.rand() + 1.0
        base_matrix = numpy.random.random((rows, cols))
        kwargs = {
            'value' : value,
            'base_matrix' : base_matrix,
        }
        return kwargs, value
    
    def test_negative_matrix_warning(self):
        data = -numpy.ones((2, 2))
        with self.assertRaisesRegex(ValueError, 'base_matrix.*negative'):
            zerosum.balance.MultiplicativeBalance(data)