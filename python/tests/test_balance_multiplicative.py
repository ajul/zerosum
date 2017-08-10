import numpy
import unittest
import warnings
import timeit
import zerosum.balance
import tests.balance_base

class TestMultiplicativeBalance(tests.balance_base.TestInitialMatrixNonSymmetricBalanceBase):
    class_to_test = zerosum.balance.MultiplicativeBalance
    
    def generate_random_matrix(self, rows, cols):
        return numpy.random.random((rows, cols))
    
    def test_negative_matrix_warning(self):
        data = -numpy.ones((2, 2))
        with self.assertWarnsRegex(zerosum.balance.ValueWarning, 'initial_payoff_matrix.*negative'):
            zerosum.balance.MultiplicativeBalance(data)
    
    def test_weight_count_error(self):
        data = numpy.ones((2, 2))
        with self.assertRaisesRegex(ValueError, 'size of row_weights'):
            zerosum.balance.MultiplicativeBalance(data, numpy.ones((3,)), numpy.ones((2,)))
        with self.assertRaisesRegex(ValueError, 'size of col_weights'):
            zerosum.balance.MultiplicativeBalance(data, numpy.ones((2,)), numpy.ones((3,)))