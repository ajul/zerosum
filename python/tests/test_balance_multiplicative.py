import numpy
import unittest
import warnings
import timeit
import zerosum.balance
import tests.balance_base

class TestMultiplicativeBalance(tests.balance_base.TestInitialMatrixNonSymmetricBalanceBase):
    class_to_test = zerosum.balance.MultiplicativeBalance
    
    def generate_random_data(self, rows, cols):
        initial_payoff_matrix = numpy.random.random((rows, cols))
        value = numpy.random.rand() + 1.0
        return initial_payoff_matrix, value
    
    def test_negative_matrix_warning(self):
        data = -numpy.ones((2, 2))
        with self.assertWarnsRegex(zerosum.balance.ValueWarning, 'initial_payoff_matrix.*negative'):
            zerosum.balance.MultiplicativeBalance(data)