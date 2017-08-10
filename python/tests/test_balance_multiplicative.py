import numpy
import unittest
import warnings
import timeit
import zerosum.balance
import tests.balance_test_base

class TestMultiplicativeBalance(tests.balance_test_base.TestInitialMatrixNonSymmetricBalanceBase):
    class_to_test = zerosum.balance.MultiplicativeBalance
    
    def generate_random_args(self, rows, cols):
        value = numpy.random.rand() + 1.0
        initial_payoff_matrix = numpy.random.random((rows, cols))
        kwargs = {
            'value' : value,
            'initial_payoff_matrix' : initial_payoff_matrix,
        }
        return kwargs, value
    
    def test_negative_matrix_warning(self):
        data = -numpy.ones((2, 2))
        with self.assertWarnsRegex(zerosum.balance.ValueWarning, 'initial_payoff_matrix.*negative'):
            zerosum.balance.MultiplicativeBalance(data)