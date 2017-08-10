import numpy
import unittest
import warnings
import timeit
import zerosum.balance
import tests.balance_test_base

"""
class TestLanchesterNonSymmetricBalance(tests.balance_test_base.TestInitialMatrixNonSymmetricBalanceBase):
    class_to_test = zerosum.balance.LanchesterNonSymmetricBalance
    
    def generate_random_args(self, rows, cols):
        value = 0.5 * (numpy.random.rand() - 0.5)
        initial_payoff_matrix = numpy.random.random((rows, cols))
        kwargs = {
            'initial_payoff_matrix' : initial_payoff_matrix,
        }
        return kwargs, value
"""

class TestLanchesterSymmetricBalance(tests.balance_test_base.TestInitialMatrixSymmetricBalanceBase):
    class_to_test = zerosum.balance.LanchesterSymmetricBalance
    
    def generate_random_args(self, rows):
        value = 0.0
        initial_payoff_matrix = numpy.random.random((rows, rows))
        kwargs = {
            'initial_payoff_matrix' : initial_payoff_matrix,
        }
        return kwargs, value
    
