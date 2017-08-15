import numpy
import unittest
import warnings
import timeit
import zerosum.balance
import tests.common

"""
class TestLanchesterNonSymmetricBalance(tests.common.TestInitialMatrixNonSymmetricBalanceBase):
    class_to_test = zerosum.balance.LanchesterNonSymmetricBalance
    
    def generate_random_args(self, rows, cols):
        value = 0.5 * (numpy.random.rand() - 0.5)
        initial_payoff_matrix = 0.1 + numpy.random.random((rows, cols))
        kwargs = {
            'initial_payoff_matrix' : initial_payoff_matrix,
        }
        return kwargs, value
"""

class TestLanchesterSymmetricBalance(tests.common.TestInitialMatrixSymmetricBalanceBase):
    class_to_test = zerosum.balance.LanchesterSymmetricBalance
    
    def generate_random_args(self, rows):
        value = 0.0
        initial_payoff_matrix = 0.1 + numpy.random.random((rows, rows))
        initial_payoff_matrix_it = 1.0 / initial_payoff_matrix.transpose()
        initial_payoff_matrix = initial_payoff_matrix * initial_payoff_matrix_it
        kwargs = {
            'initial_payoff_matrix' : initial_payoff_matrix,
        }
        return kwargs, value
    
