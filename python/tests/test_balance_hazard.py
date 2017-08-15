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
        initial_payoff_matrix = 0.1 + numpy.random.random((rows, cols))
        kwargs = {
            'initial_payoff_matrix' : initial_payoff_matrix,
            'value' : value,
        }
        return kwargs, value

class TestHazardSymmetricBalance(tests.common.TestInitialMatrixSymmetricBalanceBase):
    class_to_test = zerosum.balance.HazardSymmetricBalance
    
    def generate_random_args(self, rows):
        value = 0.0
        initial_payoff_matrix = 0.1 + numpy.random.random((rows, rows))
        initial_payoff_matrix_it = 1.0 / initial_payoff_matrix.transpose()
        initial_payoff_matrix = initial_payoff_matrix * initial_payoff_matrix_it
        kwargs = {
            'initial_payoff_matrix' : initial_payoff_matrix,
        }
        return kwargs, value
