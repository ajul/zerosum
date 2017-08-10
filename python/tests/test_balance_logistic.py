import numpy
import unittest
import warnings
import timeit
import zerosum.balance
import tests.balance_test_base

class TestLogisticNonSymmetricBalance(tests.balance_test_base.TestInitialMatrixNonSymmetricBalanceBase):
    class_to_test = zerosum.balance.LogisticNonSymmetricBalance
    
    def generate_random_args(self, rows, cols):
        max_payoff = 1.0 + numpy.random.rand()
        value = max_payoff * (0.25 + 0.5 * numpy.random.rand())
        initial_payoff_matrix = max_payoff * (0.1 + 0.8 * numpy.random.random((rows, cols)))
        kwargs = {
            'value' : value,
            'max_payoff': max_payoff,
            'initial_payoff_matrix' : initial_payoff_matrix,
        }
        return kwargs, value

class TestLogisticSymmetricBalance(tests.balance_test_base.TestInitialMatrixSymmetricBalanceBase):
    class_to_test = zerosum.balance.LogisticSymmetricBalance
    
    def generate_random_args(self, rows):
        value = 1.0 + numpy.random.rand()
        initial_payoff_matrix = numpy.random.random((rows, rows))
        initial_payoff_matrix_nt = 1.0 - initial_payoff_matrix.transpose()
        initial_payoff_matrix = value * (initial_payoff_matrix + initial_payoff_matrix_nt)
        kwargs = {
            'initial_payoff_matrix' : initial_payoff_matrix,
        }
        return kwargs, value

    def test_non_skew_symmetric_warning(self):
        data = numpy.eye(2) + 0.5
        with self.assertWarnsRegex(zerosum.balance.ValueWarning, 'skew-symmetric'):
            zerosum.balance.LogisticSymmetricBalance(data)
            
    def test_nonsquare_error(self):
        data = numpy.ones((2, 3))
        with self.assertRaisesRegex(ValueError, 'square'):
            zerosum.balance.LogisticSymmetricBalance(data)
            
    def test_saturation_error(self):
        data = numpy.array([[0.5, 1.0], 
                            [0.0, 0.5]])
        with self.assertRaisesRegex(ValueError, 'open interval'):
            zerosum.balance.LogisticSymmetricBalance(data)
            
    def test_near_saturation_warning(self):
        epsilon = 1e-12
        data = numpy.array([[0.5, 1.0 - epsilon], 
                            [epsilon, 0.5]])
        with self.assertWarnsRegex(zerosum.balance.ValueWarning, 'close to'):
            zerosum.balance.LogisticSymmetricBalance(data)
    
