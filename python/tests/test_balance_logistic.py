import numpy
import unittest
import warnings
import timeit
import zerosum.balance
import tests.common

class TestLogisticNonSymmetricBalance(tests.common.TestNonSymmetricBalanceBase):
    class_to_test = zerosum.balance.LogisticNonSymmetricBalance
    
    def generate_random_args(self, rows, cols):
        max_payoff = 1.0 + numpy.random.rand()
        value = max_payoff * (0.25 + 0.5 * numpy.random.rand())
        base_matrix = max_payoff * (0.1 + 0.8 * numpy.random.random((rows, cols)))
        kwargs = {
            'value' : value,
            'max_payoff': max_payoff,
            'base_matrix' : base_matrix,
        }
        return kwargs, value

class TestLogisticSymmetricBalance(tests.common.TestSymmetricBalanceBase):
    class_to_test = zerosum.balance.LogisticSymmetricBalance
    
    def generate_random_args(self, rows):
        value = 1.0 + numpy.random.rand()
        base_matrix = numpy.random.random((rows, rows))
        base_matrix_nt = 1.0 - base_matrix.transpose()
        base_matrix = value * (base_matrix + base_matrix_nt)
        kwargs = {
            'base_matrix' : base_matrix,
        }
        return kwargs, value
            
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
    
