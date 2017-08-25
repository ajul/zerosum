import numpy
import unittest
import warnings
import timeit
import zerosum.balance

strategy_count = 32
    
class TestWeights(unittest.TestCase):
    def test_zero_weight_sum_error(self):
        weights = numpy.zeros((strategy_count,))
        with self.assertRaisesRegex(ValueError, 'sum to 0'):
            zerosum.balance.base._process_weights(weights)
    
    def test_negative_weight_error(self):
        weights = -numpy.ones((strategy_count,))
        with self.assertRaisesRegex(ValueError, 'negative'):
            zerosum.balance.base._process_weights(weights)

class TestNonSymmetricBalance(unittest.TestCase):
    pass

class TestSymmetricBalance(unittest.TestCase):
    pass
# TODO: add input checking tests