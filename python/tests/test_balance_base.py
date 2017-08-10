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
    def test_fix_index_zero_weight_error(self):
        with self.assertWarnsRegex(zerosum.balance.ValueWarning, 'zero weight'):
            strategy_weights = numpy.array([0.0, 1.0])
            zerosum.balance.base.NonSymmetricBalance(row_weights = strategy_weights, col_weights = strategy_weights, fix_index = 0)

class TestSymmetricBalance(unittest.TestCase):

    def test_fix_index_zero_weight_error(self):
        with self.assertWarnsRegex(zerosum.balance.ValueWarning, 'zero weight'):
            strategy_weights = numpy.array([0.0, 1.0])
            zerosum.balance.base.SymmetricBalance(strategy_weights = strategy_weights, fix_index = 0)

