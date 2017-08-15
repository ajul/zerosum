import numpy
import unittest
import zerosum.nash
import tests.common

class TestNash(unittest.TestCase):
    def test_random(self):
        for i in range(tests.common.num_random_trials):
            payoff_matrix = numpy.random.normal(size = (tests.common.strategy_count, tests.common.strategy_count))
            row_result, col_result = zerosum.nash.nash(payoff_matrix)
            # Both results should have the same value (with opposite sign).
            numpy.testing.assert_allclose(row_result.value, -col_result.value)
            row_payoffs = numpy.average(payoff_matrix, axis = 1, weights = col_result.strategy)
            col_payoffs = numpy.average(payoff_matrix, axis = 0, weights = row_result.strategy)
            # Value should be close to the best possible expected payoff.
            numpy.testing.assert_allclose(numpy.max(row_payoffs), row_result.value)
            numpy.testing.assert_allclose(numpy.min(col_payoffs), -col_result.value)
