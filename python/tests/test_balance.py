import numpy
import unittest
import warnings
import zerosum.balance

strategy_count = 8
solver_tol = 1e-7
solution_atol = 1e-6
num_random_trials = 100

def random_weights(n):
    result = numpy.random.random((n,))
    return result

def random_weights_with_zeros(n):
    result = numpy.random.random((n,))
    result[:n//2] = 0.0
    numpy.random.shuffle(result)
    return result
    
class TestWeights(unittest.TestCase):
    def test_zero_weight_sum_error(self):
        weights = numpy.zeros((strategy_count,))
        with self.assertRaisesRegex(ValueError, 'sum to 0'):
            zerosum.balance._process_weights(weights)
    
    def test_negative_weight_error(self):
        weights = -numpy.ones((strategy_count,))
        with self.assertRaisesRegex(ValueError, 'negative'):
            zerosum.balance._process_weights(weights)

class TestMultiplicativeBalance(unittest.TestCase):
    value = 1.0
    def test_negative_matrix_warning(self):
        data = -numpy.ones((2, 2))
        with self.assertWarnsRegex(zerosum.balance.ValueWarning, 'initial_payoff_matrix.*negative'):
            zerosum.balance.MultiplicativeBalance(data)
        
    def test_random_unweighted(self):
        for i in range(num_random_trials):
            data = numpy.random.random((strategy_count, strategy_count))
            result = zerosum.balance.MultiplicativeBalance(data, value = self.value).optimize(tol = solver_tol)
            numpy.testing.assert_allclose(numpy.average(result.F, axis = 0), self.value, atol = solution_atol)
            numpy.testing.assert_allclose(numpy.average(result.F, axis = 1), self.value, atol = solution_atol)
        
    def test_random_weighted(self):
        for i in range(num_random_trials):
            row_weights = random_weights(strategy_count)
            col_weights = random_weights(strategy_count + 1)
            data = numpy.random.random((strategy_count, strategy_count + 1))
            result = zerosum.balance.MultiplicativeBalance(data, row_weights, col_weights, value = self.value).optimize(tol = solver_tol)

            numpy.testing.assert_allclose(numpy.average(result.F, weights = row_weights, axis = 0), self.value, atol = solution_atol)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = col_weights, axis = 1), self.value, atol = solution_atol)
        
    def test_random_weighted_with_zeros(self):
        for i in range(num_random_trials):
            row_weights = random_weights_with_zeros(strategy_count)
            col_weights = random_weights_with_zeros(strategy_count + 1)
            data = numpy.random.random((strategy_count, strategy_count +1))
            result = zerosum.balance.MultiplicativeBalance(data, row_weights, col_weights, value = self.value).optimize(tol = solver_tol)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = row_weights, axis = 0), self.value, atol = solution_atol)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = col_weights, axis = 1), self.value, atol = solution_atol)

class TestLogisticSymmetricBalance(unittest.TestCase):
    def test_non_skew_symmetric(self):
        data = numpy.eye(2) + 0.5
        with self.assertWarnsRegex(zerosum.balance.ValueWarning, 'skew-symmetric'):
            zerosum.balance.LogisticSymmetricBalance(data)
            
    def test_saturation(self):
        data = numpy.array([[0.5, 1.0], 
                            [0.0, 0.5]])
        with self.assertRaisesRegex(ValueError, 'open interval'):
            zerosum.balance.LogisticSymmetricBalance(data)
            
    def test_near_saturation(self):
        epsilon = 1e-12
        data = numpy.array([[0.5, 1.0 - epsilon], 
                            [epsilon, 0.5]])
        with self.assertWarnsRegex(zerosum.balance.ValueWarning, 'close to'):
            zerosum.balance.LogisticSymmetricBalance(data)
    
    def test_random_unweighted(self):
        for i in range(num_random_trials):
            data = numpy.random.random((strategy_count, strategy_count))
            data_nt = 1.0 - data
            data = 0.5 * (data + data_nt)
            result = zerosum.balance.LogisticSymmetricBalance(data).optimize(tol = solver_tol)
            numpy.testing.assert_allclose(numpy.average(result.F, axis = 0), 0.5, atol = solution_atol)
            numpy.testing.assert_allclose(numpy.average(result.F, axis = 1), 0.5, atol = solution_atol)
        
    def test_random_weighted(self):
        for i in range(num_random_trials):
            strategy_weights = random_weights(strategy_count)
            data = numpy.random.random((strategy_count, strategy_count))
            data_nt = 1.0 - data
            data = 0.5 * (data + data_nt)
            result = zerosum.balance.LogisticSymmetricBalance(data, strategy_weights).optimize(tol = solver_tol)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 0), 0.5, atol = solution_atol)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 1), 0.5, atol = solution_atol)
    
    def test_random_weighted_with_zeros(self):
        for i in range(num_random_trials):
            strategy_weights = random_weights_with_zeros(strategy_count)
            data = numpy.random.random((strategy_count * 2, strategy_count * 2))
            data_nt = 1.0 - data
            data = 0.5 * (data + data_nt)
            result = zerosum.balance.LogisticSymmetricBalance(data, strategy_weights).optimize(tol = solver_tol)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 0), 0.5, atol = solution_atol)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 1), 0.5, atol = solution_atol)
    
if __name__ == '__main__':
    unittest.main()
