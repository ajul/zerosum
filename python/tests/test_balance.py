import numpy
import unittest
import warnings
import timeit
import zerosum.balance
import tests.balance_base

strategy_count = 32
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

class TestMultiplicativeBalance(tests.balance_base.TestInitialMatrixNonSymmetricBalanceBase):
    class_to_test = zerosum.balance.MultiplicativeBalance
    
    def generate_random_matrix(self, rows, cols):
        return numpy.random.random((rows, cols))
    
    def test_negative_matrix_warning(self):
        data = -numpy.ones((2, 2))
        with self.assertWarnsRegex(zerosum.balance.ValueWarning, 'initial_payoff_matrix.*negative'):
            zerosum.balance.MultiplicativeBalance(data)

class TestLogisticSymmetricBalance(unittest.TestCase):
    def random_data(self, n):
        data = numpy.random.random((n, n))
        data_nt = 1.0 - data
        data = (1.0 + numpy.random.rand()) * (data + data_nt)
        return data

    def test_non_skew_symmetric_warning(self):
        data = numpy.eye(2) + 0.5
        with self.assertWarnsRegex(zerosum.balance.ValueWarning, 'skew-symmetric'):
            zerosum.balance.LogisticSymmetricBalance(data)
            
    def test_nonsquare_error(self):
        data = numpy.ones((2, 3))
        with self.assertRaisesRegex(ValueError, 'square'):
            zerosum.balance.LogisticSymmetricBalance(data)
            
    def test_weight_count_error(self):
        data = self.random_data(2)
        strategy_weights = numpy.ones((3,))
        with self.assertRaisesRegex(ValueError, 'size of strategy_weights'):
            zerosum.balance.LogisticSymmetricBalance(data, strategy_weights)
            
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
    
    def test_check_derivative(self):
        print()
        data = self.random_data(strategy_count)
        strategy_weights = random_weights(strategy_count)
        result = zerosum.balance.LogisticSymmetricBalance(data, strategy_weights).optimize(check_derivative = True)
        
    def test_check_jacobian(self):
        print()
        data = self.random_data(strategy_count)
        strategy_weights = random_weights(strategy_count)
        result = zerosum.balance.LogisticSymmetricBalance(data, strategy_weights).optimize(check_jacobian = True)
    
    def test_random_unweighted(self):
        for i in range(num_random_trials):
            data = self.random_data(strategy_count)
            balance = zerosum.balance.LogisticSymmetricBalance(data)
            result = balance.optimize(tol = solver_tol)
            self.assertTrue(result.success)
            numpy.testing.assert_allclose(numpy.average(result.F, axis = 0), balance.max_payoff * 0.5, atol = solution_atol)
            numpy.testing.assert_allclose(numpy.average(result.F, axis = 1), balance.max_payoff * 0.5, atol = solution_atol)
        
    def test_random_weighted(self):
        for i in range(num_random_trials):
            strategy_weights = random_weights(strategy_count)
            data = self.random_data(strategy_count)
            balance = zerosum.balance.LogisticSymmetricBalance(data, strategy_weights)
            result = balance.optimize(tol = solver_tol)
            self.assertTrue(result.success)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 0), balance.max_payoff * 0.5, atol = solution_atol)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 1), balance.max_payoff * 0.5, atol = solution_atol)
    
    def test_random_weighted_with_zeros(self):
        for i in range(num_random_trials):
            strategy_weights = random_weights_with_zeros(strategy_count)
            data = self.random_data(strategy_count)
            balance = zerosum.balance.LogisticSymmetricBalance(data, strategy_weights)
            result = balance.optimize(tol = solver_tol)
            self.assertTrue(result.success)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 0), balance.max_payoff * 0.5, atol = solution_atol)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 1), balance.max_payoff * 0.5, atol = solution_atol)
