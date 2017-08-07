import numpy
import unittest
import warnings
import timeit
import zerosum.balance

strategy_count = 64
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
    def dummy_handicap_function(self, x):
        return numpy.ones(x.size // 2)

    def test_fix_index_zero_weight_error(self):
        with self.assertWarnsRegex(zerosum.balance.ValueWarning, 'zero weight'):
            strategy_weights = numpy.array([0.0, 1.0])
            zerosum.balance.base.NonSymmetricBalance(self.dummy_handicap_function, row_weights = strategy_weights, col_weights = strategy_weights, fix_index = 0)

class TestSymmetricBalance(unittest.TestCase):
    def dummy_handicap_function(self, x):
        return numpy.ones(x.size)

    def test_fix_index_zero_weight_error(self):
        with self.assertWarnsRegex(zerosum.balance.ValueWarning, 'zero weight'):
            strategy_weights = numpy.array([0.0, 1.0])
            zerosum.balance.base.SymmetricBalance(self.dummy_handicap_function, strategy_weights = strategy_weights, fix_index = 0)

class TestMultiplicativeBalance(unittest.TestCase):
    def test_negative_matrix_warning(self):
        data = -numpy.ones((2, 2))
        with self.assertWarnsRegex(zerosum.balance.ValueWarning, 'initial_payoff_matrix.*negative'):
            zerosum.balance.MultiplicativeBalance(data)
             
    def test_weight_count_error(self):
        data = numpy.ones((2, 2))
        with self.assertRaisesRegex(ValueError, 'size of row_weights'):
            zerosum.balance.MultiplicativeBalance(data, numpy.ones((3,)), numpy.ones((2,)))
        with self.assertRaisesRegex(ValueError, 'size of col_weights'):
            zerosum.balance.MultiplicativeBalance(data, numpy.ones((2,)), numpy.ones((3,)))
            
            
    def test_check_derivative(self):
        print()
        value = numpy.random.rand() + 1.0
        row_weights = random_weights_with_zeros(strategy_count)
        col_weights = random_weights_with_zeros(strategy_count + 1)
        data = numpy.random.random((strategy_count, strategy_count +1))
        result = zerosum.balance.MultiplicativeBalance(data, row_weights, col_weights, value = value).optimize(check_derivative = True)
        
    def test_check_jacobian(self):
        print()
        value = numpy.random.rand() + 1.0
        row_weights = random_weights_with_zeros(strategy_count)
        col_weights = random_weights_with_zeros(strategy_count + 1)
        data = numpy.random.random((strategy_count, strategy_count +1))
        result = zerosum.balance.MultiplicativeBalance(data, row_weights, col_weights, value = value).optimize(check_jacobian = True)
        
    def test_random_unweighted(self):
        for i in range(num_random_trials):
            value = numpy.random.rand() + 1.0
            data = numpy.random.random((strategy_count, strategy_count))
            result = zerosum.balance.MultiplicativeBalance(data, value = value).optimize(tol = solver_tol)
            numpy.testing.assert_allclose(numpy.average(result.F, axis = 0), value, atol = solution_atol)
            numpy.testing.assert_allclose(numpy.average(result.F, axis = 1), value, atol = solution_atol)
        
    def test_random_weighted(self):
        for i in range(num_random_trials):
            value = numpy.random.rand() + 1.0
            row_weights = random_weights(strategy_count)
            col_weights = random_weights(strategy_count + 1)
            data = numpy.random.random((strategy_count, strategy_count + 1))
            result = zerosum.balance.MultiplicativeBalance(data, row_weights, col_weights, value = value).optimize(tol = solver_tol)

            numpy.testing.assert_allclose(numpy.average(result.F, weights = row_weights, axis = 0), value, atol = solution_atol)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = col_weights, axis = 1), value, atol = solution_atol)
        
    def test_random_weighted_with_zeros(self):
        for i in range(num_random_trials):
            value = numpy.random.rand() + 1.0
            row_weights = random_weights_with_zeros(strategy_count)
            col_weights = random_weights_with_zeros(strategy_count + 1)
            data = numpy.random.random((strategy_count, strategy_count +1))
            result = zerosum.balance.MultiplicativeBalance(data, row_weights, col_weights, value = value).optimize(tol = solver_tol)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = row_weights, axis = 0), value, atol = solution_atol)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = col_weights, axis = 1), value, atol = solution_atol)
            
    @staticmethod
    def time_trial(method):
        value = numpy.random.rand() + 1.0
        row_weights = random_weights(strategy_count)
        col_weights = random_weights(strategy_count + 1)
        data = numpy.random.random((strategy_count, strategy_count + 1))
        result = zerosum.balance.MultiplicativeBalance(data, row_weights, col_weights, value = value).optimize(tol = solver_tol)
        
    def test_method_timing(self):
        """ Very crude test for comparing the performance of different optimization methods. """
        print()
        for method in ['hybr', 'lm']:
            t = timeit.timeit("TestMultiplicativeBalance.time_trial('%s')" % method, number=num_random_trials, globals=globals())
            print("Method '%s' took %f s." % (method, t / num_random_trials))

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
    
    @staticmethod
    def time_trial(method):
        strategy_weights = random_weights(strategy_count)
        data = self.random_data(strategy_count)
        balance = zerosum.balance.LogisticSymmetricBalance(data, strategy_weights)
        result = balance.optimize(tol = solver_tol)
    
    def test_method_timing(self):
        """ Very crude test for comparing the performance of different optimization methods. """
        print()
        for method in ['hybr', 'lm']:
            t = timeit.timeit("TestMultiplicativeBalance.time_trial('%s')" % method, number=num_random_trials, globals=globals())
            print("Method '%s' took %f s." % (method, t / num_random_trials))