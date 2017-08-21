import numpy
import unittest
import warnings
import timeit

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

class TestInitialMatrixNonSymmetricBalanceBase(unittest.TestCase):
    """The NonSymmetricBalance subclass to be tested."""
    class_to_test = None

    def generate_random_args(self, rows, cols):
        """
        Returns:
            kwargs
            value
        """
        raise NotImplementedError
        
    def check_result_value(self, result, value, row_weights = None, col_weights = None):
        try:
            self.assertTrue(result.success)
            numpy.testing.assert_allclose(numpy.average(result.payoff_matrix, weights = row_weights, axis = 0), value, atol = solution_atol)
            numpy.testing.assert_allclose(numpy.average(result.payoff_matrix, weights = col_weights, axis = 1), value, atol = solution_atol)
        except Exception as e:
            print(result)
            raise e

    def test_check_derivative(self):
        print()
        row_weights = random_weights_with_zeros(strategy_count)
        col_weights = random_weights_with_zeros(strategy_count + 1)
        kwargs, value = self.generate_random_args(strategy_count, strategy_count + 1)
        result = self.class_to_test(row_weights = row_weights, col_weights = col_weights, **kwargs).optimize(check_derivative = True)
        
    def test_check_jacobian(self):
        print()
        row_weights = random_weights_with_zeros(strategy_count)
        col_weights = random_weights_with_zeros(strategy_count + 1)
        kwargs, value = self.generate_random_args(strategy_count, strategy_count + 1)
        result = self.class_to_test(row_weights = row_weights, col_weights = col_weights, **kwargs).optimize(check_jacobian = True)

    def test_random_unweighted(self):
        for i in range(num_random_trials):
            kwargs, value = self.generate_random_args(strategy_count, strategy_count + 1)
            result = self.class_to_test(**kwargs).optimize(tol = solver_tol)
            self.check_result_value(result, value)
        
    def test_random_weighted(self):
        for i in range(num_random_trials):
            row_weights = random_weights(strategy_count)
            col_weights = random_weights(strategy_count + 1)
            kwargs, value = self.generate_random_args(strategy_count, strategy_count + 1)
            result = self.class_to_test(row_weights = row_weights, col_weights = col_weights, **kwargs).optimize(tol = solver_tol)
            self.check_result_value(result, value, row_weights, col_weights)
        
    def test_random_weighted_with_zeros(self):
        for i in range(num_random_trials):
            row_weights = random_weights_with_zeros(strategy_count)
            col_weights = random_weights_with_zeros(strategy_count + 1)
            kwargs, value = self.generate_random_args(strategy_count, strategy_count + 1)
            result = self.class_to_test(row_weights = row_weights, col_weights = col_weights, **kwargs).optimize(tol = solver_tol)
            self.check_result_value(result, value, row_weights, col_weights)

class TestInitialMatrixSymmetricBalanceBase(unittest.TestCase):
    """The SymmetricBalance subclass to be tested."""
    class_to_test = None
    
    def generate_random_args(self, rows):
        """
        Returns:
            kwargs
            value
        """
        raise NotImplementedError
        
    def check_result_value(self, result, value, strategy_weights = None):
        try:
            self.assertTrue(result.success)
            numpy.testing.assert_allclose(numpy.average(result.payoff_matrix, weights = strategy_weights, axis = 0), value, atol = solution_atol)
            numpy.testing.assert_allclose(numpy.average(result.payoff_matrix, weights = strategy_weights, axis = 1), value, atol = solution_atol)
        except Exception as e:
            print(result)
            raise e

    def test_check_derivative(self):
        print()
        kwargs, value = self.generate_random_args(strategy_count)
        strategy_weights = random_weights(strategy_count)
        result = self.class_to_test(strategy_weights = strategy_weights, **kwargs).optimize(check_derivative = True)
        
    def test_check_jacobian(self):
        print()
        kwargs, value = self.generate_random_args(strategy_count)
        strategy_weights = random_weights(strategy_count)
        result = self.class_to_test(strategy_weights = strategy_weights, **kwargs).optimize(check_jacobian = True)
    
    def test_random_unweighted(self):
        for i in range(num_random_trials):
            kwargs, value = self.generate_random_args(strategy_count)
            balance = self.class_to_test(**kwargs)
            result = balance.optimize(tol = solver_tol)
            self.check_result_value(result, value)
        
    def test_random_weighted(self):
        for i in range(num_random_trials):
            strategy_weights = random_weights(strategy_count)
            kwargs, value = self.generate_random_args(strategy_count)
            balance = self.class_to_test(strategy_weights = strategy_weights, **kwargs)
            result = balance.optimize(tol = solver_tol)
            self.check_result_value(result, value, strategy_weights)
    
    def test_random_weighted_with_zeros(self):
        for i in range(num_random_trials):
            strategy_weights = random_weights_with_zeros(strategy_count)
            kwargs, value = self.generate_random_args(strategy_count)
            balance = self.class_to_test(strategy_weights = strategy_weights, **kwargs)
            result = balance.optimize(tol = solver_tol)
            self.check_result_value(result, value, strategy_weights)