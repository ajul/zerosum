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

    def generate_random_data(self, rows, cols):
        """
        Returns:
            initial_payoff_matrix: Of size rows x cols.
            value: Desired value of the game.
        """
        raise NotImplementedError

    def test_check_derivative(self):
        print()
        row_weights = random_weights_with_zeros(strategy_count)
        col_weights = random_weights_with_zeros(strategy_count + 1)
        initial_payoff_matrix, value = self.generate_random_data(strategy_count, strategy_count + 1)
        result = self.class_to_test(initial_payoff_matrix, row_weights, col_weights, value = value).optimize(check_derivative = True)
        
    def test_check_jacobian(self):
        print()
        row_weights = random_weights_with_zeros(strategy_count)
        col_weights = random_weights_with_zeros(strategy_count + 1)
        initial_payoff_matrix, value = self.generate_random_data(strategy_count, strategy_count + 1)
        result = self.class_to_test(initial_payoff_matrix, row_weights, col_weights, value = value).optimize(check_jacobian = True)

    def test_random_unweighted(self):
        for i in range(num_random_trials):
            initial_payoff_matrix, value = self.generate_random_data(strategy_count, strategy_count + 1)
            result = self.class_to_test(initial_payoff_matrix, value = value).optimize(tol = solver_tol)
            numpy.testing.assert_allclose(numpy.average(result.F, axis = 0), value, atol = solution_atol)
            numpy.testing.assert_allclose(numpy.average(result.F, axis = 1), value, atol = solution_atol)
        
    def test_random_weighted(self):
        for i in range(num_random_trials):
            row_weights = random_weights(strategy_count)
            col_weights = random_weights(strategy_count + 1)
            initial_payoff_matrix, value = self.generate_random_data(strategy_count, strategy_count + 1)
            result = self.class_to_test(initial_payoff_matrix, row_weights, col_weights, value = value).optimize(tol = solver_tol)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = row_weights, axis = 0), value, atol = solution_atol)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = col_weights, axis = 1), value, atol = solution_atol)
        
    def test_random_weighted_with_zeros(self):
        for i in range(num_random_trials):
            row_weights = random_weights_with_zeros(strategy_count)
            col_weights = random_weights_with_zeros(strategy_count + 1)
            initial_payoff_matrix, value = self.generate_random_data(strategy_count, strategy_count + 1)
            result = self.class_to_test(initial_payoff_matrix, row_weights, col_weights, value = value).optimize(tol = solver_tol)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = row_weights, axis = 0), value, atol = solution_atol)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = col_weights, axis = 1), value, atol = solution_atol)

class TestInitialMatrixSymmetricBalanceBase(unittest.TestCase):
    """The SymmetricBalance subclass to be tested."""
    class_to_test = None
    
    def generate_random_data(self, rows):
        """
        Returns:
            initial_payoff_matrix: Square matrix of size equal to rows.
            value: Desired value of the game.
        """
        raise NotImplementedError

    def test_check_derivative(self):
        print()
        initial_payoff_matrix, value = self.generate_random_data(strategy_count)
        strategy_weights = random_weights(strategy_count)
        result = self.class_to_test(initial_payoff_matrix, strategy_weights).optimize(check_derivative = True)
        
    def test_check_jacobian(self):
        print()
        initial_payoff_matrix, value = self.generate_random_data(strategy_count)
        strategy_weights = random_weights(strategy_count)
        result = self.class_to_test(initial_payoff_matrix, strategy_weights).optimize(check_jacobian = True)
    
    def test_random_unweighted(self):
        for i in range(num_random_trials):
            initial_payoff_matrix, value = self.generate_random_data(strategy_count)
            balance = self.class_to_test(initial_payoff_matrix)
            result = balance.optimize(tol = solver_tol)
            self.assertTrue(result.success)
            numpy.testing.assert_allclose(numpy.average(result.F, axis = 0), value, atol = solution_atol)
            numpy.testing.assert_allclose(numpy.average(result.F, axis = 1), value, atol = solution_atol)
        
    def test_random_weighted(self):
        for i in range(num_random_trials):
            strategy_weights = random_weights(strategy_count)
            initial_payoff_matrix, value = self.generate_random_data(strategy_count)
            balance = self.class_to_test(initial_payoff_matrix, strategy_weights)
            result = balance.optimize(tol = solver_tol)
            self.assertTrue(result.success)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 0), value, atol = solution_atol)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 1), value, atol = solution_atol)
    
    def test_random_weighted_with_zeros(self):
        for i in range(num_random_trials):
            strategy_weights = random_weights_with_zeros(strategy_count)
            initial_payoff_matrix, value = self.generate_random_data(strategy_count)
            balance = self.class_to_test(initial_payoff_matrix, strategy_weights)
            result = balance.optimize(tol = solver_tol)
            self.assertTrue(result.success)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 0), value, atol = solution_atol)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 1), value, atol = solution_atol)