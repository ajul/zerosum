import numpy
import unittest
import warnings
import zerosum.balance

strategy_count = 4
tol = 1e-7

def random_weights(n):
    result = numpy.random.random((n,))
    result /= numpy.sum(result)
    return result

def random_weights_with_zeros(n):
    result = numpy.random.random((n,))
    result[:n//2] = 0.0
    result /= numpy.sum(result)
    numpy.random.shuffle(result)
    return result

class TestMultiplicativeBalance(unittest.TestCase):
    value = 1.0
    def test_negative_matrix_warning(self):
        data = -numpy.ones((2, 2))
        with self.assertWarnsRegex(zerosum.balance.ValueWarning, 'initial_payoff_matrix.*negative'):
            zerosum.balance.MultiplicativeBalance(data)
    
    def test_weights_warning(self):
        data = numpy.random.random((strategy_count, strategy_count))
        row_weights = random_weights_with_zeros(strategy_count)
        col_weights = random_weights_with_zeros(strategy_count) * 2.0
        with self.assertWarnsRegex(zerosum.balance.ValueWarning, 'sum to the same value'):
            zerosum.balance.MultiplicativeBalance(data, row_weights, col_weights, value = self.value)
        
    def test_random_unweighted(self):
        data = numpy.random.random((strategy_count, strategy_count))
        result = zerosum.balance.MultiplicativeBalance(data, value = self.value).optimize(tol = tol)
        numpy.testing.assert_allclose(numpy.average(result.F, axis = 0), self.value, atol = tol)
        numpy.testing.assert_allclose(numpy.average(result.F, axis = 1), self.value, atol = tol)
        
    def test_random_weighted(self):
        # TODO: Add regularization.
        row_weights = random_weights(strategy_count)
        col_weights = random_weights(strategy_count + 1)
        data = numpy.random.random((strategy_count, strategy_count + 1))
        result = zerosum.balance.MultiplicativeBalance(data, row_weights, col_weights, value = self.value).optimize(tol = tol)
        print(result.nfev, result.message, result.x)
        numpy.testing.assert_allclose(numpy.average(result.F, weights = row_weights, axis = 0), self.value, atol = tol)
        numpy.testing.assert_allclose(numpy.average(result.F, weights = col_weights, axis = 1), self.value, atol = tol)
        
    def test_random_weighted_with_zeros(self):
        for i in range(1):
            # TODO: Add regularization.
            row_weights = random_weights_with_zeros(strategy_count)
            col_weights = random_weights_with_zeros(strategy_count + 1)
            data = numpy.random.random((strategy_count, strategy_count +1))
            result = zerosum.balance.MultiplicativeBalance(data, row_weights, col_weights, value = self.value).optimize(check_jacobian_epsilon = None, tol = tol)
            
            #print(result.nfev, result.message, result.x)
            if not result.success: 
                print(row_weights, col_weights)
                print(data)
                print(result)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = row_weights, axis = 0), self.value, atol = tol)
            numpy.testing.assert_allclose(numpy.average(result.F, weights = col_weights, axis = 1), self.value, atol = tol)
        

class TestLogisticSymmetricBalance(unittest.TestCase):
    def test_non_skew_symmetric(self):
        data = numpy.eye(2) + 0.5
        with self.assertWarnsRegex(zerosum.balance.ValueWarning, 'skew'):
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
        data = numpy.random.random((strategy_count, strategy_count))
        data_nt = 1.0 - data
        data = 0.5 * (data + data_nt)
        result = zerosum.balance.LogisticSymmetricBalance(data).optimize(tol = tol)
        numpy.testing.assert_allclose(numpy.average(result.F, axis = 0), 0.5, atol = tol)
        numpy.testing.assert_allclose(numpy.average(result.F, axis = 1), 0.5, atol = tol)
        
    def test_random_weighted(self):
        strategy_weights = random_weights(strategy_count)
        data = numpy.random.random((strategy_count, strategy_count))
        data_nt = 1.0 - data
        data = 0.5 * (data + data_nt)
        result = zerosum.balance.LogisticSymmetricBalance(data, strategy_weights).optimize(tol = tol)
        numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 0), 0.5, atol = tol)
        numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 1), 0.5, atol = tol)
    
    def test_random_weighted_with_zeros(self):
        strategy_weights = random_weights_with_zeros(strategy_count)
        data = numpy.random.random((strategy_count * 2, strategy_count * 2))
        data_nt = 1.0 - data
        data = 0.5 * (data + data_nt)
        result = zerosum.balance.LogisticSymmetricBalance(data, strategy_weights).optimize(tol = tol)
        numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 0), 0.5, atol = tol)
        numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 1), 0.5, atol = tol)
    
if __name__ == '__main__':
    unittest.main()
