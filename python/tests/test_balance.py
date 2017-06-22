import numpy
import unittest
import warnings
import zerosum.balance

strategy_count = 16
tol = 1e-7

def random_weights_with_zeros():
    randoms = numpy.random.random((strategy_count,))
    zeros = numpy.zeros((strategy_count,))
    result = numpy.concatenate((randoms, zeros))
    numpy.random.shuffle(result)
    return result

class TestMultiplicativeBalance(unittest.TestCase):
    def test_negative_warning(self):
        data = -numpy.ones((2, 2))
        with self.assertWarns(zerosum.balance.InitialPayoffMatrixWarning):
            zerosum.balance.MultiplicativeBalance(data)
            
    def test_random_unweighted(self):
        data = numpy.random.random((strategy_count, strategy_count))
        result = zerosum.balance.MultiplicativeBalance(data).optimize(tol = tol)
        numpy.testing.assert_allclose(numpy.average(result.F, axis = 0), 1.0, atol = tol)
        numpy.testing.assert_allclose(numpy.average(result.F, axis = 1), 1.0, atol = tol)
        
    def test_random_weighted(self):
        # TODO: Add regularization.
        row_weights = numpy.random.random((strategy_count,))
        col_weights = numpy.random.random((strategy_count,))
        data = numpy.random.random((strategy_count, strategy_count))
        result = zerosum.balance.MultiplicativeBalance(data, row_weights, col_weights).optimize(tol = tol)
        print(result.nfev, result.message)
        numpy.testing.assert_allclose(numpy.average(result.F, weights = row_weights, axis = 0), 1.0, atol = tol)
        numpy.testing.assert_allclose(numpy.average(result.F, weights = col_weights, axis = 1), 1.0, atol = tol)
        
    def test_random_weighted_with_zeros(self):
        # TODO: Add regularization.
        row_weights = random_weights_with_zeros()
        col_weights = random_weights_with_zeros()
        data = numpy.random.random((strategy_count * 2, strategy_count * 2))
        result = zerosum.balance.MultiplicativeBalance(data, row_weights, col_weights).optimize(tol = tol)
        print(result.nfev, result.message)
        numpy.testing.assert_allclose(numpy.average(result.F, weights = row_weights, axis = 0), 1.0, atol = tol)
        numpy.testing.assert_allclose(numpy.average(result.F, weights = col_weights, axis = 1), 1.0, atol = tol)
        

class TestLogisticSymmetricBalance(unittest.TestCase):
    def test_non_skew_symmetric(self):
        data = numpy.eye(2) + 0.5
        with self.assertWarnsRegex(zerosum.balance.InitialPayoffMatrixWarning, 'skew'):
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
        with self.assertWarnsRegex(zerosum.balance.InitialPayoffMatrixWarning, 'close to'):
            zerosum.balance.LogisticSymmetricBalance(data)
    
    def test_random_unweighted(self):
        data = numpy.random.random((strategy_count, strategy_count))
        data_nt = 1.0 - data
        data = 0.5 * (data + data_nt)
        result = zerosum.balance.LogisticSymmetricBalance(data).optimize(tol = tol)
        numpy.testing.assert_allclose(numpy.average(result.F, axis = 0), 0.5, atol = tol)
        numpy.testing.assert_allclose(numpy.average(result.F, axis = 1), 0.5, atol = tol)
        
    def test_random_weighted(self):
        strategy_weights = numpy.random.random((strategy_count,))
        data = numpy.random.random((strategy_count, strategy_count))
        data_nt = 1.0 - data
        data = 0.5 * (data + data_nt)
        result = zerosum.balance.LogisticSymmetricBalance(data, strategy_weights).optimize(tol = tol)
        numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 0), 0.5, atol = tol)
        numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 1), 0.5, atol = tol)
    
    def test_random_weighted_with_zeros(self):
        strategy_weights = random_weights_with_zeros()
        numpy.random.shuffle(strategy_weights)
        data = numpy.random.random((strategy_count * 2, strategy_count * 2))
        data_nt = 1.0 - data
        data = 0.5 * (data + data_nt)
        result = zerosum.balance.LogisticSymmetricBalance(data, strategy_weights).optimize(tol = tol)
        numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 0), 0.5, atol = tol)
        numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 1), 0.5, atol = tol)
    
if __name__ == '__main__':
    unittest.main()
