import numpy
import unittest
import warnings
import zerosum.balance

class TestMultiplicativeBalance(unittest.TestCase):
    def test_negative_warning(self):
        data = -numpy.ones((2, 2))
        with self.assertWarns(zerosum.balance.InitialPayoffMatrixWarning):
            zerosum.balance.MultiplicativeBalance(data)

class TestLogisticSymmetricBalance(unittest.TestCase):
    strategy_count = 16

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
        data = numpy.random.random((self.strategy_count, self.strategy_count))
        data_nt = 1.0 - data
        data = 0.5 * (data + data_nt)
        result = zerosum.balance.LogisticSymmetricBalance(data).optimize()
        numpy.testing.assert_allclose(numpy.average(result.F, axis = 0), 0.5)
        numpy.testing.assert_allclose(numpy.average(result.F, axis = 1), 0.5)
        
    def test_random_weighted(self):
        strategy_weights = numpy.random.random((self.strategy_count,))
        data = numpy.random.random((self.strategy_count, self.strategy_count))
        data_nt = 1.0 - data
        data = 0.5 * (data + data_nt)
        result = zerosum.balance.LogisticSymmetricBalance(data, strategy_weights).optimize()
        numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 0), 0.5)
        numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 1), 0.5)
    
    def test_random_weighted_with_zeros(self):
        strategy_weights = numpy.concatenate((numpy.random.random((self.strategy_count,)),
                                              numpy.zeros((self.strategy_count,))))
        numpy.random.shuffle(strategy_weights)
        data = numpy.random.random((self.strategy_count * 2, self.strategy_count * 2))
        data_nt = 1.0 - data
        data = 0.5 * (data + data_nt)
        result = zerosum.balance.LogisticSymmetricBalance(data, strategy_weights).optimize()
        numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 0), 0.5)
        numpy.testing.assert_allclose(numpy.average(result.F, weights = strategy_weights, axis = 1), 0.5)
    
if __name__ == '__main__':
    unittest.main()
