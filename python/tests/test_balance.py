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

if __name__ == '__main__':
    unittest.main()
