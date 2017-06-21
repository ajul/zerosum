import numpy
import unittest
import warnings
import zerosum.balance

class TestMultiplicativeBalance(unittest.TestCase):
    def test_negative_warning(self):
        data = -numpy.ones((2, 2))
        with self.assertWarns(zerosum.balance.InitialPayoffMatrixWarning):
            zerosum.balance.MultiplicativeBalance(data)
            

if __name__ == '__main__':
    unittest.main()
