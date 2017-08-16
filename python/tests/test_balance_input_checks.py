import numpy
import unittest
from zerosum.balance.base import ValueWarning
import zerosum.balance.input_checks

class TestBalanceInputChecks(unittest.TestCase):
    def test_shape_rows(self):
        data = numpy.ones((2, 3))
        row_weights = numpy.ones((3,))
        col_weights = numpy.ones((3,))
        with self.assertRaisesRegex(ValueError, 'shape'):
            zerosum.balance.input_checks.check_shape(data, row_weights, col_weights)
            
    def test_shape_cols(self):
        data = numpy.ones((2, 3))
        row_weights = numpy.ones((2,))
        col_weights = numpy.ones((2,))
        with self.assertRaisesRegex(ValueError, 'shape'):
            zerosum.balance.input_checks.check_shape(data, row_weights, col_weights)

    def test_square(self):
        data = numpy.ones((2, 3))
        with self.assertRaisesRegex(ValueError, 'square'):
            zerosum.balance.input_checks.check_square(data)
            
    def test_non_negative(self):
        data = -numpy.ones((2, 2))
        with self.assertRaisesRegex(ValueError, 'has negative'):
            zerosum.balance.input_checks.check_non_negative(data)

    def test_skew_symmetry(self):
        data = numpy.ones((2, 2))
        with self.assertWarnsRegex(ValueWarning, 'skew-symmetric'):
            zerosum.balance.input_checks.check_skew_symmetry(data)
            
    def test_skew_symmetry_with_neutral_value(self):
        data = numpy.array([[0.0, -1.0],
                            [1.0, 0.0]])
        neutral_value = 1.0
        with self.assertWarnsRegex(ValueWarning, 'skew-symmetric'):
            zerosum.balance.input_checks.check_skew_symmetry(data, neutral_value)
            
    def test_log_skew_symmetry_with_base(self):
        data = 2.0 * numpy.ones((2, 2))
        with self.assertWarnsRegex(ValueWarning, 'log-skew-symmetric'):
            zerosum.balance.input_checks.check_log_skew_symmetry(data) 
            
    def test_log_skew_symmetry_with_base(self):
        data = numpy.array([[1.0, 0.1],
                            [10.0, 1.0]])
        neutral_value = 2.0
        with self.assertWarnsRegex(ValueWarning, 'log-skew-symmetric'):
            zerosum.balance.input_checks.check_log_skew_symmetry(data, neutral_value) 
    
    