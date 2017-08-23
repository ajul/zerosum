import numpy
import unittest
import warnings
import timeit
import zerosum.balance
import tests.common

class TestLanchesterNonSymmetricBalance(tests.common.TestNonSymmetricBalanceBase):
    class_to_test = zerosum.balance.LanchesterNonSymmetricBalance
    # Lanchester has a sharp kink at the origin especially for high exponents.
    # We therefore relax the solution tolerance.
    solution_atol = 1e-4
    
    def generate_random_args(self, rows, cols):
        value = 0.5 * (numpy.random.rand() - 0.5)
        base_matrix = 0.1 + numpy.random.random((rows, cols))
        
        # Randomly set exponent to 1.0, 2.0, or some value in between.
        exponent_case = numpy.random.randint(3)
        if exponent_case == 0:
            exponent = 1.0 + numpy.random.rand()
        elif exponent_case == 1:
            exponent = 1.0
        else:
            exponent = 2.0
        
        kwargs = {
            'base_matrix' : base_matrix,
            'value' : value,
            'exponent' : exponent,
        }
        return kwargs, value

class TestLanchesterSymmetricBalance(tests.common.TestSymmetricBalanceBase):
    class_to_test = zerosum.balance.LanchesterSymmetricBalance
    # Lanchester has a sharp kink at the origin especially for high exponents.
    # We therefore relax the solution tolerance.
    solution_atol = 1e-4
    
    def generate_random_args(self, rows):
        value = 0.0
        base_matrix = 0.1 + numpy.random.random((rows, rows))
        base_matrix_it = 1.0 / base_matrix.transpose()
        base_matrix = base_matrix * base_matrix_it
        
        # Randomly set exponent to 1.0, 2.0, or some value in between.
        exponent_case = numpy.random.randint(3)
        if exponent_case == 0:
            exponent = 1.0 + numpy.random.rand()
        elif exponent_case == 1:
            exponent = 1.0
        else:
            exponent = 2.0
            
        kwargs = {
            'base_matrix' : base_matrix,
            'exponent' : exponent,
        }
        return kwargs, value
    
