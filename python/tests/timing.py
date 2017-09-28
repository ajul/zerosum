import numpy
import unittest
import warnings
import time
import zerosum.balance
import tests.common

"""
Example output:

2x2 mean time (10 trials): 0.003125 s
4x4 mean time (10 trials): 0.001563 s
8x8 mean time (10 trials): 0.001563 s
16x16 mean time (10 trials): 0.001563 s
32x32 mean time (10 trials): 0.014063 s
64x64 mean time (10 trials): 0.043750 s
128x128 mean time (10 trials): 0.265625 s
256x256 mean time (10 trials): 1.345312 s
512x512 mean time (10 trials): 4.893750 s
1024x1024 mean time (10 trials): 32.756250 s
"""

class TestTiming(tests.common.TestBalanceBase):
    class_to_test = zerosum.balance.MultiplicativeBalance
    
    num_timing_trials = 10
    timing_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    def generate_random_args(self, rows, cols):
        value = numpy.random.rand() + 1.0
        base_matrix = numpy.random.random((rows, cols))
        kwargs = {
            'value' : value,
            'base_matrix' : base_matrix,
        }
        return kwargs, value

    def test_timing(self):
        print()
        for timing_size in self.timing_sizes:
            total_time = 0.0
            for i in range(self.num_timing_trials):
                kwargs, value = self.generate_random_args(timing_size, timing_size)
                row_weights = tests.common.random_weights(timing_size)
                col_weights = tests.common.random_weights(timing_size)
                
                start_time = time.process_time()
                result = self.class_to_test(row_weights = row_weights, col_weights = col_weights, **kwargs).optimize(tol = self.solver_tol)
                end_time = time.process_time()
                
                total_time += (end_time - start_time)
            
            mean_time = total_time / self.num_timing_trials
            print('%dx%d mean time (%d trials): %f s' % (timing_size, timing_size, self.num_timing_trials, mean_time))
            