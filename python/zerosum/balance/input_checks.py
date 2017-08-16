from .base import ValueWarning
import numpy
import warnings

def check_shape(base_matrix, row_weights, col_weights = None):
    if col_weights is None:
        col_weights = row_weights
    
    if row_weights.size != base_matrix.shape[0]:
        raise ValueError('Mismatch in shape between row_weights (%d rows) and base_matrix (%d rows).' % (row_weights.size, base_matrix.shape[0]))
    
    if col_weights.size != base_matrix.shape[1]:
        raise ValueError('Mismatch in shape between col_weights (%d cols) and base_matrix (%d cols).' % (col_weights.size, base_matrix.shape[1]))

def check_square(base_matrix):
    if base_matrix.shape[0] != base_matrix.shape[1]:
        raise ValueError('base_matrix is not square.')
        
def check_non_negative(base_matrix):
    if numpy.any(base_matrix < 0.0):
        raise ValueError('base_matrix has negative value(s).')
        
def check_skew_symmetry(base_matrix, neutral_value = 0.0):
    base_matrix_compliment_transpose = 2.0 * neutral_value - base_matrix.transpose()
    if not numpy.allclose(base_matrix, base_matrix_compliment_transpose):
        warnings.warn('base_matrix is not (close to) skew-symmetric relative to neutral value of %f.' % neutral_value, ValueWarning)
    
def check_log_skew_symmetry(base_matrix, neutral_value = 1.0):
    base_matrix_reciprocal_transpose = neutral_value / base_matrix.transpose()
    if not numpy.allclose(base_matrix, base_matrix_reciprocal_transpose):
        warnings.warn('base_matrix is not (close to) log-skew-symmetric relative to neutral value of %f.' % neutral_value, ValueWarning)