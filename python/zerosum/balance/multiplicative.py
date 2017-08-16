from .base import *
from .input_checks import *

class MultiplicativeBalance(NonSymmetricBalance):
    """
    Handicap function defined by col_handicap / row_handicap * initial_payoff.
    """
    
    rectifier = zerosum.function.ReciprocalLinearRectifier()
    
    def __init__(self, base_matrix, row_weights = None, col_weights = None, 
        value = 1.0, fix_index = True):
        """
        Args:
            base_matrix: Should be nonnegative and preferably strictly positive.
            value: Should be strictly positive. Note that the default is 1.0.
            fix_index: Since this handicap function is invariant with respect to a global scale, we default to True.
                
        Raises:
            ValueWarning: If base_matrix has negative elements.
        """
        self.base_matrix = base_matrix
        if row_weights is None: row_weights = base_matrix.shape[0]
        if col_weights is None: col_weights = base_matrix.shape[1]
        
        check_non_negative(base_matrix)
            
        if value <= 0.0:
            warnings.warn('Value %f is non-positive.' % value, ValueWarning)
    
        NonSymmetricBalance.__init__(self, row_weights = row_weights, col_weights = col_weights, 
            value = value, fix_index = fix_index)
            
        check_shape(self.base_matrix, self.row_weights, self.col_weights)

    def handicap_function(self, row_handicaps, col_handicaps):
        return self.base_matrix * col_handicaps[None, :] / row_handicaps[:, None]
        
    def row_derivative(self, row_handicaps, col_handicaps):
        return -self.base_matrix * col_handicaps[None, :] / numpy.square(row_handicaps)[:, None]
        
    def col_derivative(self, row_handicaps, col_handicaps):
        return self.base_matrix / row_handicaps[:, None]
