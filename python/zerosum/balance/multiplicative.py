from .base import *
from .input_checks import *

class MultiplicativeBalance(NonSymmetricBalance):
    """
    Handicap function defined by col_handicap / row_handicap * initial_payoff.
    """
    
    rectify_mask = True
    
    def __init__(self, base_matrix, row_weights = None, col_weights = None, value = 1.0):
        """
        Args:
            base_matrix: Should be nonnegative and preferably strictly positive.
            value: Should be strictly positive. Note that the default is 1.0.
                
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
            value = value)
            
        check_shape(self.base_matrix, self.row_weights, self.col_weights)

    def handicap_function(self, h_r, h_c):
        return self.base_matrix * h_c[None, :] / h_r[:, None]
        
    def row_derivative(self, h_r, h_c):
        return -self.base_matrix * h_c[None, :] / numpy.square(h_r)[:, None]
        
    def col_derivative(self, h_r, h_c):
        return self.base_matrix / h_r[:, None]
