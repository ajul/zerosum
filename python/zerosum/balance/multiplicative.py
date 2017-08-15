from .base import *

class MultiplicativeBalance(NonSymmetricBalance):
    """
    A special case where the handicap functions are col_handicap / row_handicap * initial_payoff.
    The actual optimization is done by mapping raw handicaps in (-inf, inf) to the actual handicaps (0, inf) using a rectifier.
    """
    
    rectifier = zerosum.function.ReciprocalLinearRectifier()
    
    def __init__(self, initial_payoff_matrix, row_weights = None, col_weights = None, 
        value = 1.0, fix_index = True):
        """
        Args:
            initial_payoff_matrix: Should be nonnegative and preferably strictly positive.
            value: Should be strictly positive. Note that the default is 1.0.
            fix_index: Since this handicap function is invariant with respect to a global scale, we default to True.
                
        Raises:
            ValueWarning: If initial_payoff_matrix has negative elements.
        """
        self.initial_payoff_matrix = initial_payoff_matrix
        if row_weights is None: row_weights = initial_payoff_matrix.shape[0]
        if col_weights is None: col_weights = initial_payoff_matrix.shape[1]
        
        if numpy.any(initial_payoff_matrix < 0.0):
            warnings.warn('initial_payoff_matrix has negative element(s).', ValueWarning)
            
        if value <= 0.0:
            warnings.warn('Value %f is non-positive.' % value, ValueWarning)
    
        NonSymmetricBalance.__init__(self, row_weights = row_weights, col_weights = col_weights, 
            value = value, fix_index = fix_index)

    def handicap_function(self, row_handicaps, col_handicaps):
        return self.initial_payoff_matrix * col_handicaps[None, :] / row_handicaps[:, None]
        
    def row_derivative(self, row_handicaps, col_handicaps):
        return -self.initial_payoff_matrix * col_handicaps[None, :] / numpy.square(row_handicaps)[:, None]
        
    def col_derivative(self, row_handicaps, col_handicaps):
        return self.initial_payoff_matrix / row_handicaps[:, None]
