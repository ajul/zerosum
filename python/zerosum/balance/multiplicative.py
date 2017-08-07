from .base import *

class MultiplicativeBalance(NonSymmetricBalance):
    """
    A special case where the handicap functions are col_handicap / row_handicap * initial_payoff.
    The actual optimization is done by mapping raw handicaps in (-inf, inf) to the actual handicaps (0, inf) using a rectifier.
    """
    
    def __init__(self, initial_payoff_matrix, row_weights = None, col_weights = None, 
        value = 1.0, fix_index = True, rectifier = zerosum.function.ReciprocalLinearRectifier()):
        """
        Args:
            initial_payoff_matrix: Should be nonnegative and preferably strictly positive.
            value: Should be strictly positive. Note that the default is 1.0.
            fix_index: Since this handicap function is invariant with respect to a global scale, we default to True.
            rectifier: A strictly monotonically increasing function with range (0, inf).
                While an exponential is appealing from an analytic point of view, it can cause overflows in practice.
                We therefore default to a reciprocal-linear rectifier.
                
        Raises:
            ValueError: If the size of row_weights or col_weights do not match initial_payoff_matrix.
            ValueWarning: If initial_payoff_matrix has negative elements.
        """
        self.initial_payoff_matrix = initial_payoff_matrix
        if row_weights is None: row_weights = initial_payoff_matrix.shape[0]
        if col_weights is None: col_weights = initial_payoff_matrix.shape[1]
        
        if numpy.any(initial_payoff_matrix < 0.0):
            warnings.warn('initial_payoff_matrix has negative element(s).', ValueWarning)
            
        if value <= 0.0:
            warnings.warn('Value %f is non-positive.' % value, ValueWarning)
    
        NonSymmetricBalance.__init__(self, self.handicap_function, row_weights = row_weights, col_weights = col_weights, 
            row_derivative = self.row_derivative, col_derivative = self.col_derivative, 
            value = value, fix_index = fix_index)
            
        if initial_payoff_matrix.shape[0] != self.row_weights.size:
            raise ValueError('The size of row_weights does not match the row count of initial_payoff_matrix.')
        
        if initial_payoff_matrix.shape[1] != self.col_weights.size:
            raise ValueError('The size of col_weights does not match the column count of initial_payoff_matrix.')
            
        self.rectifier = rectifier

    def handicap_function(self, row_handicaps, col_handicaps):
        return self.initial_payoff_matrix * self.rectifier.evaluate(col_handicaps)[None, :] * self.rectifier.evaluate(-row_handicaps)[:, None]
        
    def row_derivative(self, row_handicaps, col_handicaps):
        return self.initial_payoff_matrix * self.rectifier.evaluate(col_handicaps)[None, :] * -self.rectifier.derivative(-row_handicaps)[:, None]
        
    def col_derivative(self, row_handicaps, col_handicaps):
        return self.initial_payoff_matrix * self.rectifier.derivative(col_handicaps)[None, :] * self.rectifier.evaluate(-row_handicaps)[:, None]
    
    def optimize(self, method = 'lm', *args, **kwargs):
        """
        Compute the handicaps that balance the game using scipy.optimize.root.
        
        Args:
            method: Used by scipy.optimize.root. 
                For this case we default to method 'lm' since it seems to produce more accurate results.
        
        Returns:
            The result of scipy.optimize.root, as with NonSymmetricBalance.
            Note that the actual optimization is done using handicaps in (-inf, inf)
            that are rectified before being used.
            These can be accessed using result.row_handicaps_pre_rectify, result.col_handicaps_pre_rectify.
        """
        
        result = NonSymmetricBalance.optimize(self, method = method, *args, **kwargs)
        result.row_handicaps_pre_rectify = result.row_handicaps
        result.col_handicaps_pre_rectify = result.col_handicaps
        result.row_handicaps = self.rectifier.evaluate(result.row_handicaps)
        result.col_handicaps = self.rectifier.evaluate(result.col_handicaps)
        return result