from .base import *
from .input_checks import *

class HazardBalance():
    """
    Handicap function taken from 
    Hazard, C. J. 2010. What every game designer should know about game theory. Triangle Game Conference. Raleigh, North Carolina.
    
    This is similar to the Lanchester (linear) law, with two sides fighting until one is eliminated.
    However, the effect of the handicap (cost) is different:
    instead of each side being able to afford a quantity of units inversely proportional to cost,
    each side gets one unit, and payoff is cost of damage dealt minus cost of damage received.
    The handicaps thus do not affect which side wins, or by how much, but rather the cost of doing so.
    
    The original example was symmetric, but it can work in the non-symmetric case as well.
    """
    rectifier = zerosum.function.ReciprocalLinearRectifier()
    
    def __init__(self, base_matrix):
        self.base_matrix = base_matrix
        check_non_negative(self.base_matrix)
        check_shape(self.base_matrix, self.row_weights, self.col_weights)
    
    def handicap_function(self, row_handicaps, col_handicaps):
        relative_strengths = self.base_matrix
        row_winner = relative_strengths > 1.0
        payoff_matrix = numpy.zeros_like(relative_strengths)
        payoff_matrix_r = col_handicaps[None, :] - row_handicaps[:, None] / relative_strengths
        payoff_matrix_c = col_handicaps[None, :] * relative_strengths - row_handicaps[:, None]
        payoff_matrix[row_winner] = payoff_matrix_r[row_winner]
        payoff_matrix[~row_winner] = payoff_matrix_c[~row_winner] 
        return payoff_matrix
        
    def row_derivative(self, row_handicaps, col_handicaps):
        relative_strengths = self.base_matrix
        row_winner = relative_strengths > 1.0
        d_payoff_matrix = numpy.zeros_like(relative_strengths)
        d_payoff_matrix_r = -1.0 / relative_strengths
        d_payoff_matrix_c = -numpy.ones_like(relative_strengths)
        d_payoff_matrix[row_winner] = d_payoff_matrix_r[row_winner]
        d_payoff_matrix[~row_winner] = d_payoff_matrix_c[~row_winner]
        return d_payoff_matrix
    
    def col_derivative(self, row_handicaps, col_handicaps):
        relative_strengths = self.base_matrix
        row_winner = relative_strengths > 1.0
        d_payoff_matrix = numpy.zeros_like(relative_strengths)
        d_payoff_matrix_r = numpy.ones_like(relative_strengths)
        d_payoff_matrix_c = relative_strengths
        d_payoff_matrix[row_winner] = d_payoff_matrix_r[row_winner]
        d_payoff_matrix[~row_winner] = d_payoff_matrix_c[~row_winner]
        return d_payoff_matrix

class HazardNonSymmetricBalance(HazardBalance,NonSymmetricBalance):
    def __init__(self, base_matrix, value = 0.0, row_weights = None, col_weights = None, fix_index = None):
        if row_weights is None: row_weights = base_matrix.shape[0]
        if col_weights is None: col_weights = base_matrix.shape[1]
        
        # We fix an index only if the desired value is zero.
        # A global scale might be required to achieve other values.
        if fix_index is None:
            fix_index = (value == 0.0)
        
        NonSymmetricBalance.__init__(self, row_weights, col_weights, value = value, fix_index = fix_index)
        HazardBalance.__init__(self, base_matrix)
        
class HazardSymmetricBalance(HazardBalance,SymmetricBalance):
    def __init__(self, base_matrix, strategy_weights = None, fix_index = True):
        if strategy_weights is None: strategy_weights = base_matrix.shape[0]
        
        check_square(base_matrix)
        check_log_skew_symmetry(base_matrix)
        
        SymmetricBalance.__init__(self, strategy_weights, fix_index = fix_index)
        HazardBalance.__init__(self, base_matrix)
