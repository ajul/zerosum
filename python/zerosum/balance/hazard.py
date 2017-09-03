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
    The h thus do not affect which side wins, or by how much, but rather the cost of doing so.
    
    The original example was symmetric, but it can work in the non-symmetric case as well.
    """
    rectifier = zerosum.function.ReciprocalLinearRectifier()
    
    def __init__(self, base_matrix):
        self.base_matrix = base_matrix
        check_non_negative(self.base_matrix)
        check_shape(self.base_matrix, self.row_weights, self.col_weights)
    
    def handicap_function(self, h_r, h_c):
        relative_strengths = self.base_matrix
        row_winner = relative_strengths > 1.0
        F = numpy.zeros_like(relative_strengths)
        Fr = h_c[None, :] - h_r[:, None] / relative_strengths
        Fc = h_c[None, :] * relative_strengths - h_r[:, None]
        F[row_winner] = Fr[row_winner]
        F[~row_winner] = Fc[~row_winner] 
        return F
        
    def row_derivative(self, h_r, h_c):
        relative_strengths = self.base_matrix
        row_winner = relative_strengths > 1.0
        dF = numpy.zeros_like(relative_strengths)
        dFr = -1.0 / relative_strengths
        dFc = -numpy.ones_like(relative_strengths)
        dF[row_winner] = dFr[row_winner]
        dF[~row_winner] = dFc[~row_winner]
        return dF
    
    def col_derivative(self, h_r, h_c):
        relative_strengths = self.base_matrix
        row_winner = relative_strengths > 1.0
        dF = numpy.zeros_like(relative_strengths)
        dFr = numpy.ones_like(relative_strengths)
        dFc = relative_strengths
        dF[row_winner] = dFr[row_winner]
        dF[~row_winner] = dFc[~row_winner]
        return dF

class HazardNonSymmetricBalance(HazardBalance,NonSymmetricBalance):
    """
    This case is unusual in that the solution may non-trivally change depending on the regularization.
    """
    def __init__(self, base_matrix, row_weights = None, col_weights = None, value = 0.0):
        if row_weights is None: row_weights = base_matrix.shape[0]
        if col_weights is None: col_weights = base_matrix.shape[1]
        
        NonSymmetricBalance.__init__(self, row_weights, col_weights, value = value)
        HazardBalance.__init__(self, base_matrix)
        
class HazardSymmetricBalance(HazardBalance,SymmetricBalance):
    def __init__(self, base_matrix, strategy_weights = None):
        if strategy_weights is None: strategy_weights = base_matrix.shape[0]
        
        check_square(base_matrix)
        check_log_skew_symmetry(base_matrix)
        
        SymmetricBalance.__init__(self, strategy_weights)
        HazardBalance.__init__(self, base_matrix)
