from .base import *

class HazardBalance():
    rectifier = zerosum.function.ReciprocalLinearRectifier()
    
    def handicap_function(self, row_handicaps, col_handicaps):
        relative_strengths = self.initial_payoff_matrix
        row_winner = relative_strengths > 1.0
        F = numpy.zeros_like(relative_strengths)
        Fr = col_handicaps[None, :] - row_handicaps[:, None] / relative_strengths
        Fc = col_handicaps[None, :] * relative_strengths - row_handicaps[:, None]
        F[row_winner] = Fr[row_winner]
        F[~row_winner] = Fc[~row_winner] 
        return F
        
    def row_derivative(self, row_handicaps, col_handicaps):
        relative_strengths = self.initial_payoff_matrix
        row_winner = relative_strengths > 1.0
        dF = numpy.zeros_like(relative_strengths)
        dFr = -1.0 / relative_strengths
        dFc = -numpy.ones_like(relative_strengths)
        dF[row_winner] = dFr[row_winner]
        dF[~row_winner] = dFc[~row_winner]
        return dF
    
    def col_derivative(self, row_handicaps, col_handicaps):
        relative_strengths = self.initial_payoff_matrix
        row_winner = relative_strengths > 1.0
        dF = numpy.zeros_like(relative_strengths)
        dFr = numpy.ones_like(relative_strengths)
        dFc = relative_strengths
        dF[row_winner] = dFr[row_winner]
        dF[~row_winner] = dFc[~row_winner]
        return dF

class HazardNonSymmetricBalance(HazardBalance,NonSymmetricBalance):
    def __init__(self, initial_payoff_matrix, value = 0.0, row_weights = None, col_weights = None, fix_index = None):
        if row_weights is None: row_weights = initial_payoff_matrix.shape[0]
        if col_weights is None: col_weights = initial_payoff_matrix.shape[1]
        
        # We fix an index only if the desired value is zero.
        # A global scale might be required to achieve other values.
        if fix_index is None:
            fix_index = (value == 0.0)   
        
        HazardBalance.__init__(self)
        NonSymmetricBalance.__init__(self, row_weights, col_weights, value = value, fix_index = fix_index)

        self.initial_payoff_matrix = initial_payoff_matrix
        
class HazardSymmetricBalance(HazardBalance,SymmetricBalance):
    """
    A symmetric case appearing in:
    Hazard, C. J. 2010. What every game designer should know about game theory. Triangle Game Conference. Raleigh, North Carolina.
    
    Similar but not the same as the Lanchester.
    """
    def __init__(self, initial_payoff_matrix, strategy_weights = None, fix_index = True):
        if strategy_weights is None: strategy_weights = initial_payoff_matrix.shape[0]
        if initial_payoff_matrix.shape[0] != initial_payoff_matrix.shape[1]:
            raise ValueError('initial_payoff_matrix is not square.')
        
        HazardBalance.__init__(self)
        SymmetricBalance.__init__(self, strategy_weights, fix_index = fix_index)

        self.initial_payoff_matrix = initial_payoff_matrix
        # TODO: check symmetry
    