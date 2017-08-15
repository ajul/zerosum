from .base import *

# TODO: non-symmetric

class HazardBalance():
    def handicap_function(self, row_handicaps, col_handicaps):
        row_costs = self.rectifier.evaluate(row_handicaps)
        col_costs = self.rectifier.evaluate(col_handicaps)
        relative_strengths = self.initial_payoff_matrix
        row_winner = relative_strengths > 1.0
        F = numpy.zeros_like(relative_strengths)
        Fr = col_costs[None, :] - row_costs[:, None] / relative_strengths
        Fc = col_costs[None, :] * relative_strengths - row_costs[:, None]
        F[row_winner] = Fr[row_winner]
        F[~row_winner] = Fc[~row_winner] 
        return F
        
    def row_derivative(self, row_handicaps, col_handicaps):
        row_costs = self.rectifier.evaluate(row_handicaps)
        relative_strengths = self.initial_payoff_matrix
        row_winner = relative_strengths > 1.0
        dF = numpy.zeros_like(relative_strengths)
        dFr = -self.rectifier.derivative(row_handicaps)[:, None] / relative_strengths
        dFc = -self.rectifier.derivative(row_handicaps)[:, None] * numpy.ones_like(relative_strengths)
        dF[row_winner] = dFr[row_winner]
        dF[~row_winner] = dFc[~row_winner]
        return dF
    
    def col_derivative(self, row_handicaps, col_handicaps):
        col_costs = self.rectifier.evaluate(col_handicaps)
        relative_strengths = self.initial_payoff_matrix
        row_winner = relative_strengths > 1.0
        dF = numpy.zeros_like(relative_strengths)
        dFr = self.rectifier.derivative(col_handicaps)[None, :] * numpy.ones_like(relative_strengths)
        dFc = self.rectifier.derivative(col_handicaps)[None, :] * relative_strengths
        dF[row_winner] = dFr[row_winner]
        dF[~row_winner] = dFc[~row_winner]
        return dF

class HazardNonSymmetricBalance(HazardBalance,NonSymmetricBalance):
    def __init__(self, initial_payoff_matrix, value = 0.0, row_weights = None, col_weights = None, fix_index = True,
        rectifier = zerosum.function.ReciprocalLinearRectifier()):
        if row_weights is None: row_weights = initial_payoff_matrix.shape[0]
        if col_weights is None: col_weights = initial_payoff_matrix.shape[1]
        
        HazardBalance.__init__(self)
        NonSymmetricBalance.__init__(self, row_weights, col_weights, value = value, fix_index = fix_index)

        self.initial_payoff_matrix = initial_payoff_matrix
        self.rectifier = rectifier
    
    def optimize(self, *args, **kwargs):
        """
        Compute the handicaps that balance the game using scipy.optimize.root.
        
        Args:
            As NonSymmetricBalance.optimize().
        
        Returns:
            As NonSymmetricBalance.optimize().
        """
        result = NonSymmetricBalance.optimize(self, *args, **kwargs)
        result.row_handicaps_pre_rectify = result.row_handicaps
        result.col_handicaps_pre_rectify = result.col_handicaps
        result.row_handicaps = self.rectifier.evaluate(result.row_handicaps)
        result.col_handicaps = self.rectifier.evaluate(result.col_handicaps)
        return result
        
class HazardSymmetricBalance(HazardBalance,SymmetricBalance):
    """
    A symmetric case appearing in:
    Hazard, C. J. 2010. What every game designer should know about game theory. Triangle Game Conference. Raleigh, North Carolina.
    
    Similar but not the same as the Lanchester.
    """
    def __init__(self, initial_payoff_matrix, strategy_weights = None, fix_index = True,
        rectifier = zerosum.function.ReciprocalLinearRectifier()):
        if strategy_weights is None: strategy_weights = initial_payoff_matrix.shape[0]
        if initial_payoff_matrix.shape[0] != initial_payoff_matrix.shape[1]:
            raise ValueError('initial_payoff_matrix is not square.')
        
        HazardBalance.__init__(self)
        SymmetricBalance.__init__(self, strategy_weights, fix_index = fix_index)

        self.initial_payoff_matrix = initial_payoff_matrix
        self.rectifier = rectifier
        # TODO: check symmetry
    
    def optimize(self, *args, **kwargs):
        """
        Compute the handicaps that balance the game using scipy.optimize.root.
        
        Args:
            As SymmetricBalance.optimize().
        
        Returns:
            As SymmetricBalance.optimize().
        """
        result = SymmetricBalance.optimize(self, *args, **kwargs)
        result.handicaps_pre_rectify = result.handicaps
        result.handicaps = self.rectifier.evaluate(result.handicaps)
        return result
    