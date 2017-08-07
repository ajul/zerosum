from .base import *

class HazardSymmetricBalance(SymmetricBalance):
    """
    A symmetric case appearing in:
    Hazard, C. J. 2010. What every game designer should know about game theory. Triangle Game Conference. Raleigh, North Carolina.
    
    Similar but not the same as the Lanchester.
    """
    def __init__(self, initial_payoff_matrix, strategy_weights = None, fix_index = True):
        if strategy_weights is None: strategy_weights = initial_payoff_matrix.shape[0]
        if initial_payoff_matrix.shape[0] != initial_payoff_matrix.shape[1]:
            raise ValueError('initial_payoff_matrix is not square.')
        
        SymmetricBalance.__init__(self, strategy_weights, fix_index = fix_index)

        self.initial_payoff_matrix = initial_payoff_matrix
        # TODO: check symmetry
    
    def handicap_function(self, row_handicaps, col_handicaps):
        row_costs = numpy.exp(row_handicaps)
        col_costs = numpy.exp(col_handicaps)
        relative_strengths = self.initial_payoff_matrix
        row_winner = relative_strengths > 1.0
        F = numpy.zeros_like(relative_strengths)
        Fr = col_costs[None, :] - row_costs[:, None] / relative_strengths
        Fc = col_costs[None, :] * relative_strengths - row_costs[:, None]
        F[row_winner] = Fr[row_winner]
        F[~row_winner] = Fc[~row_winner] 
        return F
        
    def row_derivative(self, row_handicaps, col_handicaps):
        row_costs = numpy.exp(row_handicaps)
        relative_strengths = self.initial_payoff_matrix
        row_winner = relative_strengths > 1.0
        dF = numpy.zeros_like(relative_strengths)
        dFr = -row_costs[:, None] / relative_strengths[row_winner]
        dF[row_winner] = dFr[row_winner]
        dF[~row_winner] = -1.0
        return dF
    
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
        result.handicaps = numpy.exp(result.handicaps)
        return result
    