from .base import *

# TODO: non-symmetric

class LanchesterSymmetricBalance(SymmetricBalance):
    """
    A symmetric case representing Lanchester attrition. handicaps represent unit costs.
    payoff is proportion of remaining force as t -> infinity, with the sign
    favoring the winning side.
    """
    def __init__(self, initial_payoff_matrix, strategy_weights = None, fix_index = True):
        if strategy_weights is None: strategy_weights = initial_payoff_matrix.shape[0]
        if initial_payoff_matrix.shape[0] != initial_payoff_matrix.shape[1]:
            raise ValueError('initial_payoff_matrix is not square.')
        
        SymmetricBalance.__init__(self, strategy_weights, fix_index = fix_index)

        self.initial_payoff_matrix = initial_payoff_matrix
        # TODO: check symmetry
    
    def handicap_function(self, row_handicaps, col_handicaps):
        row_scales = numpy.exp(-row_handicaps)
        col_scales = numpy.exp(col_handicaps)
        relative_strengths = row_scales[:, None] * self.initial_payoff_matrix * col_scales[None, :]
        row_winner = relative_strengths > 1.0
        col_winner = relative_strengths < 1.0
        F = numpy.zeros_like(relative_strengths)
        F[row_winner] = 1.0 - 1.0 / relative_strengths[row_winner]
        F[col_winner] = -1.0 + relative_strengths[col_winner]
        return F
        
    def row_derivative(self, row_handicaps, col_handicaps):
        row_scales = numpy.exp(-row_handicaps)
        col_scales = numpy.exp(col_handicaps)
        relative_strengths = row_scales[:, None] * self.initial_payoff_matrix * col_scales[None, :]
        row_winner = relative_strengths > 1.0 
        dF = -numpy.copy(relative_strengths)
        dF[row_winner] = -1.0 / relative_strengths[row_winner]
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