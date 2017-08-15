from .base import *

class LanchesterBalance():
    def handicap_function(self, row_handicaps, col_handicaps):
        row_scales = self.rectifier.evaluate(-row_handicaps)
        col_scales = self.rectifier.evaluate(col_handicaps)
        relative_strengths = row_scales[:, None] * self.initial_payoff_matrix * col_scales[None, :]
        row_winner = relative_strengths > 1.0
        col_winner = relative_strengths < 1.0
        F = numpy.zeros_like(relative_strengths)
        F[row_winner] = 1.0 - 1.0 / relative_strengths[row_winner]
        F[col_winner] = -1.0 + relative_strengths[col_winner]
        return F
        
    def row_derivative(self, row_handicaps, col_handicaps):
        row_scales = self.rectifier.evaluate(-row_handicaps)
        col_scales = self.rectifier.evaluate(col_handicaps)
        relative_strengths = row_scales[:, None] * self.initial_payoff_matrix * col_scales[None, :]
        drelative_strengths = -self.rectifier.derivative(-row_handicaps)[:, None] * self.initial_payoff_matrix * col_scales[None, :]
        row_winner = relative_strengths > 1.0
        dF = numpy.copy(drelative_strengths)
        dF[row_winner] = (drelative_strengths / relative_strengths / relative_strengths)[row_winner]
        return dF
        
    def col_derivative(self, row_handicaps, col_handicaps):
        row_scales = self.rectifier.evaluate(-row_handicaps)
        col_scales = self.rectifier.evaluate(col_handicaps)
        relative_strengths = row_scales[:, None] * self.initial_payoff_matrix * col_scales[None, :]
        drelative_strengths = row_scales[:, None] * self.initial_payoff_matrix * self.rectifier.derivative(col_handicaps)[None, :]
        row_winner = relative_strengths > 1.0  
        dF = numpy.copy(drelative_strengths)
        dF[row_winner] = (drelative_strengths / relative_strengths / relative_strengths)[row_winner]
        return dF
        
class LanchesterNonSymmetricBalance(LanchesterBalance,NonSymmetricBalance):
    """
    A symmetric case representing Lanchester attrition. handicaps represent unit costs.
    payoff is proportion of remaining force as t -> infinity, with the sign
    favoring the winning side.
    """
    def __init__(self, initial_payoff_matrix, value = 0.0, row_weights = None, col_weights = None, fix_index = True, 
        rectifier = zerosum.function.ReciprocalLinearRectifier()):
        if row_weights is None: row_weights = initial_payoff_matrix.shape[0]
        if col_weights is None: col_weights = initial_payoff_matrix.shape[1]
        
        LanchesterBalance.__init__(self)
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

class LanchesterSymmetricBalance(LanchesterBalance,SymmetricBalance):
    """
    A symmetric case representing Lanchester attrition. handicaps represent unit costs.
    payoff is proportion of remaining force as t -> infinity, with the sign
    favoring the winning side.
    """
    def __init__(self, initial_payoff_matrix, strategy_weights = None, fix_index = True,
        rectifier = zerosum.function.ReciprocalLinearRectifier()):
        if strategy_weights is None: strategy_weights = initial_payoff_matrix.shape[0]
        if initial_payoff_matrix.shape[0] != initial_payoff_matrix.shape[1]:
            raise ValueError('initial_payoff_matrix is not square.')
        
        LanchesterBalance.__init__(self)
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