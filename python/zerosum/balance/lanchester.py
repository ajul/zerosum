from .base import *

class LanchesterBalance():
    rectifier = zerosum.function.ReciprocalLinearRectifier()
    
    def handicap_function(self, row_handicaps, col_handicaps):
        relative_strengths = self.base_matrix * col_handicaps[None, :] / row_handicaps[:, None]
        row_winner = relative_strengths > 1.0
        col_winner = relative_strengths < 1.0
        F = numpy.zeros_like(relative_strengths)
        F[row_winner] = 1.0 - 1.0 / relative_strengths[row_winner]
        F[col_winner] = -1.0 + relative_strengths[col_winner]
        return F
        
    def row_derivative(self, row_handicaps, col_handicaps):
        relative_strengths = self.base_matrix * col_handicaps[None, :] / row_handicaps[:, None]
        drelative_strengths = -self.base_matrix * col_handicaps[None, :] / numpy.square(row_handicaps)[:, None]
        row_winner = relative_strengths > 1.0
        dF = numpy.copy(drelative_strengths)
        dF[row_winner] /= numpy.square(relative_strengths[row_winner])
        return dF
        
    def col_derivative(self, row_handicaps, col_handicaps):
        relative_strengths = self.base_matrix * col_handicaps[None, :] / row_handicaps[:, None]
        drelative_strengths = self.base_matrix / row_handicaps[:, None]
        row_winner = relative_strengths > 1.0  
        dF = numpy.copy(drelative_strengths)
        dF[row_winner] /= numpy.square(relative_strengths[row_winner])
        return dF
        
class LanchesterNonSymmetricBalance(LanchesterBalance,NonSymmetricBalance):
    """
    A symmetric case representing Lanchester attrition. handicaps represent unit costs.
    payoff is proportion of remaining force as t -> infinity, with the sign
    favoring the winning side.
    """
    def __init__(self, base_matrix, value = 0.0, row_weights = None, col_weights = None, fix_index = True):
        if row_weights is None: row_weights = base_matrix.shape[0]
        if col_weights is None: col_weights = base_matrix.shape[1]
        
        LanchesterBalance.__init__(self)
        NonSymmetricBalance.__init__(self, row_weights, col_weights, value = value, fix_index = fix_index)

        self.base_matrix = base_matrix

class LanchesterSymmetricBalance(LanchesterBalance,SymmetricBalance):
    """
    A symmetric case representing Lanchester attrition. handicaps represent unit costs.
    payoff is proportion of remaining force as t -> infinity, with the sign
    favoring the winning side.
    """
    def __init__(self, base_matrix, strategy_weights = None, fix_index = True):
        if strategy_weights is None: strategy_weights = base_matrix.shape[0]
        if base_matrix.shape[0] != base_matrix.shape[1]:
            raise ValueError('base_matrix is not square.')
        
        LanchesterBalance.__init__(self)
        SymmetricBalance.__init__(self, strategy_weights, fix_index = fix_index)

        self.base_matrix = base_matrix
        # TODO: check symmetry
