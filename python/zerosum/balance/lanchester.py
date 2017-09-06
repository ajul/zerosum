from .base import *
from .input_checks import *

class LanchesterBalance():
    """
    A handicap function based on Lanchester attrition. 
    handicaps represent unit costs, with each side having the same "budget",
    so a higher cost means a smaller force.
    The payoff magnitude is the proportion of the winning side remaining 
    after the losing side has been eliminated, with the sign corresponding to the winning side,
    i.e. positive = row player.
    
    The optimization is done over canonical handicaps h = handicap ** exponent.
    
    Note that for exponents above 1.0 this handicap function has a sharp kink
    whenever both sides are nearly evenly matched.
    """
    rectifier = zerosum.function.ReciprocalLinearRectifier()
    
    def __init__(self, base_matrix, exponent):
        self.base_matrix = base_matrix
        if exponent <= 0.0:
            raise ValueError('Lanchester exponent must be positive.')
        if exponent < 1.0 or exponent > 2.0:
            warnings.warn("Lanchester exponent of %0.2f is not within the conventional interval [1.0, 2.0]." % exponent, ValueWarning)
        self.exponent = exponent
        check_non_negative(self.base_matrix)
        check_shape(self.base_matrix, self.row_weights, self.col_weights)
    
    def handicap_function(self, h_r, h_c):
        relative_strengths = self.base_matrix * h_c[None, :] / h_r[:, None]
        
        row_winner = relative_strengths > 1.0
        col_winner = relative_strengths < 1.0
        F = numpy.zeros_like(relative_strengths)
        F[row_winner] = numpy.power(1.0 - 1.0 / relative_strengths[row_winner], 1.0 / self.exponent)
        F[col_winner] = -numpy.power(1.0 - relative_strengths[col_winner], 1.0 / self.exponent)
        return F

    def row_derivative(self, h_r, h_c):
        relative_strengths = self.base_matrix * h_c[None, :] / h_r[:, None]
        drelative_strengths = -self.base_matrix * h_c[None, :] / numpy.square(h_r)[:, None]
        
        row_winner = relative_strengths > 1.0
        col_winner = relative_strengths < 1.0
        
        # Exponents above 1.0 cause a sharp kink at the origin, so we default to a derivative of 1.0.
        dF = numpy.ones_like(relative_strengths)
        dF[row_winner] = numpy.power(1.0 - 1.0 / relative_strengths[row_winner], 1.0 / self.exponent - 1.0)
        dF[row_winner] /= numpy.square(relative_strengths[row_winner])
        dF[col_winner] = numpy.power(1.0 - relative_strengths[col_winner], 1.0 / self.exponent - 1.0)
        dF *= (drelative_strengths / self.exponent)
        
        return dF
        
    def col_derivative(self, h_r, h_c):
        relative_strengths = self.base_matrix * h_c[None, :] / h_r[:, None]
        drelative_strengths = self.base_matrix / h_r[:, None]
        
        row_winner = relative_strengths > 1.0
        col_winner = relative_strengths < 1.0
        
        # Exponents above 1.0 cause a sharp kink at the origin, so we default to a derivative of 1.0.
        dF = numpy.ones_like(relative_strengths)
        dF[row_winner] = numpy.power(1.0 - 1.0 / relative_strengths[row_winner], 1.0 / self.exponent - 1.0)
        dF[row_winner] /= numpy.square(relative_strengths[row_winner])
        dF[col_winner] = numpy.power(1.0 - relative_strengths[col_winner], 1.0 / self.exponent - 1.0)
        dF *= (drelative_strengths / self.exponent)
        
        return dF
        
    def decanonicalize_handicaps(self, h):
        handicaps = numpy.power(h, 1.0 / self.exponent)
        return handicaps
        
class LanchesterNonSymmetricBalance(LanchesterBalance,NonSymmetricBalance):
    def __init__(self, base_matrix, exponent = 1.0, row_weights = None, col_weights = None, value = 0.0):
        """
        Args:
            base_matrix: Should be strictly positive.
            exponent: The Lanchester exponent, which should be positive. Typical values are in [1.0, 2.0].
            value: Desired value of the game. Should be in the interval (-1, 1).
            row_weights, col_weights: Defines the desired Nash equilibrium in terms of strategy probability weights. 
                If only an integer is specified, a uniform distribution will be used.
        """
        if row_weights is None: row_weights = base_matrix.shape[0]
        if col_weights is None: col_weights = base_matrix.shape[1]
        
        if value <= -1.0 or value >= 1.0:
            raise ValueError("value %0.2f is not in the interval (-1, 1)" % value)
        
        NonSymmetricBalance.__init__(self, row_weights, col_weights, value = value)
        LanchesterBalance.__init__(self, base_matrix, exponent)

class LanchesterSymmetricBalance(LanchesterBalance,SymmetricBalance):
    def __init__(self, base_matrix, exponent = 1.0, strategy_weights = None):
        """
        Args:
            base_matrix: Should be strictly positive.
            exponent: The Lanchester exponent. 
                The optimization is computed using a Lanchester linear law. 
                Other Lanchester exponents are handled simply by 
                raising the resulting handicap values to (1 / exponent).
            strategy_weights: Defines the desired Nash equilibrium in terms of strategy probability weights. 
                If only an integer is specified, a uniform distribution will be used.
        """
        if strategy_weights is None: strategy_weights = base_matrix.shape[0]
        
        check_square(base_matrix)
        check_log_skew_symmetry(base_matrix)

        SymmetricBalance.__init__(self, strategy_weights)
        LanchesterBalance.__init__(self, base_matrix, exponent)
