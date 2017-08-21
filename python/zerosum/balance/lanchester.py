from .base import *
from .input_checks import *

class LanchesterBalance():
    """
    A handicap function based on Lanchester attrition. 
    Handicaps represent unit costs, with each side having the same "budget",
    so a higher cost means a smaller force.
    The payoff magnitude is the proportion of the winning side remaining 
    after the losing side has been eliminated, with the sign corresponding to the winning side,
    i.e. positive = row player.
    
    The optimization is done with a canonical Lanchester exponent of 1, 
    corresponding to Lanchester's linear law.
    The resulting handicaps are then raised to 1 / exponent.
    """
    rectifier = zerosum.function.ReciprocalLinearRectifier()
    
    def __init__(self, base_matrix, exponent):
        self.base_matrix = base_matrix
        if exponent < 1.0 or exponent > 2.0:
            warnings.warn("Lanchester exponent of %0.2f is not within the conventional interval [1.0, 2.0]." % exponent, ValueWarning)
        self.exponent = exponent
        check_non_negative(self.base_matrix)
        check_shape(self.base_matrix, self.row_weights, self.col_weights)
    
    def handicap_function(self, row_handicaps, col_handicaps):
        relative_strengths = self.base_matrix * col_handicaps[None, :] / row_handicaps[:, None]
        row_winner = relative_strengths > 1.0
        col_winner = relative_strengths < 1.0
        payoff_matrix = numpy.zeros_like(relative_strengths)
        payoff_matrix[row_winner] = 1.0 - 1.0 / relative_strengths[row_winner]
        payoff_matrix[col_winner] = -1.0 + relative_strengths[col_winner]
        return payoff_matrix
        
    def row_derivative(self, row_handicaps, col_handicaps):
        relative_strengths = self.base_matrix * col_handicaps[None, :] / row_handicaps[:, None]
        drelative_strengths = -self.base_matrix * col_handicaps[None, :] / numpy.square(row_handicaps)[:, None]
        row_winner = relative_strengths > 1.0
        d_payoff_matrix = numpy.copy(drelative_strengths)
        d_payoff_matrix[row_winner] /= numpy.square(relative_strengths[row_winner])
        return d_payoff_matrix
        
    def col_derivative(self, row_handicaps, col_handicaps):
        relative_strengths = self.base_matrix * col_handicaps[None, :] / row_handicaps[:, None]
        drelative_strengths = self.base_matrix / row_handicaps[:, None]
        row_winner = relative_strengths > 1.0  
        d_payoff_matrix = numpy.copy(drelative_strengths)
        d_payoff_matrix[row_winner] /= numpy.square(relative_strengths[row_winner])
        return d_payoff_matrix
        
    def decanonicalize(self, handicaps_canonical, payoff_matrix_canonical):
        handicaps = numpy.power(handicaps_canonical, 1.0 / self.exponent)
        return handicaps, payoff_matrix_canonical
        
class LanchesterNonSymmetricBalance(LanchesterBalance,NonSymmetricBalance):
    def __init__(self, base_matrix, exponent = 1.0, value = 0.0, row_weights = None, col_weights = None, fix_index = True):
        """
        Args:
            base_matrix: Should be strictly positive.
            exponent: The Lanchester exponent. 
                The optimization is computed using a Lanchester linear law. 
                Other Lanchester exponents are handled simply by 
                raising the resulting handicap values to (1 / exponent).
            value: Desired value of the game. Should be in the interval (-1, 1).
            row_weights, col_weights: Defines the desired Nash equilibrium in terms of strategy probability weights. 
                If only an integer is specified, a uniform distribution will be used.
            fix_index: Since this handicap function is invariant with respect to a global scale, we default to True.
        """
        if row_weights is None: row_weights = base_matrix.shape[0]
        if col_weights is None: col_weights = base_matrix.shape[1]
        
        if value <= -1.0 or value >= 1.0:
            raise ValueError("value %0.2f is not in the interval (-1, 1)" % value)
        
        NonSymmetricBalance.__init__(self, row_weights, col_weights, value = value, fix_index = fix_index)
        LanchesterBalance.__init__(self, base_matrix, exponent)
        
    def optimize(self, *args, **kwargs):
        """
        Compute the handicaps that balance the game using scipy.optimize.root.
        
        Args:
            As NonSymmetricBalance.optimize().
        
        Returns:
            As NonSymmetricBalance.optimize().
        """
        result = NonSymmetricBalance.optimize(self, *args, **kwargs)
        return result

class LanchesterSymmetricBalance(LanchesterBalance,SymmetricBalance):
    def __init__(self, base_matrix, exponent = 1.0, strategy_weights = None, fix_index = True):
        """
        Args:
            base_matrix: Should be strictly positive.
            exponent: The Lanchester exponent. 
                The optimization is computed using a Lanchester linear law. 
                Other Lanchester exponents are handled simply by 
                raising the resulting handicap values to (1 / exponent).
            strategy_weights: Defines the desired Nash equilibrium in terms of strategy probability weights. 
                If only an integer is specified, a uniform distribution will be used.
            fix_index: Since this handicap function is invariant with respect to a global scale, we default to True.
        """
        if strategy_weights is None: strategy_weights = base_matrix.shape[0]
        
        check_square(base_matrix)
        check_log_skew_symmetry(base_matrix)

        SymmetricBalance.__init__(self, strategy_weights, fix_index = fix_index)
        LanchesterBalance.__init__(self, base_matrix, exponent)
    
    def optimize(self, *args, **kwargs):
        """
        Compute the handicaps that balance the game using scipy.optimize.root.
        
        Args:
            As SymmetricBalance.optimize().
        
        Returns:
            As SymmetricBalance.optimize().
        """
        result = SymmetricBalance.optimize(self, *args, **kwargs)
        return result
