from .base import *
from .input_checks import *

class LogisticBalance():
    """
    A special case where the handicap functions are logistic functions 
    whose argument is row_handicap - col_handicap + offset, 
    where offset is chosen so that when all handicaps are zero the base_matrix is recovered.
    Commonly payoffs represent win rates.
    """
    
    def __init__(self, base_matrix):
        """
        Sets self.base_matrix.
        Args:
            base_matrix
        Raises:
            ValueError: 
                If any element of base_matrix is not in the open interval (0, max_payoff). 
            ValueWarning:
                If base_matrix is not (close to) skew-symmetric plus a constant offset.
                If base_matrix has elements close to 0 and/or max_payoff.
        """
            
        # Check bounds.
        if numpy.any(base_matrix <= 0.0) or numpy.any(base_matrix >= self.max_payoff):
            raise ValueError('base_matrix has element(s) not in the open interval (0, max_payoff), where max_payoff = %f is twice the value of the game.' % self.max_payoff)
        if numpy.any(numpy.isclose(base_matrix, 0.0)) or numpy.any(numpy.isclose(base_matrix, self.max_payoff)):
            warnings.warn('base_matrix has element(s) close to 0 and/or max_payoff, where max_payoff = %f is twice the value of the game.' % self.max_payoff, ValueWarning)
            
        self.base_matrix = base_matrix
        self.initial_offset_matrix = numpy.log(self.max_payoff / base_matrix - 1.0)
        
        check_shape(self.base_matrix, self.row_weights, self.col_weights)
    
    def handicap_function(self, row_handicaps, col_handicaps):
        # Normalized to the range (-0.5, 0.5).
        # This seems to perform better than using the original range.
        return 1.0 / (1.0 + numpy.exp(row_handicaps[:, None] - col_handicaps[None, :] + self.initial_offset_matrix)) - 0.5
        
    def row_derivative(self, row_handicaps, col_handicaps):
        payoffs = self.handicap_function(row_handicaps, col_handicaps)
        return payoffs * payoffs - 0.25
    
    def col_derivative(self, row_handicaps, col_handicaps):
        payoffs = self.handicap_function(row_handicaps, col_handicaps)
        return 0.25 - payoffs * payoffs
        
class LogisticNonSymmetricBalance(LogisticBalance, NonSymmetricBalance):
    def __init__(self, base_matrix, value, max_payoff, row_weights = None, col_weights = None, fix_index = True):
        """
        Args:
            base_matrix: The elements of base_matrix must be in (0, max_payoff), 
                where max_payoff is twice the value of the game.
                The base_matrix should be skew-symmetric plus a constant offset (namely the value of the game).
                In particular, all diagonal elements should be equal to the value of the game.
            strategy_weights: Defines the desired Nash equilibrium in terms of strategy probability weights. 
                If only an integer is specified, a uniform distribution will be used.
            fix_index: Since this handicap function is invariant with respect to a global offset, we default to True.
        Raises:
            ValueError:
                If any element of base_matrix is not in the open interval (0, max_payoff). 
                If the size of strategy_weights does not match the dimensions of base_matrix.
            ValueWarning:
                If base_matrix is not (close to) skew-symmetric plus a constant offset.
                If base_matrix has elements close to 0 and/or max_payoff.
                    max_payoff is twice the value of the game.
        """
        if row_weights is None: row_weights = base_matrix.shape[0]
        if col_weights is None: col_weights = base_matrix.shape[1]
        
        self.max_payoff = max_payoff
        normalized_value = value / max_payoff - 0.5
        
        NonSymmetricBalance.__init__(self, row_weights, col_weights, value = normalized_value, fix_index = fix_index)
        LogisticBalance.__init__(self, base_matrix)
        
    def optimize(self, *args, **kwargs):
        """
        Compute the handicaps that balance the game using scipy.optimize.root.
        
        Args:
            As NonSymmetricBalance.optimize().
        
        Returns:
            As NonSymmetricBalance.optimize().
            result.F is scaled to the original range (0, max_payoff).
        """
        result = NonSymmetricBalance.optimize(self, *args, **kwargs)
        # Expand F back to the original range (0, max_payoff).
        result.F = (result.F + 0.5) * self.max_payoff
        return result

class LogisticSymmetricBalance(LogisticBalance, SymmetricBalance):
    """
    Symmetric verison.
    """
    def __init__(self, base_matrix, strategy_weights = None, fix_index = True):
        """
        Args:
            base_matrix: The elements of base_matrix must be in (0, max_payoff), 
                where max_payoff is twice the value of the game.
                The base_matrix should be skew-symmetric plus a constant offset (namely the value of the game).
                In particular, all diagonal elements should be equal to the value of the game.
            strategy_weights: Defines the desired Nash equilibrium in terms of strategy probability weights. 
                If only an integer is specified, a uniform distribution will be used.
            fix_index: Since this handicap function is invariant with respect to a global offset, we default to True.
        Raises:
            ValueError: 
                If base_matrix is not square.
                If any element of base_matrix is not in the open interval (0, max_payoff). 
                If the size of strategy_weights does not match the dimensions of base_matrix.
            ValueWarning:
                If base_matrix is not (close to) skew-symmetric plus a constant offset.
                If base_matrix has elements close to 0 and/or max_payoff.
                    max_payoff is twice the value of the game.
        """
        if strategy_weights is None: strategy_weights = base_matrix.shape[0]
        
        value = base_matrix[0, 0]
        
        check_square(base_matrix)
        check_skew_symmetry(base_matrix, value)
        
        # The maximum possible payoff (e.g. 100% win rate) is twice the value of the game.
        self.max_payoff = 2.0 * value
            
        SymmetricBalance.__init__(self, strategy_weights, fix_index = fix_index)
        LogisticBalance.__init__(self, base_matrix)
        
    def optimize(self, *args, **kwargs):
        """
        Compute the handicaps that balance the game using scipy.optimize.root.
        
        Args:
            As SymmetricBalance.optimize().
        
        Returns:
            As SymmetricBalance.optimize().
            result.F is scaled to the original range (0, max_payoff).
        """
        result = SymmetricBalance.optimize(self, *args, **kwargs)
        # Expand F back to the original range (0, max_payoff).
        result.F = (result.F + 0.5) * self.max_payoff
        return result