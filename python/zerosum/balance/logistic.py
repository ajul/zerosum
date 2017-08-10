from .base import *

class LogisticBalance():
    """
    A special case where the handicap functions are logistic functions 
    whose argument is row_handicap - col_handicap + offset, 
    where offset is chosen so that when all handicaps are zero the initial_payoff_matrix is recovered.
    Commonly payoffs represent win rates.
    """
    
    def __init__(self, initial_payoff_matrix, value):
        """
        Sets self.initial_payoff_matrix and self.max_payoff.
        Raises:
            ValueError: 
                If any element of initial_payoff_matrix is not in the open interval (0, max_payoff). 
                If the size of strategy_weights does not match the dimensions of initial_payoff_matrix.
            ValueWarning:
                If initial_payoff_matrix is not (close to) skew-symmetric plus a constant offset.
                If initial_payoff_matrix has elements close to 0 and/or max_payoff.
                    max_payoff is twice the value of the game.
        """
        if initial_payoff_matrix.shape[0] != self.strategy_weights.size:
            raise ValueError('The size of strategy_weights does not match the dimensions of initial_payoff_matrix.')
            
        # The maximum possible payoff (e.g. 100% win rate) is twice the value of the game.
        self.max_payoff = 2.0 * value
            
        # Check bounds.
        if numpy.any(initial_payoff_matrix <= 0.0) or numpy.any(initial_payoff_matrix >= self.max_payoff):
            raise ValueError('initial_payoff_matrix has element(s) not in the open interval (0, max_payoff), where max_payoff = %f is twice the value of the game.' % self.max_payoff)
        if numpy.any(numpy.isclose(initial_payoff_matrix, 0.0)) or numpy.any(numpy.isclose(initial_payoff_matrix, self.max_payoff)):
            warnings.warn('initial_payoff_matrix has element(s) close to 0 and/or max_payoff, where max_payoff = %f is twice the value of the game.' % self.max_payoff, ValueWarning)
            
        self.initial_payoff_matrix = initial_payoff_matrix
        self.initial_offset_matrix = numpy.log(self.max_payoff / initial_payoff_matrix - 1.0)
    
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
    #TODO: Finish and test this.
    def __init__(self, initial_payoff_matrix, value, row_weights = None, col_weights = None, fix_index = True):
        """
        Args:
            initial_payoff_matrix: The elements of initial_payoff_matrix must be in (0, max_payoff), 
                where max_payoff is twice the value of the game.
                The initial_payoff_matrix should be skew-symmetric plus a constant offset (namely the value of the game).
                In particular, all diagonal elements should be equal to the value of the game.
            strategy_weights: Defines the desired Nash equilibrium in terms of strategy probability weights. 
                If only an integer is specified, a uniform distribution will be used.
            fix_index: Since this handicap function is invariant with respect to a global offset, we default to True.
        Raises:
            ValueError:
                If any element of initial_payoff_matrix is not in the open interval (0, max_payoff). 
                If the size of strategy_weights does not match the dimensions of initial_payoff_matrix.
            ValueWarning:
                If initial_payoff_matrix is not (close to) skew-symmetric plus a constant offset.
                If initial_payoff_matrix has elements close to 0 and/or max_payoff.
                    max_payoff is twice the value of the game.
        """
        if row_weights is None: row_weights = initial_payoff_matrix.shape[0]
        if col_weights is None: col_weights = initial_payoff_matrix.shape[1]
        
        NonSymmetricBalance.__init__(self, row_weights, col_weights, value = value, fix_index = fix_index)
        LogisticBalance.__init__(self, initial_payoff_matrix, value)
        
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
    def __init__(self, initial_payoff_matrix, strategy_weights = None, fix_index = True):
        """
        Args:
            initial_payoff_matrix: The elements of initial_payoff_matrix must be in (0, max_payoff), 
                where max_payoff is twice the value of the game.
                The initial_payoff_matrix should be skew-symmetric plus a constant offset (namely the value of the game).
                In particular, all diagonal elements should be equal to the value of the game.
            strategy_weights: Defines the desired Nash equilibrium in terms of strategy probability weights. 
                If only an integer is specified, a uniform distribution will be used.
            fix_index: Since this handicap function is invariant with respect to a global offset, we default to True.
        Raises:
            ValueError: 
                If initial_payoff_matrix is not square.
                If any element of initial_payoff_matrix is not in the open interval (0, max_payoff). 
                If the size of strategy_weights does not match the dimensions of initial_payoff_matrix.
            ValueWarning:
                If initial_payoff_matrix is not (close to) skew-symmetric plus a constant offset.
                If initial_payoff_matrix has elements close to 0 and/or max_payoff.
                    max_payoff is twice the value of the game.
        """
        if strategy_weights is None: strategy_weights = initial_payoff_matrix.shape[0]
        if initial_payoff_matrix.shape[0] != initial_payoff_matrix.shape[1]:
            raise ValueError('initial_payoff_matrix is not square.')
        
        value = initial_payoff_matrix[0, 0]
        
        SymmetricBalance.__init__(self, strategy_weights, fix_index = fix_index)
        LogisticBalance.__init__(self, initial_payoff_matrix, value)
        
        # Check skew-symmetry. 
        initial_payoff_matrix_compliment_transpose = self.max_payoff - self.initial_payoff_matrix.transpose()
        if not numpy.allclose(initial_payoff_matrix, initial_payoff_matrix_compliment_transpose):
            warnings.warn('The difference between initial_payoff_matrix and the value of the game is not (close to) skew-symmetric.', ValueWarning)
        
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