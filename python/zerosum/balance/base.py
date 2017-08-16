import zerosum.function
import numpy
import scipy.optimize
import warnings
    
class Balance():
    """ 
    Base class for balancing. Not intended to be used directly.
    Subclasses should extend either NonSymmetricBalance or SymmetricBalance.
    They also need to implement handicap_function, and optionally row_derivative and col_derivative 
    (Or just row_derivative for SymmetricBalance.)
    """
    
    """
    Methods and attributes to be set by subclasses:
    """
    
    rectifier = None
    """
    By default handicaps have the range (-inf, inf). 
    However, depending on the handicap function, it may only make sense to have
    strictly positive handicaps in the range (0, inf). 
    To do this we can use a strictly monotonically increasing rectifier function
    to map from the raw optimization values x to the handicaps (0, inf).
    Generally zerosum.function.ReciprocalLinearRectifier() is a good choice;
    while exponentials are mathematically elegant, they tend to suffer from accidental overflow.
    """
    
    def handicap_function(self, row_handicaps, col_handicaps):
        """
            Args:
                row_handicaps, col_handicaps 
            Returns:
                The payoff matrix.
                Each element of the payoff matrix should depend only on the corresponding row and column handicap.
                It is highly desirable that the function be strictly monotonically decreasing 
                in row_handicap and strictly monotonically increasing in col_handicap for every element. 
                NOTE: In the symmetric case the function should also have the property that 
                handicap_function(row_handicaps, col_handicaps) = -handicap_function(col_handicaps, row_handicaps) + value of the game
                where the value of the game is constant.
                This means that for any setting of the handicaps the payoff matrix is skew-symmetric plus the value of the game.
                In particular, all diagonal elements should be equal to the value of the game.
        """
        raise NotImplementedError
            
    # Optional: implement the following methods in subclasses:
    #def row_derivative(self, row_handicaps, col_handicaps):
        """Returns: the derivative of the payoff matrix with respect to the row handicaps."""
        
    #def col_derivative(self, row_handicaps, col_handicaps):
        """Returns: the derivative of the payoff matrix with respect to the column handicaps."""
    
    """
    Common methods.
    """
    
    def optimize(self, x0 = None, method = 'lm', check_derivative = False, check_jacobian = False, *args, **kwargs):
        """
        
        Args:
            x0: Starting point of the optimization. Defaults to a zero vector.
            check_derivative, check_jacobian:
                Can be used to check the provided row_derivative, col_derivative 
                against a finite difference approximation with the provided epsilon.
                A value of True uses a default value for epsilon.
            
            *args, **kwargs: Passed to scipy.optimize.root.
                In particular you may want to consider changing the solver method 
                if the default is not producing good results.
            
            Returns: 
            The result of scipy.optimize.root, with the following additional values:
                result.handicaps: The solved handicap vector, concatenated for rows and columns if appropriate.
                result.F: The resulting payoff matrix.
        """
        if x0 is None:
            x0 = numpy.zeros((self.handicap_count))
        
        # Keep the initial handicap of fix_index.
        if self.fix_index is not False:
            if x0.size == self.handicap_count:
                fix_x = x0[self.fix_index]
                x0 = numpy.delete(x0, self.fix_index)
            else:
                fix_x = 0.0
                
        def x_to_handicaps(self, x):
            if self.fix_index is not False:
                full_x = numpy.insert(x, self.fix_index, fix_x)
            else:
                full_x = x
                
            if self.rectifier is None:
                handicaps = full_x
            else:
                handicaps = self.rectifier.evaluate(full_x)
                
            return handicaps
        
        def fun(x):
            handicaps = x_to_handicaps(self, x)
            if check_derivative is not False:
                row_handicaps, col_handicaps = self.split_handicaps(handicaps)
                self.check_row_derivative(row_handicaps, col_handicaps, epsilon = check_derivative)
                self.check_col_derivative(row_handicaps, col_handicaps, epsilon = check_derivative)
            if check_jacobian is not False: 
                self.check_jacobian(handicaps, epsilon = check_jacobian)
            y = self.objective(handicaps)
            if self.fix_index is not False:
                y = numpy.delete(y, self.fix_index)
            return y
    
        if self.has_derivative():
            def jac(x):
                handicaps = x_to_handicaps(self, x)
                J = self.jacobian(handicaps)
                if self.fix_index is not False:
                    J = numpy.delete(J, self.fix_index, axis = 0)
                    J = numpy.delete(J, self.fix_index, axis = 1)
                if self.rectifier is not None:
                    J = J * self.rectifier.derivative(x)[None, :]
                return J
        else:
            jac = None
        
        result = scipy.optimize.root(fun = fun, x0 = x0, jac = jac, method = method, *args, **kwargs)
        
        result.handicaps = x_to_handicaps(self, result.x)
        result.row_handicaps, result.col_handicaps = self.split_handicaps(result.handicaps)
        result.F = self.handicap_function(result.row_handicaps, result.col_handicaps)
        
        return result
        
    def has_derivative(self):
        try:
            self.row_derivative, self.col_derivative
            return True
        except AttributeError:
            return False
    
    def jacobian_fd(self, epsilon = None):
        """ Computes a finite (central) difference approximation of the Jacobian. """
        if epsilon in [None, True]: epsilon = numpy.sqrt(numpy.finfo(float).eps)
        def result(handicaps):
            J = numpy.zeros((self.handicap_count, self.handicap_count))
            
            for input_index in range(self.handicap_count):
                hdp = handicaps.copy()
                hdp[input_index] += epsilon * 0.5
                
                hdn = handicaps.copy()
                hdn[input_index] -= epsilon * 0.5
                
                J[:, input_index] = (self.objective(hdp) - self.objective(hdn)) / epsilon
            
            return J
        return result

    def check_jacobian(self, handicaps = None, epsilon = None):
        """ 
        Checks the Jacobian computed from the handicap function derivatives 
        against a finite difference approximation. 
        """
        if handicaps is None: handicaps = numpy.zeros(self.handicap_count)
        J = self.jacobian(handicaps)
        jac_fd = self.jacobian_fd(epsilon = epsilon)
        result = J - jac_fd(handicaps)
        print('Maximum difference between evaluated Jacobian and finite difference:', 
            numpy.max(numpy.abs(result)))
        return result
        
    def row_derivative_combined_fd(self, row_handicaps, col_handicaps, epsilon = None):
        """ 
        Computes a finite (central) difference approximation of derivative of the handicap function 
        with respect to the corresponding row handicap. 
        """
        if epsilon in [None, True]: epsilon = numpy.sqrt(numpy.finfo(float).eps)
        row_handicaps_N = row_handicaps - epsilon * 0.5
        row_handicaps_P = row_handicaps + epsilon * 0.5
        return (self.handicap_function(row_handicaps_P, col_handicaps) - self.handicap_function(row_handicaps_N, col_handicaps)) / epsilon
        
    def col_derivative_combined_fd(self, row_handicaps, col_handicaps, epsilon = None):
        """ 
        Computes a finite (central) difference approximation of derivative of the handicap function 
        with respect to the corresponding column handicap. 
        """
        if epsilon in [None, True]: epsilon = numpy.sqrt(numpy.finfo(float).eps)
        col_handicaps_N = col_handicaps - epsilon * 0.5
        col_handicaps_P = col_handicaps + epsilon * 0.5
        return (self.handicap_function(row_handicaps, col_handicaps_P) - self.handicap_function(row_handicaps, col_handicaps_N)) / epsilon
        
    def check_row_derivative(self, row_handicaps, col_handicaps, epsilon = None):
        """ 
        Checks the derivative of the handicap function with respect to the corresponding row handicap 
        against a finite difference approximation.
        Also checks that all row derivatives are negative.
        """
        direct = self.row_derivative(row_handicaps, col_handicaps)
        fd = self.row_derivative_combined_fd(row_handicaps, col_handicaps, epsilon)
        if numpy.any(direct >= 0.0) or numpy.any(fd >= 0.0):
            msg = 'Found a non-negative row derivative for\nrow_handicaps = %s\ncol_handicaps = %s.' % (row_handicaps, col_handicaps)
            msg += '\nIt is highly desirable that the handicap function be strictly monotonically decreasing in the row handicap.'
            warnings.warn(msg, DerivativeWarning)
        result = direct - fd
        print('Maximum difference between evaluated row_derivative and finite difference:', 
            numpy.max(numpy.abs(result)))
        return result
    
    def check_col_derivative(self, row_handicaps, col_handicaps, epsilon = None):
        """
        Checks the derivative of the handicap function with respect to the corresponding column handicap 
        against a finite difference approximation.
        Also checks that all column derivatives are negative.
        """
        direct = self.col_derivative(row_handicaps, col_handicaps)
        fd = self.col_derivative_combined_fd(row_handicaps, col_handicaps, epsilon)
        if numpy.any(direct <= 0.0) or numpy.any(fd <= 0.0):
            msg = 'Found a non-positive column derivative for\nrow_handicaps = %s\ncol_handicaps = %s.' % (row_handicaps, col_handicaps)
            msg += '\nIt is highly desirable that the handicap function be strictly monotonically increasing in the column handicap.'
            warnings.warn(msg, DerivativeWarning)
        result = direct - fd
        print('Maximum difference between evaluated col_derivative and finite difference:', 
            numpy.max(numpy.abs(result)))
        return result

class NonSymmetricBalance(Balance):
    """
    This version of Balance for non-symmetric games, where each player is choosing
    from an independent set of strategies.
    """
    def __init__(self, row_weights, col_weights, value = 0.0, fix_index = False):
        """
        Args:
            row_weights, col_weights: Defines the desired Nash equilibrium 
                in terms of row and column strategy probability weights. 
                If only an integer is specified, a uniform distribution will be used.
                Weights will be normalized.
            value: The desired value of the resulting game. 
                This is equal to the row player's payoff and the negative of the column player's payoff.
            fix_index: If set to an integer, this will fix one handicap at its starting value and ignore the corresponding payoff.
                This is useful if the handicap function is known to have a degree of invariance.
                If set to True, a strategy with maximum weight will be selected.
                The value None is reserved for subclasses to implement their own default fix_index.
        Raises:
            ValueError if only one of row_derivative and col_derivative is provided.
        """
    
        self.row_count, self.row_weights, self.row_objective_weights = _process_weights(row_weights)
        self.col_count, self.col_weights, self.col_objective_weights = _process_weights(col_weights)
        
        self.handicap_count = self.row_count + self.col_count
        
        self.value = value
        
        weights = numpy.concatenate((self.row_weights, self.col_weights))
        if fix_index is True:
            # Select the first nonzero weight.
            fix_index = numpy.argmax(weights)
        elif fix_index is not False:
            if weights[fix_index] == 0.0:
                warnings.warn('fix_index %d corresponds to a strategy with zero weight.' % fix_index, ValueWarning)
        
        self.fix_index = fix_index
        
    def split_handicaps(self, handicaps):
        """ Splits handicaps into row_handicaps and col_handicaps. """
        return handicaps[:self.row_count], handicaps[-self.col_count:]
        
    def objective(self, handicaps):
        """
        Compute the objective vector, which is desired to be zero. 
        This is the expected payoff of each strategy for that player, times the weight of that strategy.
        In order to balance them at the edge of being played, zero-weighted strategies are given a weight of 1.0. 
        This works since they do not affect the expected payoff of other strategies.
        """
        
        row_handicaps, col_handicaps = self.split_handicaps(handicaps)
    
        F = self.handicap_function(row_handicaps, col_handicaps)
        
        # Dot products are weighted.
        row_objectives = (numpy.tensordot(F, self.col_weights, axes = ([1], [0])) - self.value) * self.row_objective_weights
        col_objectives = (self.value - numpy.tensordot(F, self.row_weights, axes = ([0], [0]))) * self.col_objective_weights
        
        return numpy.concatenate((row_objectives, col_objectives))
        
    def jacobian(self, handicaps):
        """ Compute the Jacobian of the objective using the provided handicaps. """
        
        row_handicaps, col_handicaps = self.split_handicaps(handicaps)
        
        # J_ij = derivative of payoff i with respect to handicap j.
        
        dFdr = self.row_derivative(row_handicaps, col_handicaps)
        dFdc = self.col_derivative(row_handicaps, col_handicaps)
        
        # Derivative of row payoffs with respect to row handicaps.
        Jrr = numpy.tensordot(dFdr, self.col_weights, axes = ([1], [0])) * self.row_objective_weights
        Jrr = numpy.diag(Jrr)
        
        # Derivative of col payoffs with respect to col handicaps.
        Jcc = -numpy.tensordot(dFdc, self.row_weights, axes = ([0], [0])) * self.col_objective_weights
        Jcc = numpy.diag(Jcc)
        
        # Derivative of row payoffs with respect to col handicaps.
        Jrc = dFdc * self.col_weights[None, :] * self.row_objective_weights[:, None]
        
        # Derivative of col payoffs with respect to row handicaps.
        Jcr = -dFdr * self.row_weights[:, None] * self.col_objective_weights[None, :]
        Jcr = numpy.transpose(Jcr)
        
        # Assemble full Jacobian.
        J = numpy.block([[Jrr, Jrc],
                         [Jcr, Jcc]])
        
        return J

class SymmetricBalance(Balance):
    def __init__(self, strategy_weights, row_derivative = None, value = None, fix_index = False):
        """
        This version of Balance for symmetric games, 
        where both players are choosing from the same set of strategies.
        Thus there are no independent inputs for column strategies.
        
        Args:
            strategy_weights: Defines the desired Nash equilibrium in terms of strategy probability weights. 
                If only an integer is specified, a uniform distribution will be used.
            value: Value of the game. 
                If not supplied it will be set automatically based on the diagonal elements
                when the payoff matrix is first evaluated.
            fix_index: If set to an integer, this will fix one handicap at its starting value and ignore the corresponding payoff.
                This is useful if the handicap function is known to have a degree of invariance.
                If set to True, a strategy with maximum weight will be selected.
                The value None is reserved for subclasses to implement their own default fix_index.
        """
        self.handicap_count, self.strategy_weights, self.strategy_objective_weights = _process_weights(strategy_weights)
        self.row_count = self.handicap_count
        self.col_count = self.handicap_count
            
        self.value = value
        
        if fix_index is True:
            # Select the maximum weight.
            fix_index = numpy.argmax(self.strategy_weights)
        elif fix_index is not False:
            if self.strategy_weights[fix_index] == 0.0:
                warnings.warn('fix_index %d corresponds to a strategy with zero weight.' % fix_index, ValueWarning)
        
        self.fix_index = fix_index
   
    def split_handicaps(self, handicaps):
        """ Splits handicaps into row_handicaps and col_handicaps. """
        return handicaps, handicaps
        
    def objective(self, handicaps):
        """
        Compute the objective vector, which is desired to be zero. 
        This is the expected payoff of each strategy for that player, times the weight of that stategy.
        In order to balance them at the edge of being played, zero-weighted strategies are given a weight of 1.0. 
        This works since they do not affect the expected payoff of other strategies.
        """
        
        row_handicaps, col_handicaps = self.split_handicaps(handicaps)
        F = self.handicap_function(row_handicaps, col_handicaps)
        if self.value is None:
            self.value = numpy.average(numpy.diag(F), weights = self.strategy_weights)
        
        # Dot products are weighted.
        objectives = (numpy.tensordot(F, self.strategy_weights, axes = ([1], [0])) - self.value) * self.strategy_objective_weights
        
        return objectives
        
    def jacobian(self, handicaps):
        """ Compute the Jacobian of the objective using the provided row_derivative. """
        row_handicaps, col_handicaps = self.split_handicaps(handicaps)
        dFdr = self.row_derivative(row_handicaps, col_handicaps)
        
        # Derivative of row payoffs with respect to row handicaps.
        Jrr = numpy.tensordot(dFdr, self.strategy_weights, axes = ([1], [0])) * self.strategy_objective_weights
        Jrr = numpy.diag(Jrr)
        
        # Derivative of row payoffs with respect to col handicaps.
        dFdc = -numpy.transpose(dFdr)
        Jrc = dFdc * self.strategy_weights[None, :] * self.strategy_objective_weights[:, None]
        
        # Variables change both row and col handicaps at the same time, so Jacobian is the sum of their effects.
        J = Jrr + Jrc
        
        return J
    
    def col_derivative(self, row_handicaps, col_handicaps):
        """ Using the skew-symmetry property. """
        return -self.row_derivative(row_handicaps, col_handicaps).transpose()
        
class DerivativeWarning(RuntimeWarning):
    pass
    
class ValueWarning(RuntimeWarning):
    pass

def _process_weights(arg):
    """ 
    Helper function for processing weight arguments.
    
    Args:
        arg: Either:
            An integer, in which case the weights will be uniform over that many strategies.
            A weight distribution, which will be normalized to sum to 1.
            
    Returns:
        count: The number of weights/strategies.
        weights: The (normalized) weights.
        objective_weights: As weights, but 0-weights are replaced by 1.
            Used for weighting the objective function.
            
    Raises:
        ValueError: If weights sum to 0, or any of the weights are negative.
    
    """
    try:
        weights = arg.copy()
        count = weights.size
    except:
        weights = numpy.ones((arg)) / arg
        count = arg
        
    weight_sum = numpy.sum(weights)
    
    if weight_sum == 0.0:
        raise ValueError('Weights sum to 0.')
        
    if numpy.any(weights < 0.0):
        raise ValueError('Received negative weight(s).')
        
    weights = weights / weight_sum
    
    # Replace zeros with 1.0 / weights.size for purposes of weighting the objective vector.
    objective_weights = weights.copy()
    objective_weights[objective_weights == 0.0] = 1.0 / weights.size
    
    return count, weights, objective_weights