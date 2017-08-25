import zerosum.function
import numpy
import scipy.optimize
import warnings

# Same as scipy.optimize.
_epsilon = numpy.sqrt(numpy.finfo(float).eps)
    
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
    
    def handicap_function(self, h_r, h_c):
        """
        Mandatory to override.
        
        Args:
            h_r, h_c: The canonical row and column handicaps.
        Returns:
            The canoncial payoff matrix.
            Each element of the canoncial payoff matrix should depend only on the corresponding canonical row and column handicap.
            It is highly desirable that the function be strictly monotonically decreasing 
            in row_handicap and strictly monotonically increasing in col_handicap for every element. 
            NOTE: In the symmetric case the function should also have the property that 
            handicap_function(h_r, h_c) = -handicap_function(h_c, h_r) + value of the game
            where the value of the game is constant.
            This means that for any setting of h_r, h_c the payoff matrix is skew-symmetric plus the value of the game.
            In particular, all diagonal elements should be equal to the value of the game.
        """
        raise NotImplementedError("Balance subclasses must implement a handicap_function.")

    rectifier = None
    """
    Optional to override.
    
    By default the canonical handicaps have the range (-inf, inf). 
    However, depending on the handicap function, it may only make sense to have
    strictly positive canonical handicaps in the range (0, inf). 
    To do this we can use a strictly monotonically increasing rectifier function
    to map from the raw optimization values x to the canonical handicaps.
    Generally zerosum.function.ReciprocalLinearRectifier() is a good choice;
    while exponentials are mathematically elegant, they tend to suffer from accidental overflow.
    """
            
    def row_derivative(self, h_r, h_c):
        """
        Optional to override. Defaults to a finite-difference implementation.
        
        Returns: the derivative of the payoff matrix with respect to the row h.
        """
        return self.row_derivative_fd(h_r, h_c)
        
    def col_derivative(self, h_r, h_c):
        """
        Optional to override.  Defaults to a finite-difference implementation.
        
        Returns: the derivative of the payoff matrix with respect to the column h.
        """
        return self.col_derivative_fd(h_r, h_c)
    
    def decanonicalize(self, h, F):
        """
        In some cases the problem may be transformed into some canonical form before solving it.
        Subclasses implement this method to transform the result back into a form corresponding to the problem statement.
        This may modify result.F and/or result.handicap.
        
        Args:
            h: the canonical handicaps.
            F: the canonical payoff matrix.
        Returns:
            handicaps
            F
        """
        return h, F
    
    """
    Common methods.
    """
    
    def optimize(self, x0 = None, method = 'lm', use_jacobian = True, check_derivative = False, check_jacobian = False, *args, **kwargs):
        """
        Solves the balance problem using scipy.optimize.root.
        Args:
            x0: Starting point of the optimization. Defaults to a zero vector.
            method: Optimization method to be used by scipy.optimize.root. 
                We default to 'lm' since it seems to produce the best results empirically.
            use_jacobian: If set to true, the Jacobian will be computed from the row and column derivatives,
                instead of using scipy.optimize.root's default Jacobian.
            check_derivative, check_jacobian:
                Can be used to check the provided row_derivative, col_derivative 
                against a finite difference approximation with the provided epsilon.
                A value of True uses a default value for epsilon.
            
            *args, **kwargs: Passed to scipy.optimize.root.
                In particular you may want to consider changing the solver method 
                if the default is not producing good results.
            
            Returns: 
            The result of scipy.optimize.root, with the following additional values:
                result.row_handicaps: The solved decanonicalized row handicaps.
                    (Canonical: result.h_r)
                result.col_handicaps: The solved decanonicalized row handicaps.
                    (Canonical: result.h_c)
                result.payoff_matrix: The solved decanonicalized payoff matrix.
                    (Canonical: result.F)
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
                
        def x_to_h(self, x):
            """ 
            Converts the raw optimization variables x to the 
            canonical handicaps h.
            """
            if self.fix_index is not False:
                full_x = numpy.insert(x, self.fix_index, fix_x)
            else:
                full_x = x
                
            if self.rectifier is None:
                h = full_x
            else:
                h = self.rectifier.evaluate(full_x)
                
            return h
        
        def fun(x):
            """
            The objective function in terms of the raw optimization variables.
            """
            h = x_to_h(self, x)
            y = self.objective(h)
            if self.fix_index is not False:
                y = numpy.delete(y, self.fix_index)
                
            if check_derivative is not False:
                h_r, h_c = self.split_handicaps(h)
                epsilon = _epsilon if check_derivative is True else check_derivative
                self.check_row_derivative(h_r, h_c, epsilon = epsilon)
                self.check_col_derivative(h_r, h_c, epsilon = epsilon)
            if check_jacobian is not False:
                epsilon = _epsilon if check_jacobian is True else check_jacobian
                self.check_jacobian(h, epsilon = check_jacobian)
            
            return y
    
        if use_jacobian:
            def jac(x):
                """
                Jacobian of the objective function.
                """
                h = x_to_h(self, x)
                J = self.jacobian(h)
                if self.fix_index is not False:
                    J = numpy.delete(J, self.fix_index, axis = 0)
                    J = numpy.delete(J, self.fix_index, axis = 1)
                if self.rectifier is not None:
                    J = J * self.rectifier.derivative(x)[None, :]
                return J
        else:
            jac = None
        
        result = scipy.optimize.root(fun = fun, x0 = x0, jac = jac, method = method, *args, **kwargs)
        
        result.h = x_to_h(self, result.x)
        result.h_r, result.h_c = self.split_handicaps(result.h)
        result.F = self.handicap_function(result.h_r, result.h_c)
        
        # Decanonicalize the canonical handicaps into the final values.
        result.handicaps, result.payoff_matrix = self.decanonicalize(result.h, result.F)
        result.row_handicaps, result.col_handicaps = self.split_handicaps(result.h)
        
        return result
    
    """
    Methods for checking derivatives and Jacobians.
    """
    
    def jacobian_fd(self, h, epsilon = _epsilon):
        """ Computes a finite (central) difference approximation of the Jacobian. """
        J = numpy.zeros((self.handicap_count, self.handicap_count))
        
        for input_index in range(self.handicap_count):
            hdp = h.copy()
            hdp[input_index] += epsilon * 0.5
            
            hdn = h.copy()
            hdn[input_index] -= epsilon * 0.5
            
            J[:, input_index] = (self.objective(hdp) - self.objective(hdn)) / epsilon
        
        return J

    def check_jacobian(self, h = None, epsilon = _epsilon):
        """ 
        Checks the Jacobian computed from the handicap function derivatives 
        against a finite difference approximation. 
        """
        if h is None: h = numpy.zeros(self.handicap_count)
        result = self.jacobian(h) - self.jacobian_fd(h)
        print('Maximum difference between evaluated Jacobian and finite difference:', 
            numpy.max(numpy.abs(result)))
        return result
        
    def row_derivative_fd(self, h_r, h_c, epsilon = _epsilon):
        """ 
        Computes a finite (central) difference approximation of derivative of the handicap function 
        with respect to the corresponding row handicap. 
        """
        
        h_r_N = h_r - epsilon * 0.5
        h_r_P = h_r + epsilon * 0.5
        return (self.handicap_function(h_r_P, h_c) - self.handicap_function(h_r_N, h_c)) / epsilon
        
    def col_derivative_fd(self, h_r, h_c, epsilon = _epsilon):
        """ 
        Computes a finite (central) difference approximation of derivative of the handicap function 
        with respect to the corresponding column handicap. 
        """
        h_c_N = h_c - epsilon * 0.5
        h_c_P = h_c + epsilon * 0.5
        return (self.handicap_function(h_r, h_c_P) - self.handicap_function(h_r, h_c_N)) / epsilon
        
    def check_row_derivative(self, h_r, h_c, epsilon = _epsilon):
        """ 
        Checks the derivative of the handicap function with respect to the corresponding row handicap 
        against a finite difference approximation.
        Also checks that all row derivatives are negative.
        """
        direct = self.row_derivative(h_r, h_c)
        fd = self.row_derivative_fd(h_r, h_c, epsilon)
        if numpy.any(direct >= 0.0) or numpy.any(fd >= 0.0):
            msg = 'Found a non-negative row derivative for\nh_r = %s\nh_c = %s.' % (h_r, h_c)
            msg += '\nIt is highly desirable that the handicap function be strictly monotonically decreasing in the row handicap.'
            warnings.warn(msg, DerivativeWarning)
        result = direct - fd
        print('Maximum difference between evaluated row_derivative and finite difference:', 
            numpy.max(numpy.abs(result)))
        return result
    
    def check_col_derivative(self, h_r, h_c, epsilon = _epsilon):
        """
        Checks the derivative of the handicap function with respect to the corresponding column handicap 
        against a finite difference approximation.
        Also checks that all column derivatives are negative.
        """
        direct = self.col_derivative(h_r, h_c)
        fd = self.col_derivative_fd(h_r, h_c, epsilon)
        if numpy.any(direct <= 0.0) or numpy.any(fd <= 0.0):
            msg = 'Found a non-positive column derivative for\nh_r = %s\nh_c = %s.' % (h_r, h_c)
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
        
    def split_handicaps(self, h):
        """ Splits handicaps (canonical or not) into row and col handicaps."""
        return h[:self.row_count], h[-self.col_count:]
        
    def objective(self, h):
        """
        Compute the objective vector, which is desired to be zero. 
        This is the expected payoff of each strategy for that player, times the weight of that strategy.
        In order to balance them at the edge of being played, zero-weighted strategies are given a weight of 1.0. 
        This works since they do not affect the expected payoff of other strategies.
        """
        
        h_r, h_c = self.split_handicaps(h)
    
        F = self.handicap_function(h_r, h_c)
        
        # Dot products are weighted.
        row_objectives = (numpy.tensordot(F, self.col_weights, axes = ([1], [0])) - self.value) * self.row_objective_weights
        col_objectives = (self.value - numpy.tensordot(F, self.row_weights, axes = ([0], [0]))) * self.col_objective_weights
        
        return numpy.concatenate((row_objectives, col_objectives))
        
    def jacobian(self, h):
        """ Compute the Jacobian of the objective using the provided canonical handicaps h. """
        
        h_r, h_c = self.split_handicaps(h)
        
        # J_ij = derivative of payoff i with respect to handicap j.
        
        dFr = self.row_derivative(h_r, h_c)
        dFc = self.col_derivative(h_r, h_c)
        
        # Derivative of row payoffs with respect to row h.
        Jrr = numpy.tensordot(dFr, self.col_weights, axes = ([1], [0])) * self.row_objective_weights
        Jrr = numpy.diag(Jrr)
        
        # Derivative of col payoffs with respect to col h.
        Jcc = -numpy.tensordot(dFc, self.row_weights, axes = ([0], [0])) * self.col_objective_weights
        Jcc = numpy.diag(Jcc)
        
        # Derivative of row payoffs with respect to col h.
        Jrc = dFc * self.col_weights[None, :] * self.row_objective_weights[:, None]
        
        # Derivative of col payoffs with respect to row h.
        Jcr = -dFr * self.row_weights[:, None] * self.col_objective_weights[None, :]
        Jcr = numpy.transpose(Jcr)
        
        # Assemble full Jacobian.
        J = numpy.block([[Jrr, Jrc],
                         [Jcr, Jcc]])
        
        return J

class SymmetricBalance(Balance):
    def __init__(self, strategy_weights, value = None, fix_index = False):
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
        self.row_weights = self.strategy_weights
        self.col_weights = self.strategy_weights
            
        self.value = value
        
        if fix_index is True:
            # Select the maximum weight.
            fix_index = numpy.argmax(self.strategy_weights)
        elif fix_index is not False:
            if self.strategy_weights[fix_index] == 0.0:
                warnings.warn('fix_index %d corresponds to a strategy with zero weight.' % fix_index, ValueWarning)
        
        self.fix_index = fix_index
   
    def split_handicaps(self, h):
        """ Splits handicaps (canonical or not) into row and col handicaps."""
        return h, h
        
    def objective(self, h):
        """
        Compute the objective vector, which is desired to be zero. 
        This is the expected payoff of each strategy for that player, times the weight of that stategy.
        In order to balance them at the edge of being played, zero-weighted strategies are given a weight of 1.0. 
        This works since they do not affect the expected payoff of other strategies.
        """
        
        h_r, h_c = self.split_handicaps(h)
        F = self.handicap_function(h_r, h_c)
        if self.value is None:
            self.value = numpy.average(numpy.diag(F), weights = self.strategy_weights)
        
        # Dot products are weighted.
        objectives = (numpy.tensordot(F, self.strategy_weights, axes = ([1], [0])) - self.value) * self.strategy_objective_weights
        
        return objectives
        
    def jacobian(self, h):
        """ Compute the Jacobian of the objective using self.row_derivative. """
        h_r, h_c = self.split_handicaps(h)
        dFr = self.row_derivative(h_r, h_c)
        
        # Derivative of row payoffs with respect to row h.
        Jrr = numpy.tensordot(dFr, self.strategy_weights, axes = ([1], [0])) * self.strategy_objective_weights
        Jrr = numpy.diag(Jrr)
        
        # Derivative of row payoffs with respect to col h.
        dFc = -numpy.transpose(dFr)
        Jrc = dFc * self.strategy_weights[None, :] * self.strategy_objective_weights[:, None]
        
        # Variables change both row and col h at the same time, so Jacobian is the sum of their effects.
        J = Jrr + Jrc
        
        return J
    
    def col_derivative(self, h_r, h_c):
        """ Using the skew-symmetry property. """
        return -self.row_derivative(h_r, h_c).transpose()
        
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
