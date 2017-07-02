import zerosum.function
import numpy
import scipy.optimize
import warnings

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
    
    nonzero_weight_index = numpy.flatnonzero(weights)[0]
    
    # Replace zeros with 1.0 / weights.size for purposes of weighting the objective vector.
    objective_weights = weights.copy()
    objective_weights[objective_weights == 0.0] = 1.0 / weights.size
    
    return count, weights, objective_weights, nonzero_weight_index
    
class Balance():
    """ 
    Base class for balancing. Not intended to be used directly.
    """
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
        
    def row_derivative_combined(self, handicaps): 
        """ 
        Computes a matrix consisting of the derivative of the handicap function 
        with respect to the corresponding row handicap. 
        """
        row_handicaps = handicaps[:self.row_count]
        col_handicaps = handicaps[-self.col_count:]
        
        return self.row_derivative(row_handicaps, col_handicaps)
        
    def col_derivative_combined(self, handicaps):
        """ 
        Computes a matrix consisting of the derivative of the handicap function 
        with respect to the corresponding column handicap. 
        """
        row_handicaps = handicaps[:self.row_count]
        col_handicaps = handicaps[-self.col_count:]
        
        return self.col_derivative(row_handicaps, col_handicaps)
        
    def row_derivative_combined_fd(self, handicaps, epsilon = None):
        """ 
        Computes a finite (central) difference approximation of derivative of the handicap function 
        with respect to the corresponding row handicap. 
        """
        if epsilon in [None, True]: epsilon = numpy.sqrt(numpy.finfo(float).eps)
        row_handicaps_N = handicaps[:self.row_count] - epsilon * 0.5
        row_handicaps_P = handicaps[:self.row_count] + epsilon * 0.5
        col_handicaps = handicaps[-self.col_count:]
        return (self.handicap_function(row_handicaps_P, col_handicaps) - self.handicap_function(row_handicaps_N, col_handicaps)) / epsilon
        
    def col_derivative_combined_fd(self, handicaps, epsilon = None):
        """ 
        Computes a finite (central) difference approximation of derivative of the handicap function 
        with respect to the corresponding column handicap. 
        """
        if epsilon in [None, True]: epsilon = numpy.sqrt(numpy.finfo(float).eps)
        row_handicaps = handicaps[:self.row_count]
        col_handicaps_N = handicaps[-self.col_count:] - epsilon * 0.5
        col_handicaps_P = handicaps[-self.col_count:] + epsilon * 0.5
        return (self.handicap_function(row_handicaps, col_handicaps_P) - self.handicap_function(row_handicaps, col_handicaps_N)) / epsilon
        
    def check_row_derivative(self, handicaps = None, epsilon = None):
        """ 
        Checks the derivative of the handicap function with respect to the corresponding row handicap 
        against a finite difference approximation.
        Also checks that all row derivatives are negative.
        """
        if handicaps is None: handicaps = numpy.zeros(self.handicap_count)
        direct = self.row_derivative_combined(handicaps)
        fd = self.row_derivative_combined_fd(handicaps, epsilon)
        if numpy.any(direct >= 0.0) or numpy.any(fd >= 0.0):
            msg = 'Found a non-negative row derivative for\nx = %s.' % handicaps
            msg += '\nIt is highly desirable that the handicap function be strictly monotonically decreasing in the row handicap.'
            warnings.warn(msg, DerivativeWarning)
        result = direct - fd
        print('Maximum difference between evaluated row_derivative and finite difference:', 
            numpy.max(numpy.abs(result)))
        return result
    
    def check_col_derivative(self, handicaps = None, epsilon = None):
        """
        Checks the derivative of the handicap function with respect to the corresponding column handicap 
        against a finite difference approximation.
        Also checks that all column derivatives are negative.
        """
        if handicaps is None: handicaps = numpy.zeros(self.handicap_count)
        direct = self.col_derivative_combined(handicaps)
        fd = self.col_derivative_combined_fd(handicaps, epsilon)
        if numpy.any(direct <= 0.0) or numpy.any(fd <= 0.0):
            msg = 'Found a non-positive column derivative for\nx = %s.' % handicaps
            msg += '\nIt is highly desirable that the handicap function be strictly monotonically increasing in the column handicap.'
            warnings.warn(msg, DerivativeWarning)
        result = direct - fd
        print('Maximum difference between evaluated col_derivative and finite difference:', 
            numpy.max(numpy.abs(result)))
        return result
    
    def optimize_common(self, x0 = None, check_derivative = False, check_jacobian = False, *args, **kwargs):
        """
        Common optimization code.
        
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
        if self.fix_index is not None:
            if x0.size == self.handicap_count:
                fix_handicap = x0[self.fix_index]
                x0 = numpy.delete(x0, self.fix_index)
            else:
                fix_handicap = 0.0
        
        def fun(x):
            if self.fix_index is not None:
                x = numpy.insert(x, self.fix_index, fix_handicap)
            if check_derivative is not False:
                self.check_row_derivative(x, epsilon = check_derivative)
                self.check_col_derivative(x, epsilon = check_derivative)
            if check_jacobian is not False: 
                self.check_jacobian(x, epsilon = check_jacobian)
            y = self.objective(x)
            if self.fix_index is not None:
                y = numpy.delete(y, self.fix_index)
            return y
    
        if self.row_derivative is None:
            jac = None
        else:
            def jac(x):
                if self.fix_index is not None:
                    x = numpy.insert(x, self.fix_index, fix_handicap)
                J = self.jacobian(x)
                if self.fix_index is not None:
                    J = numpy.delete(J, self.fix_index, axis = 0)
                    J = numpy.delete(J, self.fix_index, axis = 1)
                return J
        
        result = scipy.optimize.root(fun = fun, x0 = x0, jac = jac, *args, **kwargs)
        
        result.handicaps = result.x
        if self.fix_index is not None:
            result.handicaps = numpy.insert(result.handicaps, self.fix_index, fix_handicap)
        
        result.F = self.evaluate_payoff_matrix(result.handicaps)
        
        return result

class NonSymmetricBalance(Balance):
    """
    This version of Balance for non-symmetric games, where each player is choosing
    from an independent set of strategies.
    """
    def __init__(self, handicap_function, row_weights, col_weights, 
        row_derivative = None, col_derivative = None, value = 0.0, fix_index = None):
        """
        Args:
            handicap_function: A function that takes the arguments row_handicaps, col_handicaps 
                and produces the payoff matrix.
                Each element of the payoff matrix should depend only on the corresponding 
                row and column handicap.
                It is highly desirable that the function be strictly monotonically decreasing in row_handicap 
                and strictly monotonically increasing in col_handicap for every element. 
            row_weights, col_weights: Defines the desired Nash equilibrium 
                in terms of row and column strategy probability weights. 
                If only an integer is specified, a uniform distribution will be used.
                Weights will be normalized.
            row_derivative, col_derivative: Functions that take the arguments row_handicap, col_handicap 
                and produce the derviative of the payoff matrix with respect to each element's row or column handicap.
            value: The desired value of the resulting game. 
                This is equal to the row player's payoff and the negative of the column player's payoff.
            fix_index: If set, this will fix one handicap at its starting value and ignore the corresponding payoff.
                This is useful if the handicap function is known to have a degree of invariance.
                If set to true, a strategy with nonzero weight will automatically be selected.
        """
        self.handicap_function = handicap_function
        
        if (row_derivative is None) != (col_derivative is None):
            raise ValueError('Both row_derivative and col_derivative must be provided for the Jacobian to function.')
        
        self.row_derivative = row_derivative
        self.col_derivative = col_derivative
    
        self.row_count, self.row_weights, self.row_objective_weights, nonzero_weight_index = _process_weights(row_weights)
        self.col_count, self.col_weights, self.col_objective_weights, _ = _process_weights(col_weights)
        
        self.handicap_count = self.row_count + self.col_count
        
        self.value = value
        
        if fix_index is True:
            # Select the first nonzero weight.
            fix_index = nonzero_weight_index
        elif fix_index is not None:
            weights = numpy.concatenate((self.row_weights, self.col_weights))
            if weights[fix_index] == 0.0:
                raise ValueError('fix_index %d corresponds to a strategy with zero weight.' % fix_index)
        
        self.fix_index = fix_index
        
    def evaluate_payoff_matrix(self, handicaps):
        """ Evaluate F in terms of the variables, namely the handicap variable vectors. """
        row_handicaps = handicaps[:self.row_count]
        col_handicaps = handicaps[-self.col_count:]
        return self.handicap_function(row_handicaps, col_handicaps)
        
    def objective(self, handicaps):
        """
        Compute the objective vector, which is desired to be zero. 
        This is the expected payoff of each strategy for that player, times the weight of that stategy.
        In order to balance them at the edge of being played, zero-weighted strategies are given a weight of 1.0. 
        This works since they do not affect the expected payoff of other strategies.
        """
    
        F = self.evaluate_payoff_matrix(handicaps)
        
        # Dot products are weighted.
        row_objectives = (numpy.tensordot(F, self.col_weights, axes = ([1], [0])) - self.value) * self.row_objective_weights
        col_objectives = (self.value - numpy.tensordot(F, self.row_weights, axes = ([0], [0]))) * self.col_objective_weights
        
        return numpy.concatenate((row_objectives, col_objectives))
        
    def jacobian(self, handicaps):
        """ Compute the Jacobian of the objective using the provided row_derivative, col_derivative. """
        
        # J_ij = derivative of payoff i with respect to handicap j.
        
        dFdr = self.row_derivative_combined(handicaps)
        dFdc = self.col_derivative_combined(handicaps)
        
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
        J = numpy.bmat([[Jrr, Jrc],
                        [Jcr, Jcc]])
        
        return J
        
    def optimize(self, x0 = None, 
        check_derivative = False, check_jacobian = False, *args, **kwargs):
        """
        Compute the handicaps that balance the game using scipy.optimize.root.
        
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
                result.row_handicaps: The solved row handicaps.
                result.col_handicaps: The solved column handicaps.
                result.F: The resulting payoff matrix.
        """
        
        result = self.optimize_common(x0 = x0, 
            check_derivative = check_derivative, check_jacobian = check_jacobian, 
            *args, **kwargs)
        
        result.row_handicaps = result.handicaps[:self.row_count]
        result.col_handicaps = result.handicaps[-self.col_count:]
        
        return result

class SymmetricBalance(Balance):
    def __init__(self, handicap_function, strategy_weights, row_derivative = None, value = None, fix_index = None):
        """
        This version of Balance for symmetric games, 
        where both players are choosing from the same set of strategies.
        Thus there are no independent inputs for column strategies.
        
        Args:
            handicap_function: A function that takes the arguments row_handicaps, col_handicaps 
                and produces the payoff matrix.
                Each element of the payoff matrix should depend only on the corresponding row and column handicap.
                It is highly desirable that the function be strictly monotonically decreasing 
                in row_handicap and strictly monotonically increasing in col_handicap for every element. 
                NOTE: In this symmetric case the function should also have the property that 
                handicap_function(row_handicaps, col_handicaps) = -handicap_function(col_handicaps, row_handicaps) + value of the game
                where the value of the game is constant.
                This means that for any setting of the handicaps the payoff matrix is skew-symmetric plus the value of the game.
                In particular, all diagonal elements should be equal to the value of the game.
            strategy_weights: Defines the desired Nash equilibrium in terms of strategy probability weights. 
                If only an integer is specified, a uniform distribution will be used.
            row_derivative: A function that takes the arguments row_handicaps, col_handicaps
                and produces the derviative of the payoff matrix with respect to the row handicaps.
                The skew-symmetry property means that the column derivative is 
                the negative of the row derivative with the players interchanged.
            value: Value of the game. 
                If not supplied it will be set automatically based on the diagonal elements
                when the payoff matrix is first evaluated.
            fix_index: If set, this will fix one handicap at its starting value and ignore the corresponding payoff.
                This is useful if the handicap function is known to have a degree of invariance.
                If set to true, a strategy with nonzero weight will automatically be selected.
        """
        self.handicap_count, self.strategy_weights, self.strategy_objective_weights, nonzero_weight_index = _process_weights(strategy_weights)
        self.row_count = self.handicap_count
        self.col_count = self.handicap_count
        
        if row_derivative is not None:
            self.row_derivative = row_derivative
            # Using the skew-symmetric property.
            self.col_derivative = lambda row_handicaps, col_handicaps: -row_derivative(row_handicaps, col_handicaps)
            
        self.value = value
        
        if fix_index is True:
            # Select the first nonzero weight.
            fix_index = nonzero_weight_index
        elif fix_index is not None:
            if self.strategy_weights[fix_index] == 0.0:
                raise ValueError('fix_index %d corresponds to a strategy with zero weight.' % fix_index)
        
        self.fix_index = fix_index
   
    def evaluate_payoff_matrix(self, handicaps):
        """
        Evaluates the payoff matrix by calling handicap_function.
        Also sets self.value if not already set.
        """
        result = self.handicap_function(handicaps, handicaps)
        if self.value is None:
            self.value = numpy.average(numpy.diag(result), weights = self.strategy_weights)
        return result
        
    def objective(self, handicaps):
        """
        Compute the objective vector, which is desired to be zero. 
        This is the expected payoff of each strategy for that player, times the weight of that stategy.
        In order to balance them at the edge of being played, zero-weighted strategies are given a weight of 1.0. 
        This works since they do not affect the expected payoff of other strategies.
        """
        
        F = self.evaluate_payoff_matrix(handicaps)
        
        # Dot products are weighted.
        objectives = (numpy.tensordot(F, self.strategy_weights, axes = ([1], [0])) - self.value) * self.strategy_objective_weights
        
        return objectives
        
    def jacobian(self, handicaps):
        """ Compute the Jacobian of the objective using the provided row_derivative. """
        dFdr = self.row_derivative_combined(handicaps)
        
        # Derivative of row payoffs with respect to row handicaps.
        Jrr = numpy.tensordot(dFdr, self.strategy_weights, axes = ([1], [0])) * self.strategy_objective_weights
        Jrr = numpy.diag(Jrr)
        
        # Derivative of row payoffs with respect to col handicaps.
        dFdc = -numpy.transpose(dFdr)
        Jrc = dFdc * self.strategy_weights[None, :] * self.strategy_objective_weights[:, None]
        
        # Variables change both row and col handicaps at the same time, so Jacobian is the sum of their effects.
        J = Jrr + Jrc
        
        return J
        
    def optimize(self, x0 = None, check_derivative = False, check_jacobian = False, *args, **kwargs):
        """
        Compute the handicaps that balance the game using scipy.optimize.root.
        
        Args:
            x0: Starting point of the optimization. Defaults to a zero vector.
            check_derivative, check_jacobian:
                Can be used to check the provided row_derivative, col_derivative 
                against a finite difference approximation with the provided epsilon.
                A value of True uses a default value for epsilon.
            *args, **kwargs: Passed to scipy.optimize.root.
                In particular you may want to consider changing the solver method if the default is not producing good results.
        
        Returns:
            The result of scipy.optimize.root, with the following additional values:
                result.handicaps: The solved handicaps.
                result.F: The resulting payoff matrix.
        """
        result = self.optimize_common(x0 = x0, 
            check_derivative = check_derivative, check_jacobian = check_jacobian, 
            *args, **kwargs)
        
        return result
    
class MultiplicativeBalance(NonSymmetricBalance):
    """
    A special case where the handicap functions are col_handicap / row_handicap * initial_payoff.
    The actual optimization is done by mapping raw handicaps in (-inf, inf) to the actual handicaps (0, inf) using a rectifier.
    """
    
    def __init__(self, initial_payoff_matrix, row_weights = None, col_weights = None, 
        value = 1.0, fix_index = True, rectifier = zerosum.function.ReciprocalLinearRectifier()):
        """
        Args:
            initial_payoff_matrix: Should be nonnegative.
            value: Should be strictly positive. Note that the default is 1.0.
            rectifier: A strictly monotonically increasing function with range (0, inf).
                While an exponential is appealing from an analytic point of view, it can cause overflows in practice.
                We therefore default to a reciprocal-linear rectifier.
                
        Raises:
            ValueError: If the size of row_weights or col_weights do not match initial_payoff_matrix.
            ValueWarning: If initial_payoff_matrix has negative elements.
        """
        self.initial_payoff_matrix = initial_payoff_matrix
        if row_weights is None: row_weights = initial_payoff_matrix.shape[0]
        if col_weights is None: col_weights = initial_payoff_matrix.shape[1]
        
        if numpy.any(initial_payoff_matrix < 0.0):
            warnings.warn('initial_payoff_matrix has negative element(s).', ValueWarning)
            
        if value <= 0.0:
            warnings.warn('Value %f is non-positive.' % value, ValueWarning)
    
        NonSymmetricBalance.__init__(self, self.handicap_function, row_weights = row_weights, col_weights = col_weights, 
            row_derivative = self.row_derivative, col_derivative = self.col_derivative, 
            value = value, fix_index = fix_index)
            
        if initial_payoff_matrix.shape[0] != self.row_weights.size:
            raise ValueError('The size of row_weights does not match the row count of initial_payoff_matrix.')
        
        if initial_payoff_matrix.shape[1] != self.col_weights.size:
            raise ValueError('The size of col_weights does not match the column count of initial_payoff_matrix.')
            
        self.rectifier = rectifier

    def handicap_function(self, row_handicaps, col_handicaps):
        return self.initial_payoff_matrix * self.rectifier.evaluate(col_handicaps)[None, :] * self.rectifier.evaluate(-row_handicaps)[:, None]
        
    def row_derivative(self, row_handicaps, col_handicaps):
        return self.initial_payoff_matrix * self.rectifier.evaluate(col_handicaps)[None, :] * -self.rectifier.derivative(-row_handicaps)[:, None]
        
    def col_derivative(self, row_handicaps, col_handicaps):
        return self.initial_payoff_matrix * self.rectifier.derivative(col_handicaps)[None, :] * self.rectifier.evaluate(-row_handicaps)[:, None]
    
    def optimize(self, method = 'lm', *args, **kwargs):
        """
        Compute the handicaps that balance the game using scipy.optimize.root.
        
        Args:
            method: Used by scipy.optimize.root. 
                For this case we default to method 'lm' since it seems to produce more accurate results.
        
        Returns:
            The result of scipy.optimize.root, as with NonSymmetricBalance.
            Note that the actual optimization is done using handicaps in (-inf, inf)
            that are rectified before being used.
            These can be accessed using result.row_handicaps_pre_rectify, result.col_handicaps_pre_rectify.
        """
        
        result = NonSymmetricBalance.optimize(self, method = method, *args, **kwargs)
        result.row_handicaps_pre_rectify = result.row_handicaps
        result.col_handicaps_pre_rectify = result.col_handicaps
        result.row_handicaps = self.rectifier.evaluate(result.row_handicaps)
        result.col_handicaps = self.rectifier.evaluate(result.col_handicaps)
        return result
    
class LogisticSymmetricBalance(SymmetricBalance):
    """
    A special symmetric case where the handicap functions are logistic functions 
    whose argument is row_handicap - col_handicap + offset, 
    where offset is chosen so that when all handicaps are zero the initial_payoff_matrix is recovered.
    Commonly payoffs represent win rates.
    """
    def __init__(self, initial_payoff_matrix, strategy_weights = None, fix_index = True):
        """
        Args:
            initial_payoff_matrix: The elements of initial_payoff_matrix must be in (0, max_payoff), 
                where max_payoff is twice the value of the game.
                The initial_payoff_matrix should be skew-symmetric plus a constant offset (namely the value of the game).
                In particular, all diagonal elements should be equal to the value of the game.
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
        
        SymmetricBalance.__init__(self, self.handicap_function, strategy_weights, row_derivative = self.row_derivative, fix_index = fix_index)
        
        if initial_payoff_matrix.shape[0] != self.strategy_weights.size:
            raise ValueError('The size of strategy_weights does not match the dimensions of initial_payoff_matrix.')
        
        # The maximum possible payoff (e.g. 100% win rate) is twice the value of the game.
        self.max_payoff = 2.0 * initial_payoff_matrix[0, 0]
        
        # Check skew-symmetry. 
        initial_payoff_matrix_nt = self.max_payoff - initial_payoff_matrix.transpose()
        if not numpy.allclose(initial_payoff_matrix, initial_payoff_matrix_nt):
            warnings.warn('initial_payoff_matrix is not (close to) skew-symmetric plus a constant offset.', ValueWarning)
            
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