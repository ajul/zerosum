from zerosum.function import HarmonicLinearRectifier as HLR
import numpy
import scipy.optimize
import warnings

class DerivativeWarning(RuntimeWarning):
    pass
    
class ValueWarning(RuntimeWarning):
    pass

def _process_weights(arg):
    try:
        weights = arg.copy()
        count = weights.size
    except:
        weights = numpy.ones((arg)) / arg
        count = arg
        
    weight_sum = numpy.sum(weights)
    
    if weight_sum == 0:
        raise ValueError('Weights sum to 0.')
        
    if numpy.any(weights < 0.0):
        raise ValueError('Received negative weight(s).')
        
    weights = weights / weight_sum
    
    # Replace zeros with 1.0 / weights.size for purposes of weighting the objective vector.
    objective_weights = weights.copy()
    objective_weights[objective_weights == 0.0] = 1.0 / weights.size
    
    return count, weights, objective_weights
    
class Balance():
    # Base class for balancing.
    
    def evaluate_F(self, row_handicaps, col_handicaps):
        # Evaluates the payoff matrix F for a given set of handicap variable vectors.
        F = numpy.zeros((self.row_count, self.col_count))
        
        for row_index in range(self.row_count):
            for col_index in range(self.col_count):
                F[row_index, col_index] = self.handicap_function(row_index, col_index, row_handicaps[row_index], col_handicaps[col_index])
                
        return F
    
    def jacobian_fd(self, epsilon = None):
        # Computes a finite (central) difference approximation of the Jacobian.
        if epsilon is None: epsilon = numpy.sqrt(numpy.finfo(float).eps)
        def result(x):
            J = numpy.zeros((self.x_count, self.x_count))
            
            for input_index in range(self.x_count):
                xdp = x.copy()
                xdp[input_index] += epsilon * 0.5
                
                xdn = x.copy()
                xdn[input_index] -= epsilon * 0.5
                
                J[:, input_index] = (self.objective(xdp) - self.objective(xdn)) / epsilon
            
            return J
        return result

    def check_jacobian(self, x = None, epsilon = None):
        # Checks the Jacobian computed from the handicap function derivatives against a finite difference approximation.
        if x is None: x = numpy.zeros(self.x_count)
        J = self.jacobian(x)
        jac_fd = self.jacobian_fd(epsilon = epsilon)
        result = J - jac_fd(x)
        print('Maximum difference between evaluated Jacobian and finite difference:', numpy.max(numpy.abs(result)))
        return result
        
    def row_derivative_matrix(self, x):
        # Computes a matrix consisting of the derivative of the handicap function with respect to the corresponding row handicap.
        row_handicaps = x[:self.row_count]
        col_handicaps = x[-self.col_count:]
        
        dFdr = numpy.zeros((self.row_count, self.col_count))
        
        for row_index in range(self.row_count):
            for col_index in range(self.col_count):
                dFdr[row_index, col_index] = self.row_derivative(row_index, col_index, row_handicaps[row_index], col_handicaps[col_index])
        
        return dFdr
        
    def col_derivative_matrix(self, x):
        # Computes a matrix consisting of the derivative of the handicap function with respect to the corresponding column handicap.
        row_handicaps = x[:self.row_count]
        col_handicaps = x[-self.col_count:]
        
        dFdc = numpy.zeros((self.row_count, self.col_count))
        
        for row_index in range(self.row_count):
            for col_index in range(self.col_count):
                dFdc[row_index, col_index] = self.col_derivative(row_index, col_index, row_handicaps[row_index], col_handicaps[col_index])
        
        return dFdc
        
    def row_derivative_matrix_fd(self, x, epsilon = None):
        # Computes a finite (central) difference approximation of derivative of the handicap function with respect to the corresponding row handicap.
        if epsilon is None: epsilon = numpy.sqrt(numpy.finfo(float).eps)
        row_handicaps_N = x[:self.row_count] - epsilon * 0.5
        row_handicaps_P = x[:self.row_count] + epsilon * 0.5
        col_handicaps = x[-self.col_count:]
        return (self.evaluate_F(row_handicaps_P, col_handicaps) - self.evaluate_F(row_handicaps_N, col_handicaps)) / epsilon
        
    def col_derivative_matrix_fd(self, x, epsilon = None):
        # Computes a finite (central) difference approximation of derivative of the handicap function with respect to the corresponding column handicap.
        if epsilon is None: epsilon = numpy.sqrt(numpy.finfo(float).eps)
        row_handicaps = x[:self.row_count]
        col_handicaps_N = x[-self.col_count:] - epsilon * 0.5
        col_handicaps_P = x[-self.col_count:] + epsilon * 0.5
        return (self.evaluate_F(row_handicaps, col_handicaps_P) - self.evaluate_F(row_handicaps, col_handicaps_N)) / epsilon
        
    def check_row_derivative(self, x = None, epsilon = None):
        # Checks the derivative of the handicap function with respect to the corresponding row handicap against a finite difference approximation.
        # Also checks that all row derivatives are negative.
        if x is None: x = numpy.zeros(self.x_count)
        direct = self.row_derivative_matrix(x)
        fd = self.row_derivative_matrix_fd(x, epsilon)
        if numpy.any(direct >= 0.0) or numpy.any(fd >= 0.0):
            msg = 'Found a non-negative row derivative for\nx = %s.' % x
            msg += '\nIt is highly desirable that the handicap function be strictly monotonically decreasing in the row handicap.'
            warnings.warn(msg, DerivativeWarning)
        result = direct - fd
        print('Maximum difference between evaluated row_derivative and finite difference:', numpy.max(numpy.abs(result)))
        return result
    
    def check_col_derivative(self, x = None, epsilon = None):
        # Checks the derivative of the handicap function with respect to the corresponding column handicap against a finite difference approximation.
        # Also checks that all column derivatives are negative.
        if x is None: x = numpy.zeros(self.x_count)
        direct = self.col_derivative_matrix(x)
        fd = self.col_derivative_matrix_fd(x, epsilon)
        if numpy.any(direct <= 0.0) or numpy.any(fd <= 0.0):
            msg = 'Found a non-positive column derivative for\nx = %s.' % x
            msg += '\nIt is highly desirable that the handicap function be strictly monotonically increasing in the column handicap.'
            warnings.warn(msg, DerivativeWarning)
        result = direct - fd
        print('Maximum difference between evaluated col_derivative and finite difference:', numpy.max(numpy.abs(result)))
        return result

class NonSymmetricBalance(Balance):
    def __init__(self, handicap_function, row_weights, col_weights, row_derivative = None, col_derivative = None, value = 0.0):
        # handicap_function: A function that takes the arguments row_index, col_index, row_handicap, col_handicap 
        #     and produces the (row_index, col_index) element of the payoff matrix. 
        #     It is highly desirable that the function be strictly monotonically decreasing in row_handicap and strictly monotonically increasing in col_derivative for every element. 
        # row_weights, col_weights: Defines the desired Nash equilibrium in terms of row and column strategy probability weights. 
        #     If only an integer is specified, a uniform distribution will be used.
        #     Weights will be normalized.
        # row_derivative, col_derivative: Functions that take the arguments row_index, col_index, row_handicap, col_handicap 
        #     and produce the derviative of the (row_index, col_index) element of the payoff matrix with respect to the row or column handicap.
        # value: The desired value of the resulting game. This is equal to the row player's payoff and the negative of the column player's payoff.
        self.handicap_function = handicap_function
        
        if (row_derivative is None) != (col_derivative is None):
            raise ValueError('Both row_derivative and col_derivative must be provided for the Jacobian to function.')
        
        self.row_derivative = row_derivative
        self.col_derivative = col_derivative
    
        self.row_count, self.row_weights, self.row_objective_weights = _process_weights(row_weights)
        self.col_count, self.col_weights, self.col_objective_weights = _process_weights(col_weights)
        
        self.x_count = self.row_count + self.col_count
        
        if value <= 0.0:
            warnings.warn('Value %f is non-positive.' % value, ValueWarning)
        
        self.value = value
        
    def evaluate_Fx(self, x):
        # Evaluate F in terms of the variables, namely the handicap variable vectors.
        return self.evaluate_F(x[:self.row_count], x[-self.col_count:])
        
    def objective(self, x):
        # Compute the objective vector, which is desired to be zero. This is the expected payoff of each strategy for that player, times the weight of that stategy.
        # In order to balance them at the edge of being played, zero-weighted strategies are given a weight of 1.0. This works since they do not affect the expected payoff of other strategies.
    
        F = self.evaluate_Fx(x)
        
        # Dot products are weighted.
        row_objectives = (numpy.tensordot(F, self.col_weights, axes = ([1], [0])) - self.value) * self.row_objective_weights
        col_objectives = (self.value - numpy.tensordot(F, self.row_weights, axes = ([0], [0]))) * self.col_objective_weights
        
        return numpy.concatenate((row_objectives, col_objectives))
        
    def jacobian(self, x):
        # Compute the Jacobian of the objective using the provided row_derivative, col_derivative.
        
        # J_ij = derivative of payoff i with respect to handicap j.
        
        dFdr = self.row_derivative_matrix(x)
        dFdc = self.col_derivative_matrix(x)
        
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
        
    def optimize(self, x0 = None, check_derivative_epsilon = False, check_jacobian_epsilon = False, *args, **kwargs):
        # Compute the handicaps that balance the game using scipy.optimize.root.
        # check_derivative_epsilon, check_jacobian_epsilon can be used to check the provided row_derivative, col_derivative against a finite difference approximation.
        #     A value of None uses a default value.
        # *args, **kwargs are passed to using scipy.optimize.root.
        #     In particular you may want to consider changing the solver method if the default is not producing good results.
        
        # Returns the result of scipy.optimize.root, with the following additional values:
        #     result.row_handicaps: The solved row handicaps.
        #     result.col_handicaps: The solved column handicaps.
        #     result.F: The resulting payoff matrix.
        if self.row_derivative is None or self.col_derivative is None:
            jac = None
        else:
            jac = self.jacobian
            
        def fun(x):
            if check_derivative_epsilon is not False:
                self.check_row_derivative(x, epsilon = check_derivative_epsilon)
                self.check_col_derivative(x, epsilon = check_derivative_epsilon)
            if check_jacobian_epsilon is not False: 
                self.check_jacobian(x, epsilon = check_jacobian_epsilon)
            return self.objective(x)   
        
        if x0 is None:
            x0 = numpy.zeros((self.x_count))
        result = scipy.optimize.root(fun = fun, x0 = x0, jac = jac, *args, **kwargs)
        result.row_handicaps = result.x[:self.row_count]
        result.col_handicaps = result.x[-self.col_count:]
        result.F = self.evaluate_Fx(result.x)
        
        return result

class SymmetricBalance(Balance):
    def __init__(self, handicap_function, strategy_weights, row_derivative = None):
        # This version is for symmetric games, where both players are choosing from the same set of strategies.
        # Thus there are no independent inputs for column strategies.
        
        # handicap_function: A function that takes the arguments row_index, col_index, row_handicap, col_handicap 
        #     and produces the (row_index, col_index) element of the payoff matrix. 
        #     It is highly desirable that the function be strictly monotonically decreasing in row_handicap and strictly monotonically increasing in col_derivative for every element. 
        #     NOTE: In this symmetric case the function should also have the property that 
        #     handicap_function(row_index, col_index, row_handicap, col_handicap) = -handicap_function(col_index, row_index, col_handicap, row_handicap)
        #     This means that for any setting of the handicaps the payoff matrix is skew-symmetric.
        #     In particular, all diagonal elements should be equal to 0.
        # strategy_weights: Defines the desired Nash equilibrium in terms of strategy probability weights. 
        #     If only an integer is specified, a uniform distribution will be used.
        # row_derivative: A function that takes the arguments row_index, col_index, row_handicap, col_handicap 
        #     and produces the derviative of the (row_index, col_index) element of the payoff matrix with respect to the row handicap.
        #     The skew-symmetry property means that the column derivative is the negative of the row derivative with the players interchanged.
        self.x_count, self.strategy_weights, self.strategy_objective_weights = _process_weights(strategy_weights)
        self.row_count = self.x_count
        self.col_count = self.x_count
        
        if row_derivative is not None:
            self.row_derivative = row_derivative
            # Using the skew-symmetric property.
            self.col_derivative = lambda row_index, col_index, row_handicap, col_handicap: -row_derivative(col_index, row_index, col_handicap, row_handicap)
   
    def evaluate_Fx(self, x):
        # Evaluate F in terms of the variables, namely the shared handicap variable vector. This uses the fact that the matrix is skew-symmetric.
        F = numpy.zeros((self.x_count, self.x_count))
        
        for row_index in range(self.x_count - 1):
            for col_index in range(row_index + 1, self.x_count):
                payoff = self.handicap_function(row_index, col_index, x[row_index], x[col_index])
                F[row_index, col_index] = payoff
                F[col_index, row_index] = -payoff
                
        return F
        
    def objective(self, x):
        # Compute the objective vector, which is desired to be zero. This is the expected payoff of each strategy for that player, times the weight of that stategy.
        # In order to balance them at the edge of being played, zero-weighted strategies are given a weight of 1.0. This works since they do not affect the expected payoff of other strategies.
        
        F = self.evaluate_Fx(x)
        
        # Dot products are weighted .
        objectives = numpy.tensordot(F, self.strategy_weights, axes = ([1], [0])) * self.strategy_objective_weights
        
        return objectives
        
    def jacobian(self, x):
        # Compute the Jacobian of the objective using the provided row_derivative.
        dFdr = self.row_derivative_matrix(x)
        
        # Derivative of row payoffs with respect to row handicaps.
        Jrr = numpy.tensordot(dFdr, self.strategy_weights, axes = ([1], [0])) * self.strategy_objective_weights
        Jrr = numpy.diag(Jrr)
        
        # Derivative of row payoffs with respect to col handicaps.
        dFdc = -numpy.transpose(dFdr)
        Jrc = dFdc * self.strategy_weights[None, :] * self.strategy_objective_weights[:, None]
        
        # Variables change both row and col handicaps at the same time, so Jacobian is the sum of their effects.
        J = Jrr + Jrc
        
        return J
        
    def optimize(self, x0 = None, check_derivative_epsilon = False, check_jacobian_epsilon = False, *args, **kwargs):
        # Compute the handicaps that balance the game using scipy.optimize.root.
        # check_derivative_epsilon, check_jacobian_epsilon can be used to check the provided row_derivative, col_derivative against a finite difference approximation.
        #     A value of None uses a default value.
        # *args, **kwargs are passed to using scipy.optimize.root.
        #     In particular you may want to consider changing the solver method if the default is not producing good results.
        
        # Returns the result of scipy.optimize.root, with the following additional values:
        #     result.handicaps: The solved handicaps.
        #     result.F: The resulting payoff matrix.
        
        if self.row_derivative is None:
            jac = None
        else:
            jac = self.jacobian
            
        def fun(x):
            if check_derivative_epsilon is not False:
                self.check_row_derivative(x, epsilon = check_derivative_epsilon)
                self.check_col_derivative(x, epsilon = check_derivative_epsilon)
            if check_jacobian_epsilon is not False: 
                self.check_jacobian(x, epsilon = check_jacobian_epsilon)
            return self.objective(x)
            
        if x0 is None:
            x0 = numpy.zeros((self.x_count))
        result = scipy.optimize.root(fun = fun, x0 = x0, jac = jac, *args, **kwargs)
        result.handicaps = result.x
        result.F = self.evaluate_Fx(result.x)
        return result
    
class MultiplicativeBalance(NonSymmetricBalance):
    # A special case where the handicap functions are col_handicap / row_handicap * initial_payoff.
    # The actual optimization is done using the log of the handicaps.
    
    def __init__(self, initial_payoff_matrix, row_weights = None, col_weights = None, value = 1.0):
        # initial_payoff_matrix: Should be nonnegative.
        # value: Should be strictly positive. Note that the default is 1.0.
        self.initial_payoff_matrix = initial_payoff_matrix
        if row_weights is None: row_weights = initial_payoff_matrix.shape[0]
        if col_weights is None: col_weights = initial_payoff_matrix.shape[1]
        
        if numpy.any(initial_payoff_matrix < 0.0):
            warnings.warn('initial_payoff_matrix has negative element(s).', ValueWarning)
    
        NonSymmetricBalance.__init__(self, self.handicap_function, row_weights = row_weights, col_weights = col_weights, 
            row_derivative = self.row_derivative, col_derivative = self.col_derivative, 
            value = value)

    def handicap_function(self, row_index, col_index, row_handicap, col_handicap):
        return self.initial_payoff_matrix[row_index, col_index] * HLR.evaluate(col_handicap) * HLR.evaluate(-row_handicap)
        
    def row_derivative(self, row_index, col_index, row_handicap, col_handicap):
        return self.initial_payoff_matrix[row_index, col_index] * HLR.evaluate(col_handicap) * -HLR.derivative(-row_handicap)
        
    def col_derivative(self, row_index, col_index, row_handicap, col_handicap):
        return self.initial_payoff_matrix[row_index, col_index] * HLR.derivative(col_handicap) * HLR.evaluate(-row_handicap)
    
    def optimize(self, method = 'lm', *args, **kwargs):
        # The actual optimization is done using handicaps in (-inf, inf) that are rectified before being used.
        # These can be accessed using result.row_pre_rect_handicaps, result.col_pre_rect_handicaps.
        
        # We default to method 'lm' since it seems to tend to be more accurate in this case.
        
        result = NonSymmetricBalance.optimize(self, method = method, *args, **kwargs)
        result.row_pre_rect_handicaps = result.row_handicaps
        result.col_pre_rect_handicaps = result.col_handicaps
        result.row_handicaps = HLR.evaluate(result.row_handicaps)
        result.col_handicaps = HLR.evaluate(result.col_handicaps)
        return result
    
class LogisticSymmetricBalance(SymmetricBalance):
    # A special symmetric case where the handicap functions are logistic functions who argument is row_handicap - col_handicap + offset, 
    # where offset is chosen so that when all handicaps are zero the initial_payoff_matrix is recovered.
    # Commonly payoffs represent win rates.
    def __init__(self, initial_payoff_matrix, strategy_weights = None):
        # initial_payoff_matrix 
        #     The elements of initial_payoff_matrix must be in (0, max_payoff), where max_payoff is twice the value of the game.
        #     The initial_payoff_matrix should be skew-symmetric plus a constant offset (namely the value of the game).
        #     In particular, all diagonal elements should be equal to the value of the game.
        if strategy_weights is None: strategy_weights = initial_payoff_matrix.shape[0]
        SymmetricBalance.__init__(self, self.handicap_function, strategy_weights, row_derivative = self.row_derivative)
        
        # The maximum possible payoff (e.g. 100% win rate) is twice the value of the game.
        self.max_payoff = 2.0 * initial_payoff_matrix[0, 0]
        
        # Check skew-symmetry. 
        initial_payoff_matrix_nt = self.max_payoff - initial_payoff_matrix.transpose()
        if not numpy.allclose(initial_payoff_matrix, initial_payoff_matrix_nt):
            warnings.warn('initial_payoff_matrix is not skew-symmetric plus a constant offset.', ValueWarning)
            
        # Check bounds.
        if numpy.any(initial_payoff_matrix <= 0.0) or numpy.any(initial_payoff_matrix >= self.max_payoff):
            raise ValueError('initial_payoff_matrix has element(s) not in the open interval (0, max_payoff), where max_payoff = %f is twice the value of the game.' % self.max_payoff)
        if numpy.any(numpy.isclose(initial_payoff_matrix, 0.0)) or numpy.any(numpy.isclose(initial_payoff_matrix, self.max_payoff)):
            warnings.warn('initial_payoff_matrix has element(s) close to 0 and/or max_payoff, where max_payoff = %f is twice the value of the game.' % self.max_payoff, ValueWarning)
            
        self.initial_payoff_matrix = initial_payoff_matrix
        self.initial_offset_matrix = numpy.log(self.max_payoff / initial_payoff_matrix - 1.0)
    
    def handicap_function(self, row_index, col_index, row_handicap, col_handicap):
        # Normalized to the range (-0.5, 0.5).
        offset = self.initial_offset_matrix[row_index, col_index]
        return 1.0 / (1.0 + numpy.exp(row_handicap - col_handicap + offset)) - 0.5
        
    def row_derivative(self, row_index, col_index, row_handicap, col_handicap):
        normalized_payoff = self.handicap_function(row_index, col_index, row_handicap, col_handicap)
        return normalized_payoff * normalized_payoff - 0.25
        
    def optimize(self, *args, **kwargs):
        result = SymmetricBalance.optimize(self, *args, **kwargs)
        # Expand F back to the original range (0, max_payoff).
        result.F = (result.F + 0.5) * self.max_payoff
        return result