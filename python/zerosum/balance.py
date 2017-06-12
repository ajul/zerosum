import numpy
import scipy.optimize

def _process_weights(arg):
    try:
        count = arg.size
        weights = arg
    except:
        count = arg
        weights = numpy.ones((arg)) / count
    
    # replace zeros with ones for purposes of weighting the objective vector
    objective_weights = weights
    objective_weights[objective_weights == 0.0] = 1.0
    
    return count, weights, objective_weights
    
class Balance():
    def jacobian_fd(self, epsilon = None):
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
        if x is None: x = numpy.zeros(self.x_count)
        J = self.jacobian(x)
        jac_fd = self.jacobian_fd(epsilon = epsilon)
        result = J - jac_fd(x)
        print('Maximum difference between evaluated Jacobian and finite difference:', numpy.max(numpy.abs(result)))
        return result
        
    def check_row_derivative(self, x = None, epsilon = None):
        if x is None: x = numpy.zeros(self.x_count)
        result = self.row_derivative_matrix(x) - self.row_derivative_matrix_fd(x, epsilon)
        print('Maximum difference between evaluated row_derivative and finite difference:', numpy.max(numpy.abs(result)))
        return result
    
    def check_col_derivative(self, x = None, epsilon = None):
        if x is None: x = numpy.zeros(self.x_count)
        result = self.col_derivative_matrix(x) - self.col_derivative_matrix_fd(x, epsilon)
        print('Maximum difference between evaluated col_derivative and finite difference:', numpy.max(numpy.abs(result)))
        return result
        
    def evaluate_F(self, row_handicaps, col_handicaps):
        F = numpy.zeros((self.row_count, self.col_count))
        
        for row_index in range(self.row_count):
            for col_index in range(self.col_count):
                F[row_index, col_index] = self.handicap_function(row_index, col_index, row_handicaps[row_index], col_handicaps[col_index])
                
        return F
        
    def row_derivative_matrix_fd(self, x, epsilon = None):
        if epsilon is None: epsilon = numpy.sqrt(numpy.finfo(float).eps)
        row_handicaps_N = x[:self.row_count] - epsilon * 0.5
        row_handicaps_P = x[:self.row_count] + epsilon * 0.5
        col_handicaps = x[-self.col_count:]
        return (self.evaluate_F(row_handicaps_P, col_handicaps) - self.evaluate_F(row_handicaps_N, col_handicaps)) / epsilon
        
    def col_derivative_matrix_fd(self, x, epsilon = None):
        if epsilon is None: epsilon = numpy.sqrt(numpy.finfo(float).eps)
        row_handicaps = x[:self.row_count]
        col_handicaps_N = x[-self.col_count:] - epsilon * 0.5
        col_handicaps_P = x[-self.col_count:] + epsilon * 0.5
        return (self.evaluate_F(row_handicaps, col_handicaps_P) - self.evaluate_F(row_handicaps, col_handicaps_N)) / epsilon
        
    def row_derivative_matrix(self, x):
        row_handicaps = x[:self.row_count]
        col_handicaps = x[-self.col_count:]
        
        dFdr = numpy.zeros((self.row_count, self.col_count))
        
        for row_index in range(self.row_count):
            for col_index in range(self.col_count):
                dFdr[row_index, col_index] = self.row_derivative(row_index, col_index, row_handicaps[row_index], col_handicaps[col_index])
        
        return dFdr
        
    def col_derivative_matrix(self, x):
        row_handicaps = x[:self.row_count]
        col_handicaps = x[-self.col_count:]
        
        dFdc = numpy.zeros((self.row_count, self.col_count))
        
        for row_index in range(self.row_count):
            for col_index in range(self.col_count):
                dFdc[row_index, col_index] = self.col_derivative(row_index, col_index, row_handicaps[row_index], col_handicaps[col_index])
        
        return dFdc

class NonSymmetricBalance(Balance):
    def __init__(self, handicap_function, row_weights, col_weights = None, row_derivative = None, col_derivative = None):
        self.handicap_function = handicap_function
        
        if (row_derivative is None) != (col_derivative is None):
            raise ValueError('Both row_derivative and col_derivative must be provided for Jacobian to function.')
        
        self.row_derivative = row_derivative
        self.col_derivative = col_derivative
    
        self.row_count, self.row_weights, self.rowobjective_weights = _process_weights(row_weights)
        self.col_count, self.col_weights, self.colobjective_weights = _process_weights(col_weights)
        
        self.x_count = self.row_count + self.col_count
        
    def evaluate_Fx(self, x):
        return self.evaluate_F(x[:self.row_count], x[-self.col_count:])
        
    def objective(self, x):
        F = self.evaluate_Fx(x)
        
        # dot products are weighted 
        row_objectives = numpy.tensordot(F, self.col_weights, axes = ([1], [0])) * self.rowobjective_weights
        col_objectives = numpy.tensordot(F, self.row_weights, axes = ([0], [0])) * self.colobjective_weights
        
        return numpy.concatenate((row_objectives, col_objectives))
        
    def jacobian(self, x):
        # J_ij = derivative of payoff i with respect to handicap j
        
        dFdr = self.row_derivative_matrix(x)
        dFdc = self.col_derivative_matrix(x)
        
        # derivative of row payoffs with respect to row handicaps
        Jrr = numpy.tensordot(dFdr, self.col_weights, axes = ([1], [0])) * self.rowobjective_weights
        Jrr = numpy.diag(Jrr)
        
        # derivative of col payoffs with respect to col handicaps
        Jcc = numpy.tensordot(dFdc, self.row_weights, axes = ([0], [0])) * self.colobjective_weights
        Jcc = numpy.diag(Jcc)
        
        # derivative of row payoffs with respect to col handicaps
        Jrc = dFdc * self.col_weights[None, :] * self.rowobjective_weights[:, None]
        
        # derivative of col payoffs with respect to row handicaps
        Jcr = dFdr * self.row_weights[:, None] * self.colobjective_weights[None, :]
        Jcr = numpy.transpose(Jcr)
        
        # assemble full Jacobian
        J = numpy.bmat([[Jrr, Jrc],
                        [Jcr, Jcc]])
        
        return J
        
    def optimize(self, check_derivative_epsilon = False, check_jacobian_epsilon = False, *args, **kwargs):
        if self.row_derivative is None or self.col_derivative is None:
            jac = None
        else:
            jac = self.jacobian
            
        def fun(x):
            if check_derivative_epsilon is not False:
                self.check_row_derivative(x, epsilon = check_derivative_epsilon)
                self.check_col_derivative(x, epsilon = check_derivative_epsilon)
            if check_jacobian_epsilon is not False: self.check_jacobian(x, epsilon = check_jacobian_epsilon)
            return self.objective(x)   
        
        x0 = numpy.zeros((self.x_count))
        result = scipy.optimize.root(fun = fun, x0 = x0, jac = jac, *args, **kwargs)
        result.row_handicaps = result.x[:self.row_count]
        result.col_handicaps = result.x[-self.col_count:]
        result.F = self.evaluate_Fx(result.x)
        
        return result

class SymmetricBalance(Balance):
    def __init__(self, handicap_function, strategy_weights, row_derivative = None):
        self.x_count, self.strategy_weights, self.strategy_objective_weights = _process_weights(strategy_weights)
        self.row_count = self.x_count
        self.col_count = self.x_count
        
        if row_derivative is None:
            self.col_derivative = None
   
    def col_derivative(self, row_index, col_index, row_handicap, col_handicap): 
        return -self.row_derivative(col_index, row_index, col_handicap, row_handicap)
        
    def evaluate_Fx(self, x):
        F = numpy.zeros((self.x_count, self.x_count))
        
        for row_index in range(self.x_count-1):
            for col_index in range(row_index+1, self.x_count):
                payoff = self.handicap_function(row_index, col_index, x[row_index], x[col_index])
                F[row_index, col_index] = payoff
                F[col_index, row_index] = -payoff
                
        return F
        
    def objective(self, x):
        F = self.evaluate_Fx(x)
        
        # dot products are weighted 
        objectives = numpy.tensordot(F, self.strategy_weights, axes = ([1], [0])) * self.strategy_objective_weights
        
        return objectives
        
    def jacobian(self, x):
        dFdr = self.row_derivative_matrix(x)
        
        # derivative of row payoffs with respect to row handicaps
        Jrr = numpy.tensordot(dFdr, self.strategy_weights, axes = ([1], [0])) * self.strategy_objective_weights
        Jrr = numpy.diag(Jrr)
        
        # derivative of row payoffs with respect to col handicaps
        dFdc = -numpy.transpose(dFdr)
        Jrc = dFdc * self.strategy_weights[None, :] * self.strategy_objective_weights[:, None]
        
        J = Jrr + Jrc
        
        return J
        
    def optimize(self, check_derivative_epsilon = False, check_jacobian_epsilon = False, *args, **kwargs):
        if self.row_derivative is None:
            jac = None
        else:
            jac = self.jacobian
            
        def fun(x):
            if check_derivative_epsilon is not False:
                self.check_row_derivative(x, epsilon = check_derivative_epsilon)
                self.check_col_derivative(x, epsilon = check_derivative_epsilon)
            if check_jacobian_epsilon is not False: self.check_jacobian(x, epsilon = check_jacobian_epsilon)
            return self.objective(x)      
            
        x0 = numpy.zeros((self.x_count))
        result = scipy.optimize.root(fun = fun, x0 = x0, jac = jac, *args, **kwargs)
        result.handicaps = result.x
        result.F = self.evaluate_Fx(result.x)
        return result
    
class MultiplicativeBalance(NonSymmetricBalance):
    def __init__(self, initial_payoff_matrix, row_weights = None, col_weights = None):
        self.initial_payoff_matrix = initial_payoff_matrix
        if row_weights is None: row_weights = initial_payoff_matrix.shape[0]
        if col_weights is None: col_weights = initial_payoff_matrix.shape[1]
    
        NonSymmetricBalance.__init__(self, self.handicap_function, row_weights = row_weights, col_weights = col_weights, row_derivative = self.row_derivative, col_derivative = self.col_derivative)

    def handicap_function(self, row_index, col_index, row_handicap, col_handicap):
        return self.initial_payoff_matrix[row_index, col_index] * numpy.exp(col_handicap - row_handicap) - 1.0
        
    def row_derivative(self, row_index, col_index, row_handicap, col_handicap):
        return -self.initial_payoff_matrix[row_index, col_index] * numpy.exp(col_handicap - row_handicap)
        
    def col_derivative(self, row_index, col_index, row_handicap, col_handicap):
        return self.initial_payoff_matrix[row_index, col_index] * numpy.exp(col_handicap - row_handicap)
    
    def optimize(self, *args, **kwargs):
        result = NonSymmetricBalance.optimize(self, *args, **kwargs)
        result.row_log_handicaps = result.row_handicaps
        result.col_log_handicaps = result.col_handicaps
        result.row_handicaps = numpy.exp(result.row_handicaps)
        result.col_handicaps = numpy.exp(result.col_handicaps)
        return result
    
class LogisticSymmetricBalance(SymmetricBalance):
    def __init__(self, initial_payoff_matrix, strategy_weights = None):
        self.offset_matrix = numpy.log(1.0 / initial_payoff_matrix - 1.0)
        if strategy_weights is None: strategy_weights = initial_payoff_matrix.shape[0]
        SymmetricBalance.__init__(self, self.handicap_function, strategy_weights, row_derivative = self.row_derivative)
    
    def handicap_function(self, row_index, col_index, row_handicap, col_handicap):
        offset = self.offset_matrix[row_index, col_index]
        return 1.0 / (1.0 + numpy.exp(row_handicap - col_handicap + offset)) - 0.5
        
    def row_derivative(self, row_index, col_index, row_handicap, col_handicap):
        payoff = self.handicap_function(row_index, col_index, row_handicap, col_handicap)
        return payoff * payoff - 0.25
        
    def optimize(self, *args, **kwargs):
        result = SymmetricBalance.optimize(self, *args, **kwargs)
        result.F = result.F + 0.5
        return result