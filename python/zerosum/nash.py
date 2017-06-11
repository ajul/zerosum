import numpy
import scipy.optimize

def nash_row(payoff_matrix, *args, **kwargs):
    # Nash equilibrium for the row player (maximizer).
    row_count, col_count = payoff_matrix.shape
    
    # Variables: Maximizer strategy weights, value of the game.
    
    # Objective: Maximize the minimum possible row player's payoff.
    c = numpy.zeros((row_count + 1))
    c[-1] = -1.0 # SciPy uses the minimization convention.
    
    # Payoff when column player plays any strategy must be at least the value of the game.
    A_ub = numpy.concatenate((-payoff_matrix.transpose(), numpy.ones((col_count, 1))), axis = 1)
    b_ub = numpy.zeros(col_count)
    
    # Probabilities must add to 1.
    A_eq = numpy.ones((1, row_count + 1))
    A_eq[0, -1] = 0
    
    b_eq = numpy.ones((1, 1))
    
    # Weights must be nonnegative.
    bounds = [(0, None)] * row_count + [(None, None)]
    
    result = scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, *args, **kwargs)
    
    result.strategy = result.x[:-1]
    result.value = result.x[-1]
    
    return result

def nash(payoff_matrix):
    row_result = nash_row(payoff_matrix)
    col_result = nash_row(-payoff_matrix.transpose())
    return row_result, col_result