import numpy
import scipy.optimize

def nash(payoff_matrix, *args, **kwargs):
    # Returns the scipy.optimize.linprog result of finding the Nash equilbrium for a zero-sum game.
    
    # payoff_matrix is a 2-D array describing the payoffs for the row player (maximizer).
    # *args, **kwargs are passed to scipy.optimize.linprog.
    
    # Each result.strategy is the weights of a mixed Nash equilibrium.
    # Each result.value is the expected payoff for that player, i.e. row_result.value = -col_result.value (to machine precision)
    row_result = nash_row(payoff_matrix, *args, **kwargs)
    col_result = nash_row(-payoff_matrix.transpose(), *args, **kwargs)
    return row_result, col_result

def nash_row(payoff_matrix, *args, **kwargs):
    # Nash equilibrium for the row player (maximizer).
    row_count, col_count = payoff_matrix.shape
    
    # Variables: Row strategy weights, value of the game.
    
    # Objective: Maximize the minimum possible row player's payoff.
    c = numpy.zeros((row_count + 1))
    c[-1] = -1.0 # SciPy uses the minimization convention.
    
    # Payoff when column player plays any strategy must be at least the value of the game.
    A_ub = numpy.concatenate((-payoff_matrix.transpose(), numpy.ones((col_count, 1))), axis = 1)
    b_ub = numpy.zeros(col_count)
    
    # Probabilities must sum to 1.
    A_eq = numpy.ones((1, row_count + 1))
    A_eq[0, -1] = 0
    
    b_eq = numpy.ones((1))
    
    # Weights must be nonnegative. Payoff must be between the minimum and maximum value in the payoff matrix.
    min_payoff = numpy.min(payoff_matrix)
    max_payoff = numpy.max(payoff_matrix)
    bounds = [(0, None)] * row_count + [(min_payoff, max_payoff)]
    
    result = scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, *args, **kwargs)
    
    result.strategy = result.x[:-1]
    result.value = result.x[-1]
    
    return result
