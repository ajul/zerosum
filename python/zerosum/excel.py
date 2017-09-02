"""
Excel integration. Requires xlwings.

WARNING: Return values are dynamic arrays and will overwrite worksheet values without warning!
"""

import numpy
import xlwings
import zerosum.balance
import zerosum.nash

@xlwings.func
@xlwings.arg('payoff_matrix', numpy.array, ndim=2)
@xlwings.ret(expand='table')
def nash_weights(payoff_matrix):
    """
    Computes Nash equilbrium weights.
    Results are put into two columns, 
    the first corresponding to the row player, the second to the column player.
    """
    row_result, col_result = zerosum.nash.nash(payoff_matrix)
    return numpy.stack((row_result.strategy, col_result.strategy), axis = 1)

@xlwings.func
@xlwings.arg('payoff_matrix', numpy.array, ndim=2)
def nash_value(payoff_matrix):
    """
    Computes value of a payoff_matrix.
    """
    row_result, _ = zerosum.nash.nash_row(payoff_matrix)
    return row_result.value

def compute_balance_result(balance_class, *args, **kwargs):
    """
    Common result format:
           | m cols          1 col
    --------------------------------------
    n rows | payoff_matrix   row_handicaps
    1 row  | col_handicaps   blank
    """
    balance = balance_class(*args, **kwargs)
    result = balance.optimize()
    
    col_handicaps = numpy.reshape(result.col_handicaps, (1, -1))
    row_handicaps = numpy.reshape(result.row_handicaps, (-1, 1))
    result = numpy.block([[result.payoff_matrix, row_handicaps],
                          [col_handicaps, 0.0]])
    result = result.tolist()
    result[-1][-1] = None
    return result
    
@xlwings.func
@xlwings.arg('base_matrix', numpy.array, ndim=2)
@xlwings.arg('row_weights', numpy.array, ndim=1)
@xlwings.arg('col_weights', numpy.array, ndim=1)
@xlwings.ret(expand='table')
def multiplicative_balance(base_matrix, row_weights = None, col_weights = None, value = 1.0):
    return compute_balance_result(zerosum.balance.MultiplicativeBalance, base_matrix, row_weights, col_weights, value)
    
@xlwings.func
@xlwings.arg('base_matrix', numpy.array, ndim=2)
@xlwings.arg('row_weights', numpy.array, ndim=1)
@xlwings.arg('col_weights', numpy.array, ndim=1)
@xlwings.ret(expand='table')
def logistic_non_symmetric_balance(base_matrix, max_payoff, row_weights = None, col_weights = None, value = None):
    return compute_balance_result(zerosum.balance.LogisticNonSymmetricBalance, base_matrix, max_payoff, row_weights, col_weights, value)

@xlwings.func
@xlwings.arg('base_matrix', numpy.array, ndim=2)
@xlwings.arg('strategy_weights', numpy.array, ndim=1)
@xlwings.ret(expand='table')
def logistic_symmetric_balance(base_matrix, strategy_weights = None):
    return compute_balance_result(zerosum.balance.LogisticSymmetricBalance, base_matrix, strategy_weights)

