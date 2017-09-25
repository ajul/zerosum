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
def zerosum_nash(payoff_matrix):
    """
    Computes Nash equilbrium strategy weights and value.
    Result format:
    
    row value                col value (should be negative of row value)
    blank                    blank
    row strategy 0 weight    col strategy 0 weight
    row strategy 1 weight    col strategy 1 weight
    ...                      ...
    """
    row_result, col_result = zerosum.nash.nash(payoff_matrix)
    
    max_strategy_length = max(row_result.strategy.size, col_result.strategy.size)
    row_strategy = numpy.zeros((max_strategy_length, 1))
    row_strategy[:row_result.strategy.size, 0] = row_result.strategy
    col_strategy = numpy.zeros((max_strategy_length, 1))
    col_strategy[:col_result.strategy.size, 0] = col_result.strategy
    
    result = numpy.block(
        [[row_result.value, col_result.value],
         [0.0, 0.0],
         [row_strategy, col_strategy]])
    result = result.tolist()
    # Blank row separating value from weights.
    result[1][0] = None
    result[1][1] = None
    # Blank cells not belonging to a strategy.
    for i in range(row_result.strategy.size + 2, len(result)):
        result[i][0] = None
    for i in range(col_result.strategy.size + 2, len(result)):
        result[i][1] = None
    return result

def compute_balance_result(balance_class, *args, **kwargs):
    """
    Common result format:
            | 1 col           m cols
    --------------------------------------
    1 row   | blank          col_handicaps
    n rows  | row_handicaps  payoff_matrix 
    """
    balance = balance_class(*args, **kwargs)
    result = balance.optimize()
    
    col_handicaps = numpy.reshape(result.col_handicaps, (1, -1))
    row_handicaps = numpy.reshape(result.row_handicaps, (-1, 1))
    result = numpy.block(
        [[0.0, col_handicaps],
         [row_handicaps, result.payoff_matrix]])
    result = result.tolist()
    # Blank the upper-left corner.
    result[0][0] = None
    return result
    
@xlwings.func
@xlwings.arg('base_matrix', numpy.array, ndim=2)
@xlwings.arg('row_weights', numpy.array, ndim=1)
@xlwings.arg('col_weights', numpy.array, ndim=1)
@xlwings.ret(expand='table')
def hazard_non_symmetric_balance(base_matrix, row_weights = None, col_weights = None, value = 0.0):
    return compute_balance_result(zerosum.balance.HazardNonSymmetricBalance, base_matrix, row_weights, col_weights, value)
    
@xlwings.func
@xlwings.arg('base_matrix', numpy.array, ndim=2)
@xlwings.arg('strategy_weights', numpy.array, ndim=1)
@xlwings.ret(expand='table')
def hazard_symmetric_balance(base_matrix, strategy_weights = None):
    return compute_balance_result(zerosum.balance.HazardSymmetricBalance, base_matrix, strategy_weights)
    
@xlwings.func
@xlwings.arg('base_matrix', numpy.array, ndim=2)
@xlwings.arg('row_weights', numpy.array, ndim=1)
@xlwings.arg('col_weights', numpy.array, ndim=1)
@xlwings.ret(expand='table')
def lanchester_non_symmetric_balance(base_matrix, exponent = 1.0, row_weights = None, col_weights = None, value = 0.0):
    return compute_balance_result(zerosum.balance.LanchesterNonSymmetricBalance, base_matrix, exponent, row_weights, col_weights, value)
    
@xlwings.func
@xlwings.arg('base_matrix', numpy.array, ndim=2)
@xlwings.arg('strategy_weights', numpy.array, ndim=1)
@xlwings.ret(expand='table')
def lanchester_symmetric_balance(base_matrix, exponent = 1.0, strategy_weights = None):
    return compute_balance_result(zerosum.balance.LanchesterSymmetricBalance, base_matrix, exponent, strategy_weights)
    
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

@xlwings.func
@xlwings.arg('base_matrix', numpy.array, ndim=2)
@xlwings.arg('row_weights', numpy.array, ndim=1)
@xlwings.arg('col_weights', numpy.array, ndim=1)
@xlwings.ret(expand='table')
def multiplicative_balance(base_matrix, row_weights = None, col_weights = None, value = 1.0):
    return compute_balance_result(zerosum.balance.MultiplicativeBalance, base_matrix, row_weights, col_weights, value)