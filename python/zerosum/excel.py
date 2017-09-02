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
    
