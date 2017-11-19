import _initpath
import numpy
import zerosum.nash
import matplotlib
import matplotlib.pyplot as plt

def minimizer_advantage(a, b):
    b_power_a = numpy.power(b, a) # max_handicap
    denom = b - b_power_a + a * b_power_a * numpy.log(b)
    p = (b - b_power_a) / denom
    v = b_power_a * (b - 1.0) / denom 
    return p, v

def maximizer_advantage(a, b):
    b_power_a = numpy.power(b, a) # max_handicap
    denom = b - b_power_a + a * b * numpy.log(b)
    p = (b - b_power_a) / denom
    v = b * (b - 1.0) / denom 
    return p, v

# verification code
"""
n = 1024
row_frac = 0.5
col_frac = 1.0
max_payoff = 2.0

x = numpy.linspace(0.0, 1.0, n)
costs = numpy.power(max_payoff, x)

payoff_matrix = costs[None, :] / costs[:, None]
payoff_matrix[numpy.tril_indices(n)] *= max_payoff

n_row = int(n * row_frac)
n_col = int(n * col_frac)

row_result, col_result = zerosum.nash.nash(payoff_matrix[:n_row, :n_col])

if row_frac < col_frac:
    #c = max_payoff - numpy.power(max_payoff, row_frac)
    #denom = (c + row_frac * numpy.power(max_payoff, row_frac) * numpy.log(max_payoff))
    #p = c / denom
    #v = numpy.power(max_payoff, row_frac) * (max_payoff - 1.0) / denom
    p, v = minimizer_advantage(row_frac, max_payoff)
else:
    #c = max_payoff - numpy.power(max_payoff, col_frac)
    #denom = (c + col_frac * max_payoff * numpy.log(max_payoff))
    #p = c / denom
    #v = max_payoff * (max_payoff - 1.0) / denom
    p, v = maximizer_advantage(col_frac, max_payoff)

print(p, max(row_result.strategy), max(col_result.strategy))
print(v, row_result.value)

plt.plot(x[:n_row], row_result.strategy, x[:n_col], col_result.strategy)
plt.show()

"""
