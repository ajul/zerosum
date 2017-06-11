import _initpath
import numpy
import zerosum.balance
import zerosum.nash
from dataset.nonsymmetric import pokemon_type_chart_6

nash_row, nash_col = zerosum.nash.nash(pokemon_type_chart_6.data)

print(nash_row)
print(nash_col)

balance = zerosum.balance.MultiplicativeBalance(pokemon_type_chart_6.data)

result = balance.optimize(check_derivative_epsilon = None)

print(result)
