import _initpath
import numpy
import zerosum.balance
from dataset.nonsymmetric import pokemon_type_chart_6

balance = zerosum.balance.MultiplicativeBalance(pokemon_type_chart_6.data)

result = balance.optimize(check_derivative_epsilon = None)

print(result)
