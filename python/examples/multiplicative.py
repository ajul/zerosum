import _initpath
import numpy
import zerosum.balance
from dataset.nonsymmetric import pokemonTypeChart6

balance = zerosum.balance.MultiplicativeBalance(pokemonTypeChart6.data)

result = balance.optimize(check_jacobian_epsilon = True)

print(result)
