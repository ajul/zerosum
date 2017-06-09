import numpy
import zerosum.balance
from dataset.nonsymmetric import pokemonTypeChart6

balance = zerosum.balance.MultiplicativeBalance(pokemonTypeChart6.data)

balance.check_jacobian()

result = balance.optimize()

print(result)
