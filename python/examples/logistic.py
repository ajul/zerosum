import _initpath
import zerosum.balance
from dataset.matchup import ssf2t

init = ssf2t.sortedBySum()

balance = zerosum.balance.LogisticSymmetricBalance(init.data)

result = balance.optimize(check_derivative_epsilon = None)

print(result)
