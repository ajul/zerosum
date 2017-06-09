import zerosum.balance
from dataset.matchup import ssf2t

init = ssf2t.sortedBySum()

balance = zerosum.balance.LogisticSymmetricBalance(init.data)

balance.check_jacobian()

result = balance.optimize()

print(result)
