import zerosum.balance
from dataset.matchup import ssf2t

init = ssf2t.sortedBySum()

result = zerosum.balance.logisticSymmetric(init.data)

print(result)
