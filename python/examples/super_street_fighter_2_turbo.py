import _initpath

import numpy
from dataset.matchup import ssf2t
import dataset.csv
import zerosum.balance
import zerosum.nash

init = ssf2t.sorted_by_sum()

# verify initial state

dataset.csv.write_csv('out/ssf2t_init.csv', init.data, init.names)

balance = zerosum.balance.LogisticSymmetricBalance(init.data).optimize()

dataset.csv.write_csv('out/ssf2t_opt.csv', balance.F, init.names)

